#!/usr/bin/env python3
"""
params/force_field_calibrator.py
=================================

QM-Kalibrierung der Lennard-Jones-Epsilon-Parameter für das
Rosetta-Kraftfeld.

LJ-Potential:
    E_LJ(r) = eps * [(sigma/r)^12 - 2*(sigma/r)^6]

Kalibrierungsstrategie:
    1. Für jedes Aminosäurepaar wird eine QM-Referenzkurve bereitgestellt.
    2. Der Kalibrator minimiert den RMS-Fehler zwischen
       der LJ-Kurve (mit variablem eps) und der QM-Kurve.
    3. Die Sigma-Werte bleiben fix (aus VDW_RADII).
    4. Bootstrap-Konfidenzintervalle werden berechnet.

Autor: Klaus Weber
        Parameter Engineer — Force Field Calibration
Branch: dev/param-klaus
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import erfc as _erfc

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    _SKLEARN_VERFÜGBAR: bool = True
except ImportError:
    _SKLEARN_VERFÜGBAR = False

_PROJEKT_PFAD: Path = Path(__file__).resolve().parent.parent
if str(_PROJEKT_PFAD) not in sys.path:
    sys.path.insert(0, str(_PROJEKT_PFAD))

from folding_engine import VDW_RADII

STANDARD_EPSILON: dict[tuple[str, str], float] = {
    ("ALA", "ALA"): 0.15, ("LEU", "LEU"): 0.28, ("VAL", "VAL"): 0.22,
    ("ILE", "ILE"): 0.26, ("PHE", "PHE"): 0.35, ("TRP", "TRP"): 0.40,
    ("TYR", "TYR"): 0.38, ("GLY", "GLY"): 0.10,
}

_ALLE_AA: tuple[str, ...] = (
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS",
    "ILE", "LYS", "LEU", "MET", "ASN", "PRO", "GLN",
    "ARG", "SER", "THR", "VAL", "TRP", "TYR",
)

AMINOSAEURE_PAARE: list[tuple[str, str]] = [
    (aa_i, aa_j) for idx, aa_i in enumerate(_ALLE_AA) for aa_j in _ALLE_AA[idx:]
]

_ABSTANDS_RASTER: np.ndarray = np.arange(2.5, 12.0, 0.25, dtype=np.float64)
_EPSILON_GRENZEN: tuple[float, float] = (0.001, 2.0)
_DE_POPSIZE: int = 25
_DE_MAXITER: int = 200
_BAYES_INITIAL: int = 20
_BAYES_ITERATIONEN: int = 15
_BOOTSTRAP_REPLIKATE: int = 500


def _abstands_gewicht(abstand: np.ndarray) -> np.ndarray:
    return np.exp(-((abstand - 4.0) ** 2) / 8.0) + 0.2


@dataclass
class QMDatenPunkt:
    paar: tuple[str, str]
    abstand: float
    energie_qm: float
    quelle: str = "theoretisch"


@dataclass
class QMReferenzKurve:
    paar: tuple[str, str]
    abstände: np.ndarray
    energien_qm: np.ndarray
    sigma: float
    quelle: str = "theoretisch"

    def __post_init__(self) -> None:
        self.abstände = np.asarray(self.abstände, dtype=np.float64).ravel()
        self.energien_qm = np.asarray(self.energien_qm, dtype=np.float64).ravel()
        if self.abstände.shape != self.energien_qm.shape:
            raise ValueError("Abstände und Energien müssen gleiche Form haben.")

    def __len__(self) -> int:
        return int(self.abstände.shape[0])


def _lj_energie(epsilon: float, sigma: float, abstände: np.ndarray) -> np.ndarray:
    r_inv: np.ndarray = sigma / np.maximum(abstände, 0.1)
    sr6: np.ndarray = r_inv ** 6
    sr12: np.ndarray = sr6 * sr6
    return epsilon * (sr12 - 2.0 * sr6)


class LJKalibrator:
    """Kalibriere LJ-Epsilon-Werte gegen QM-Referenzdaten."""

    def __init__(self, zufalls_saat: int = 42) -> None:
        self.zufalls_saat: int = zufalls_saat
        random.seed(zufalls_saat)
        np.random.seed(zufalls_saat)
        self.kurven: dict[tuple[str, str], QMReferenzKurve] = {}
        self.ergebnisse: dict[tuple[str, str], "KalibrierErgebnis"] = {}
        self.gesamt_protokoll: list[dict[str, object]] = []

    def lade_qm_daten(self, kurven: list[QMReferenzKurve]) -> None:
        for kurve in kurven:
            paar = self._normalisiere_paar(kurve.paar)
            self.kurven[paar] = kurve
        print(f"  [LJKalibrator] {len(self.kurven)} QM-Kurven geladen.")

    def lade_synthetische_daten(
        self, paare: Optional[list[tuple[str, str]]] = None, rausch_niveau: float = 0.02,
    ) -> None:
        if paare is None:
            paare = list(STANDARD_EPSILON.keys())
        for paar in paare:
            paar_norm = self._normalisiere_paar(paar)
            aa1, aa2 = paar_norm
            sigma = (VDW_RADII.get(aa1, 1.8) + VDW_RADII.get(aa2, 1.8)) / 2.0
            eps_wahr = STANDARD_EPSILON.get(paar_norm, 0.15)
            energien_rein = _lj_energie(eps_wahr, sigma, _ABSTANDS_RASTER)
            maximal = float(np.max(np.abs(energien_rein)))
            rauschen = np.random.normal(0.0, maximal * rausch_niveau, _ABSTANDS_RASTER.shape)
            self.kurven[paar_norm] = QMReferenzKurve(
                paar=paar_norm, abstände=_ABSTANDS_RASTER.copy(),
                energien_qm=energien_rein + rauschen, sigma=sigma,
                quelle=f"synthetisch (eps={eps_wahr:.3f}, Rauschen={rausch_niveau:.1%})",
            )
        print(f"  [LJKalibrator] {len(self.kurven)} synthetische Kurven erzeugt.")

    @staticmethod
    def _normalisiere_paar(paar: tuple[str, str]) -> tuple[str, str]:
        aa1, aa2 = paar
        return (aa2, aa1) if aa1 > aa2 else (aa1, aa2)

    def _kalibriere_paar(self, paar: tuple[str, str], methode: str = "de") -> "KalibrierErgebnis":
        kurve = self.kurven[paar]
        sigma, abstände, energien_qm = kurve.sigma, kurve.abstände, kurve.energien_qm
        gewichte = _abstands_gewicht(abstände)
        eps_start = STANDARD_EPSILON.get(paar, 0.15)

        def zielfunktion(x: np.ndarray) -> float:
            eps = float(x[0])
            diff = (_lj_energie(eps, sigma, abstände) - energien_qm) * gewichte
            return math.sqrt(np.mean(diff ** 2)) + 0.001 * (eps - eps_start) ** 2

        if methode == "de":
            res = differential_evolution(zielfunktion, [_EPSILON_GRENZEN],
                                         maxiter=_DE_MAXITER, popsize=_DE_POPSIZE,
                                         tol=1e-6, seed=self.zufalls_saat, disp=False)
            eps_opt, fehler_opt, konvergiert = float(res.x[0]), float(res.fun), bool(res.success)
        elif methode == "scipy":
            res = minimize(zielfunktion, np.array([eps_start]), bounds=[_EPSILON_GRENZEN],
                           method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-8})
            eps_opt, fehler_opt, konvergiert = float(res.x[0]), float(res.fun), bool(res.success)
        else:
            raise ValueError(f"Unbekannte Methode: {methode}")

        konfidenz = self._bootstrap_epsilon(zielfunktion, eps_opt, sigma, abstände, energien_qm, gewichte)
        e_lj_opt = _lj_energie(eps_opt, sigma, abstände)
        rms_orig = math.sqrt(np.mean((e_lj_opt - energien_qm) ** 2))
        max_abs = float(np.max(np.abs(energien_qm)))
        return KalibrierErgebnis(
            paar=paar, sigma=sigma, epsilon_optimiert=round(eps_opt, 6),
            epsilon_referenz=round(STANDARD_EPSILON.get(paar, 0.15), 6),
            rms_fehler_kcal_mol=round(rms_orig, 6),
            relativer_fehler=round(rms_orig / max(max_abs, 1e-10), 6),
            konfidenz_intervall_95=(round(konfidenz[0], 6), round(konfidenz[1], 6)),
            konvergiert=konvergiert, methode=methode, anzahl_datenpunkte=len(kurve),
        )

    def _bootstrap_epsilon(self, zielfunktion, eps_opt, sigma, abstände, energien_qm, gewichte):
        n = len(abstände)
        e_lj_opt = _lj_energie(eps_opt, sigma, abstände)
        residuen = energien_qm - e_lj_opt
        bootstrap_eps = []
        for b in range(_BOOTSTRAP_REPLIKATE):
            idx = np.random.choice(n, size=n, replace=True)
            ab, eb, gb = abstände[idx], energien_qm[idx] + residuen[idx], gewichte[idx]
            def boot_fun(x):
                diff = (_lj_energie(float(x[0]), sigma, ab) - eb) * gb
                return float(np.sqrt(np.mean(diff ** 2)))
            try:
                res = differential_evolution(boot_fun, [_EPSILON_GRENZEN],
                                             maxiter=50, popsize=15, tol=1e-4,
                                             seed=self.zufalls_saat + b, disp=False)
                bootstrap_eps.append(float(res.x[0]))
            except Exception:
                continue
        if not bootstrap_eps:
            return (0.0, 0.0)
        bootstrap_eps.sort()
        alpha = 0.05
        return (bootstrap_eps[max(0, int(_BOOTSTRAP_REPLIKATE * alpha / 2))],
                bootstrap_eps[min(len(bootstrap_eps)-1, int(_BOOTSTRAP_REPLIKATE * (1 - alpha / 2)))])

    def kalibriere(self, methode: str = "de", ausgabe_intervall: int = 5) -> "KalibrationsErgebnis":
        if not self.kurven:
            raise RuntimeError("Keine QM-Daten geladen.")
        print(f"\n  [LJKalibrator] Starte Kalibrierung von {len(self.kurven)} Paaren...")
        einzel = {}
        for idx, paar in enumerate(sorted(self.kurven.keys())):
            ergebnis = self._kalibriere_paar(paar, methode=methode)
            einzel[paar] = ergebnis
            self.ergebnisse[paar] = ergebnis
            if (idx + 1) % ausgabe_intervall == 0 or idx == len(self.kurven) - 1:
                print(f"    [{idx+1}/{len(self.kurven)}] {paar[0]}-{paar[1]}: "
                      f"eps = {ergebnis.epsilon_optimiert:.4f} "
                      f"(Ref: {ergebnis.epsilon_referenz:.4f}), "
                      f"RMS = {ergebnis.rms_fehler_kcal_mol:.4f}")
        rms_liste = [e.rms_fehler_kcal_mol for e in einzel.values()]
        return KalibrationsErgebnis(
            einzel_ergebnisse=einzel,
            durchschnittlicher_rms_fehler=round(sum(rms_liste)/len(rms_liste), 6),
            maximaler_rms_fehler=round(max(rms_liste), 6),
            anzahl_kalibriert=len(einzel),
        )

    def bayesianische_nachoptimierung(self, paar: tuple[str, str], iterationen: int = _BAYES_ITERATIONEN) -> "KalibrierErgebnis":
        if not _SKLEARN_VERFÜGBAR:
            raise ImportError("sklearn erforderlich.")
        if paar not in self.ergebnisse:
            self.ergebnisse[paar] = self._kalibriere_paar(paar)
        kurve = self.kurven[paar]
        sigma, abstände, energien_qm = kurve.sigma, kurve.abstände, kurve.energien_qm
        gewichte = _abstands_gewicht(abstände)
        X_obs, y_obs = [], []
        eps_init = self.ergebnisse[paar].epsilon_optimiert
        for _ in range(_BAYES_INITIAL):
            probe = np.clip(eps_init + np.random.normal(0, eps_init*0.1),
                            _EPSILON_GRENZEN[0], _EPSILON_GRENZEN[1])
            diff = (_lj_energie(probe, sigma, abstände) - energien_qm) * gewichte
            X_obs.append(probe); y_obs.append(math.sqrt(np.mean(diff ** 2)))
        X_arr = np.array(X_obs).reshape(-1, 1)
        y_arr = np.array(y_obs)
        kernel = Matern(length_scale=0.1, nu=2.5) + WhiteKernel(noise_level=1e-4)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True,
                                       random_state=self.zufalls_saat)
        beste_eps, beste_y = eps_init, min(y_obs)
        for it in range(iterationen):
            gp.fit(X_arr, y_arr)
            kandidaten = np.linspace(_EPSILON_GRENZEN[0], _EPSILON_GRENZEN[1], 1000).reshape(-1, 1)
            mu, sgp = gp.predict(kandidaten, return_std=True)
            sgp = np.maximum(sgp, 1e-10)
            gamma = (beste_y - mu.ravel()) / sgp.ravel()
            ei = sgp.ravel() * (gamma * 0.5 * _erfc(-gamma/math.sqrt(2))
                                + np.exp(-0.5*gamma**2)/math.sqrt(2*math.pi))
            ei = np.where(np.isfinite(ei), ei, 0.0)
            next_eps = float(kandidaten[int(np.argmax(ei)), 0])
            diff = (_lj_energie(next_eps, sigma, abstände) - energien_qm) * gewichte
            y_new = math.sqrt(np.mean(diff ** 2))
            X_arr = np.vstack([X_arr, [[next_eps]]])
            y_arr = np.append(y_arr, y_new)
            if y_new < beste_y:
                beste_eps, beste_y = next_eps, y_new
        e_final = _lj_energie(beste_eps, sigma, abstände)
        rms_final = math.sqrt(np.mean((e_final - energien_qm) ** 2))
        max_abs = float(np.max(np.abs(energien_qm)))
        return KalibrierErgebnis(
            paar=paar, sigma=sigma, epsilon_optimiert=round(beste_eps, 6),
            epsilon_referenz=round(STANDARD_EPSILON.get(paar, 0.15), 6),
            rms_fehler_kcal_mol=round(rms_final, 6),
            relativer_fehler=round(rms_final/max(max_abs, 1e-10), 6),
            konfidenz_intervall_95=(0.0, 0.0), konvergiert=True,
            methode="de+bayes", anzahl_datenpunkte=len(kurve),
        )


@dataclass
class KalibrierErgebnis:
    paar: tuple[str, str]
    sigma: float
    epsilon_optimiert: float
    epsilon_referenz: float
    rms_fehler_kcal_mol: float
    relativer_fehler: float
    konfidenz_intervall_95: tuple[float, float]
    konvergiert: bool
    methode: str
    anzahl_datenpunkte: int


@dataclass
class KalibrationsErgebnis:
    einzel_ergebnisse: dict[tuple[str, str], KalibrierErgebnis]
    durchschnittlicher_rms_fehler: float
    maximaler_rms_fehler: float
    anzahl_kalibriert: int

    def ausgabe(self) -> str:
        zeilen = ["\n" + "="*65, "  KALIBRATIONSERGEBNIS — LJ-Kalibrator", "="*65, "",
                  f"  Kalibrierte Paare: {self.anzahl_kalibriert}",
                  f"  RMS-Fehler (Durchschnitt): {self.durchschnittlicher_rms_fehler:.6f} kcal/mol",
                  f"  RMS-Fehler (Maximum):       {self.maximaler_rms_fehler:.6f} kcal/mol",
                  "", "  Ergebnisse pro Paar:",
                  f"  {'Paar':12s} {'Sigma':>7s} {'eps_opt':>9s} {'eps_ref':>9s} {'RMS':>9s} {'95%-KI':>20s}",
                  "  " + "-"*65]
        for paar in sorted(self.einzel_ergebnisse.keys()):
            e = self.einzel_ergebnisse[paar]
            ki = f"[{e.konfidenz_intervall_95[0]:.4f}, {e.konfidenz_intervall_95[1]:.4f}]"
            zeilen.append(f"  {paar[0]:4s}-{paar[1]:4s}  {e.sigma:>7.3f}  {e.epsilon_optimiert:>9.4f}  "
                          f"{e.epsilon_referenz:>9.4f}  {e.rms_fehler_kcal_mol:>9.6f}  {ki:>20s}")
        zeilen.append("  " + "-"*65 + "\n" + "="*65)
        return "\n".join(zeilen)

    def to_dict(self) -> dict[str, object]:
        paare = {}
        for paar, e in self.einzel_ergebnisse.items():
            paare[f"{paar[0]}_{paar[1]}"] = {
                "paar": list(paar), "sigma_angstroem": e.sigma,
                "epsilon_optimiert_kcal_mol": e.epsilon_optimiert,
                "epsilon_referenz_kcal_mol": e.epsilon_referenz,
                "rms_fehler_kcal_mol": e.rms_fehler_kcal_mol,
                "konfidenz_intervall_95": list(e.konfidenz_intervall_95),
                "konvergiert": e.konvergiert, "methode": e.methode,
            }
        return {"anzahl_kalibriert": self.anzahl_kalibriert,
                "durchschnittlicher_rms_fehler_kcal_mol": self.durchschnittlicher_rms_fehler,
                "maximaler_rms_fehler_kcal_mol": self.maximaler_rms_fehler,
                "einzel_ergebnisse": paare}

    def speichere_json(self, datei_pfad: str | Path) -> None:
        with Path(datei_pfad).open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        print(f"  [Kalibrator] Ergebnis gespeichert: {datei_pfad}")


def _simuliere_kalibrierung() -> None:
    print("\n" + "="*65, "\n  LJ-EPSILON-KALIBRIERUNG — Simulation", "="*65)
    kalibrator = LJKalibrator(zufalls_saat=42)
    print("\n  [Setup] Erzeuge synthetische QM-Referenzdaten...")
    kalibrator.lade_synthetische_daten(
        paare=[("ALA","ALA"),("LEU","LEU"),("VAL","VAL"),("PHE","PHE"),("GLY","GLY")],
        rausch_niveau=0.03,
    )
    ergebnis = kalibrator.kalibriere(methode="de", ausgabe_intervall=2)
    print(ergebnis.ausgabe())
    ergebnis.speichere_json(_PROJEKT_PFAD / "params" / "lj_kalibrierung_ergebnis.json")
    if _SKLEARN_VERFÜGBAR:
        print("\n  [Bayes] Verfeinere ALA-ALA...")
        try:
            v = kalibrator.bayesianische_nachoptimierung(("ALA","ALA"), iterationen=10)
            print(f"    Vorher: eps={ergebnis.einzel_ergebnisse[('ALA','ALA')].epsilon_optimiert:.4f}")
            print(f"    Nachher: eps={v.epsilon_optimiert:.4f} (RMS={v.rms_fehler_kcal_mol:.6f})")
        except Exception as fe:
            print(f"    Bayes-Fehler: {fe}")

if __name__ == "__main__":
    _simuliere_kalibrierung()
