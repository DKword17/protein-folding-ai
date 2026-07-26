#!/usr/bin/env python3
"""
params/weight_optimizer.py
==========================

Optimierung der 5 Rosetta-Energiegewichte durch differentielle Evolution
und bayesianische Verfeinerung gegen eine PDB-Referenzdatenbank.

E_total = w1 * E_LJ  +  w2 * E_Hbond  +  w3 * E_Solvation
        + w4 * E_Rama  +  w5 * E_Repulsive

Ablauf:
    1. Dekoysampling: Für jedes Testprotein werden N Dekoys mit
       Startgewichten erzeugt und zwischengespeichert.
    2. Gewichtsoptimierung: Differentielle Evolution minimiert die
       durchschnittliche RMSD über alle Testproteine.
    3. Bayessche Verfeinerung: Sklearn-GaussianProcess optimiert lokal
       um das gefundene Minimum (optional).

Referenzen:
    - Alford et al. (2017) JCTC 13:3031 — Rosetta energy function
    - Park et al. (2016) JCTC 12:6201 — weight optimization protocol
    - Storn & Price (1997) J Global Optim 11:341 — differential evolution

Autor: Klaus Weber
        Parameter Engineer — Force Field Calibration
Branch: dev/param-klaus
"""

from __future__ import annotations

import json
import math
import random
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import erfc as _erfc

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel

    _SKLEARN_VERFÜGBAR: bool = True
except ImportError:
    _SKLEARN_VERFÜGBAR = False

# ─── Projektimport ────────────────────────────────────────────────────
_PROJEKT_PFAD: Path = Path(__file__).resolve().parent.parent
if str(_PROJEKT_PFAD) not in sys.path:
    sys.path.insert(0, str(_PROJEKT_PFAD))

from folding_engine import (
    AA_CODES,
    Conformation,
    Residue,
    RosettaEnergyFunction,
)

# ─── Konstanten ────────────────────────────────────────────────────────

STANDARD_GEWICHTE: tuple[float, float, float, float, float] = (
    0.50,   # w_lj
    0.75,   # w_hbond
    0.60,   # w_solvation
    0.35,   # w_rama
    1.10,   # w_repulsive
)

GEWICHT_GRENZEN: tuple[tuple[float, float], ...] = (
    (0.05, 3.0),   # w_lj
    (0.05, 3.0),   # w_hbond
    (0.05, 3.0),   # w_solvation
    (0.05, 3.0),   # w_rama
    (0.05, 5.0),   # w_repulsive
)

GEWICHT_NAMEN: tuple[str, str, str, str, str] = (
    "w_lj",
    "w_hbond",
    "w_solvation",
    "w_rama",
    "w_repulsive",
)

_DEKOY_PRO_START: int = 50       # Dekoys pro Protein im ersten Sampling
_MAX_BEWERTUNGEN: int = 5000     # max. Evaluierungen im Optimierer
_BAYES_START_PUNKTE: int = 30     # initiale Stützstellen für GP
_BAYES_ITERATIONEN: int = 20      # bayesianische Optimierungsschritte


# ═══════════════════════════════════════════════════════════════════════
#  1. PDB-Parser
# ═══════════════════════════════════════════════════════════════════════

class PDB_Ladung:
    """Parser für PDB-Dateien → Conformation-Objekt.

    Liest ATOM-Datensätze, extrahiert Rückgrat-Atomkoordinaten
    und baut eine Conformation mit Residuen auf.
    """

    @staticmethod
    def aus_Datei(datei_pfad: str | Path) -> Conformation:
        """Lade eine PDB-Datei und gib eine Conformation zurück.

        Args:
            datei_pfad: Pfad zur .pdb-Datei.

        Returns:
            Conformation mit den geladenen Residuen.

        Raises:
            FileNotFoundError: Wenn die Datei nicht existiert.
            ValueError: Wenn keine ATOM-Datensätze gefunden wurden.
        """
        pfad: Path = Path(datei_pfad)
        if not pfad.exists():
            raise FileNotFoundError(f"PDB-Datei nicht gefunden: {pfad}")

        # ATOM-Datensätze parsen
        atome: list[dict[str, object]] = []
        with pfad.open("r", encoding="utf-8") as fh:
            for zeile in fh:
                if zeile.startswith("ATOM") or zeile.startswith("HETATM"):
                    atom = PDB_Ladung._parse_atom_zeile(zeile)
                    if atom is not None:
                        atome.append(atom)

        if not atome:
            raise ValueError(f"Keine ATOM-Datensätze in {pfad}")

        # Nach Kette und Residuen-ID gruppieren
        residuen_map: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        aa_map: dict[tuple[str, int], str] = {}

        for atom in atome:
            schluessel: tuple[str, int] = (atom["kette"], atom["res_id"])
            name: str = atom["name"]
            coord: np.ndarray = np.array(atom["xyz"], dtype=np.float64)

            if schluessel not in residuen_map:
                residuen_map[schluessel] = {}
                aa_map[schluessel] = atom["res_name"]

            if name in ("N", "CA", "C", "O", "CB"):
                residuen_map[schluessel][name] = coord

        residuen: list[Residue] = []
        for idx, ((kette, res_id), atome_map) in enumerate(
            sorted(residuen_map.items(), key=lambda x: (x[0][0], x[0][1]))
        ):
            drei_buchstaben: str = aa_map[(kette, res_id)]
            res = Residue(seq_index=res_id, aa_code=drei_buchstaben)
            if "N" in atome_map:
                res.N = atome_map["N"]
            if "CA" in atome_map:
                res.CA = atome_map["CA"]
            if "C" in atome_map:
                res.C = atome_map["C"]
            if "O" in atome_map:
                res.O = atome_map["O"]
            if "CB" in atome_map:
                res.CB = atome_map["CB"]
            residuen.append(res)

        konformation: Conformation = Conformation(residues=residuen)
        return konformation

    @staticmethod
    def _parse_atom_zeile(zeile: str) -> Optional[dict[str, object]]:
        try:
            name: str = zeile[12:16].strip()
            res_name: str = zeile[17:20].strip()
            kette: str = zeile[21].strip()
            if not kette:
                kette = "A"
            res_id: int = int(zeile[22:26].strip())
            x: float = float(zeile[30:38].strip())
            y: float = float(zeile[38:46].strip())
            z: float = float(zeile[46:54].strip())
            return {
                "name": name,
                "res_name": res_name,
                "kette": kette,
                "res_id": res_id,
                "xyz": (x, y, z),
            }
        except (ValueError, IndexError):
            return None

    @staticmethod
    def schreibe_datei(konformation: Conformation, datei_pfad: str | Path) -> None:
        pfad: Path = Path(datei_pfad)
        with pfad.open("w", encoding="utf-8") as fh:
            fh.write("REMARK   Generated by weight_optimizer.py (Klaus Weber)\n")
            atom_zähler: int = 1
            for res in konformation.residues:
                for atom_name, coord in [
                    ("N", res.N),
                    ("CA", res.CA),
                    ("C", res.C),
                    ("O", res.O),
                    ("CB", res.CB),
                ]:
                    x, y, z = coord
                    fh.write(
                        f"ATOM  {atom_zähler:>5} {atom_name:<4} "
                        f"{res.aa_code:<3} A{res.seq_index:>4}    "
                        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00\n"
                    )
                    atom_zähler += 1
            fh.write("END\n")


# ═══════════════════════════════════════════════════════════════════════
#  2. RMSD-Berechnung (Kabsch-Algorithmus)
# ═══════════════════════════════════════════════════════════════════════

class RMSD_Berechnung:
    """Berechne das RMSD zwischen zwei Konformationen mit Kabsch-Superposition."""

    _EPS: float = 1e-10

    @classmethod
    def berechne(cls, referenz: Conformation, modell: Conformation) -> float:
        n_referenz: int = len(referenz.residues)
        n_modell: int = len(modell.residues)
        if n_referenz != n_modell:
            return float("inf")

        ref_xyz: list[np.ndarray] = []
        mod_xyz: list[np.ndarray] = []

        for i in range(n_referenz):
            r_ref: np.ndarray = referenz.residues[i].CA.copy()
            r_mod: np.ndarray = modell.residues[i].CA.copy()
            if np.linalg.norm(r_ref) < cls._EPS:
                r_ref = (referenz.residues[i].N + referenz.residues[i].C) / 2.0
            if np.linalg.norm(r_mod) < cls._EPS:
                r_mod = (modell.residues[i].N + modell.residues[i].C) / 2.0
            ref_xyz.append(r_ref)
            mod_xyz.append(r_mod)

        A: np.ndarray = np.array(ref_xyz)
        B: np.ndarray = np.array(mod_xyz)
        return cls._kabsch_rmsd(A, B)

    @classmethod
    def _kabsch_rmsd(cls, A: np.ndarray, B: np.ndarray) -> float:
        zentrum_A: np.ndarray = A.mean(axis=0)
        zentrum_B: np.ndarray = B.mean(axis=0)
        A_z: np.ndarray = A - zentrum_A
        B_z: np.ndarray = B - zentrum_B
        H: np.ndarray = B_z.T @ A_z
        V, S, Wt = np.linalg.svd(H)
        d: float = np.linalg.det(V @ Wt)
        if d < 0:
            V[:, -1] *= -1.0
        R: np.ndarray = V @ Wt
        B_rotiert: np.ndarray = B_z @ R.T
        n: int = A.shape[0]
        quadrat_summe: float = np.sum((A_z - B_rotiert) ** 2)
        rmsd: float = math.sqrt(quadrat_summe / n)
        return rmsd


# ═══════════════════════════════════════════════════════════════════════
#  3. Dekoy-Sammlung und Bewertung
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DekoyEintrag:
    rmsd_zum_nativ: float
    komponenten: dict[str, float]
    sequenz: str


@dataclass
class ProteinDatensatz:
    name: str
    sequenz: str
    nativ_pfad: Optional[Path]
    native_konf: Optional[Conformation]
    dekoys: list[DekoyEintrag] = field(default_factory=list)


def _energie_komponenten_aus(
    gewichte: tuple[float, float, float, float, float],
    konformation: Conformation,
    energie_funktion: RosettaEnergyFunction,
) -> dict[str, float]:
    orig: tuple[float, float, float, float, float] = (
        energie_funktion.w_lj, energie_funktion.w_hbond,
        energie_funktion.w_solvation, energie_funktion.w_rama,
        energie_funktion.w_repulsive,
    )
    energie_funktion.w_lj = 1.0
    energie_funktion.w_hbond = 1.0
    energie_funktion.w_solvation = 1.0
    energie_funktion.w_rama = 1.0
    energie_funktion.w_repulsive = 1.0
    energie_funktion.evaluate(konformation)
    komponenten: dict[str, float] = dict(konformation.energy_components)
    (energie_funktion.w_lj, energie_funktion.w_hbond,
     energie_funktion.w_solvation, energie_funktion.w_rama,
     energie_funktion.w_repulsive) = orig
    return komponenten


def _gewichtete_energie(
    gewichte: tuple[float, float, float, float, float],
    komponenten: dict[str, float],
) -> float:
    (w_lj, w_hbond, w_solv, w_rama, w_rep) = gewichte
    return (
        w_lj * komponenten.get("lennard_jones", 0.0)
        + w_hbond * komponenten.get("hydrogen_bond", 0.0)
        + w_solv * komponenten.get("solvation", 0.0)
        + w_rama * komponenten.get("ramachandran", 0.0)
        + w_rep * komponenten.get("repulsive", 0.0)
    )


# ═══════════════════════════════════════════════════════════════════════
#  4. GewichtsOptimierer
# ═══════════════════════════════════════════════════════════════════════

class GewichtsOptimierer:
    """Optimiere die 5 Rosetta-Energiegewichte gegen eine PDB-Datenbank."""

    def __init__(
        self,
        dekoys_protein: int = _DEKOY_PRO_START,
        zufalls_saat: int = 42,
    ) -> None:
        self.dekoys_protein: int = dekoys_protein
        self.zufalls_saat: int = zufalls_saat
        random.seed(zufalls_saat)
        np.random.seed(zufalls_saat)
        self.proteine: list[ProteinDatensatz] = []
        self.energie_funktion: RosettaEnergyFunction = RosettaEnergyFunction()
        self.beste_gewichte: Optional[tuple[float, float, float, float, float]] = None
        self.beste_bewertung: float = float("inf")
        self.verlauf: list[dict[str, object]] = []

    def lade_testproteine(
        self,
        pdb_dateien: list[str | Path],
        sequenzen: Optional[list[str]] = None,
    ) -> None:
        for idx, pfad_str in enumerate(pdb_dateien):
            pfad: Path = Path(pfad_str)
            try:
                native: Conformation = PDB_Ladung.aus_Datei(pfad)
                name: str = pfad.stem
                rueck_map: dict[str, str] = {v: k for k, v in AA_CODES.items()}
                seq_roh: str = ""
                for res in native.residues:
                    seq_roh += rueck_map.get(res.aa_code, "X")
                sequenz: str = sequenzen[idx] if sequenzen else seq_roh
                datensatz: ProteinDatensatz = ProteinDatensatz(
                    name=name, sequenz=sequenz, nativ_pfad=pfad, native_konf=native,
                )
                self.proteine.append(datensatz)
                print(f"  [GewichtsOptimierer] Geladen: {name} "
                      f"({len(native.residues)} Residuen)")
            except (FileNotFoundError, ValueError) as fehler:
                print(f"  [GewichtsOptimierer] Warnung: Kann {pfad_str} nicht laden: {fehler}")

    def setze_testproteine_direkt(self, sequenzen: list[str]) -> None:
        from folding_engine import fold_protein
        for idx, sequenz in enumerate(sequenzen):
            name: str = f"test_protein_{idx + 1}"
            print(f"  [GewichtsOptimierer] Simuliere native Struktur für: {name}")
            native_hilf: Conformation = fold_protein(sequence=sequenz, n_steps=5000)
            datensatz: ProteinDatensatz = ProteinDatensatz(
                name=name, sequenz=sequenz, nativ_pfad=None, native_konf=native_hilf,
            )
            self.proteine.append(datensatz)

    def sample_dekoys(self, schritte: int = 10000) -> None:
        from folding_engine import fold_protein
        for protein in self.proteine:
            print(f"  [Sampling] Generiere {self.dekoys_protein} Dekoys "
                  f"für {protein.name} ({schritte} Schritte)...")
            protein.dekoys.clear()
            for d_idx in range(self.dekoys_protein):
                konformation: Conformation = fold_protein(
                    sequence=protein.sequenz, n_steps=schritte,
                )
                rmsd: float = RMSD_Berechnung.berechne(
                    protein.native_konf, konformation
                ) if protein.native_konf is not None else float("inf")
                komponenten: dict[str, float] = _energie_komponenten_aus(
                    STANDARD_GEWICHTE, konformation, self.energie_funktion
                )
                protein.dekoys.append(DekoyEintrag(
                    rmsd_zum_nativ=rmsd, komponenten=komponenten, sequenz=protein.sequenz,
                ))
            rmsds: list[float] = [d.rmsd_zum_nativ for d in protein.dekoys
                                   if d.rmsd_zum_nativ < float("inf")]
            if rmsds:
                print(f"    RMSD: {min(rmsds):.2f}–{max(rmsds):.2f} Å, "
                      f"mean: {sum(rmsds)/len(rmsds):.2f} Å")

    def _zielfunktion(self, gewichte_array: np.ndarray) -> float:
        gewichte: tuple[float, float, float, float, float] = tuple(
            float(gewichte_array[i]) for i in range(5)
        )
        gesamt_rmsd: float = 0.0
        anzahl: int = 0
        for protein in self.proteine:
            if not protein.dekoys:
                continue
            energie_rmsd_liste: list[tuple[float, float]] = []
            for de in protein.dekoys:
                e = _gewichtete_energie(gewichte, de.komponenten)
                energie_rmsd_liste.append((e, de.rmsd_zum_nativ))
            energie_rmsd_liste.sort(key=lambda x: x[0])
            top_n: int = min(5, len(energie_rmsd_liste))
            if top_n > 0:
                mittel_rmsd: float = sum(
                    r for _, r in energie_rmsd_liste[:top_n]
                ) / top_n
                gesamt_rmsd += mittel_rmsd
                anzahl += 1
        return 1e10 if anzahl == 0 else gesamt_rmsd / anzahl

    def _differentielle_evolution(self) -> tuple[np.ndarray, float]:
        print("\n  [Optimierung] Starte differentielle Evolution...")
        grenzen: list[tuple[float, float]] = list(GEWICHT_GRENZEN)
        resultat = differential_evolution(
            func=self._zielfunktion, bounds=grenzen,
            maxiter=100, popsize=30, tol=0.01,
            mutation=(0.5, 1.0), recombination=0.7,
            seed=self.zufalls_saat, disp=True, workers=1,
        )
        print(f"  [Optimierung] DE abgeschlossen. Best: {resultat.fun:.4f} Å")
        self.verlauf.append({
            "methode": "differential_evolution",
            "beste_gewichte": resultat.x.tolist(),
            "beste_bewertung": float(resultat.fun),
            "konvergenz": resultat.success,
        })
        return resultat.x, float(resultat.fun)

    def _bayesianische_verfeinerung(
        self, start_gewichte: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        if not _SKLEARN_VERFÜGBAR:
            print("  [Bayesianisch] sklearn nicht verfügbar. Überspringe.")
            return start_gewichte, self._zielfunktion(start_gewichte)
        print("\n  [Bayesianisch] Starte bayesianische Verfeinerung...")
        n_dim: int = len(start_gewichte)
        X_train: list[np.ndarray] = []
        y_train: list[float] = []
        for _ in range(_BAYES_START_PUNKTE):
            probe: np.ndarray = start_gewichte.copy()
            for i in range(n_dim):
                spanne: float = GEWICHT_GRENZEN[i][1] - GEWICHT_GRENZEN[i][0]
                probe[i] = np.clip(
                    probe[i] + np.random.uniform(-0.15, 0.15) * spanne,
                    GEWICHT_GRENZEN[i][0], GEWICHT_GRENZEN[i][1],
                )
            X_train.append(probe)
            y_train.append(self._zielfunktion(probe))
        X_arr: np.ndarray = np.array(X_train)
        y_arr: np.ndarray = np.array(y_train)
        kernel = Matern(length_scale=np.ones(n_dim), nu=2.5) + WhiteKernel(noise_level=0.01)
        gp: GaussianProcessRegressor = GaussianProcessRegressor(
            kernel=kernel, alpha=0.0, normalize_y=True, random_state=self.zufalls_saat,
        )
        beste_x: np.ndarray = start_gewichte.copy()
        beste_y: float = min(y_train)
        for iteration in range(_BAYES_ITERATIONEN):
            gp.fit(X_arr, y_arr)
            kandidaten: list[np.ndarray] = [
                np.array([np.random.uniform(GEWICHT_GRENZEN[i][0], GEWICHT_GRENZEN[i][1])
                          for i in range(n_dim)])
                for _ in range(500)
            ]
            kand_arr: np.ndarray = np.array(kandidaten)
            mu, sigma = gp.predict(kand_arr, return_std=True)
            sigma = np.maximum(sigma, 1e-10)
            gamma: np.ndarray = (beste_y - mu) / sigma
            ei: np.ndarray = sigma * (
                gamma * (0.5 * _erfc(-gamma / math.sqrt(2)))
                + (1.0 / math.sqrt(2 * math.pi)) * np.exp(-0.5 * gamma ** 2)
            )
            ei = np.where(np.isfinite(ei), ei, 0.0)
            bester_idx: int = int(np.argmax(ei))
            naechste_x: np.ndarray = kand_arr[bester_idx]
            naechste_y: float = self._zielfunktion(naechste_x)
            X_arr = np.vstack([X_arr, naechste_x])
            y_arr = np.append(y_arr, naechste_y)
            if naechste_y < beste_y:
                beste_x = naechste_x.copy()
                beste_y = naechste_y
            if (iteration + 1) % 5 == 0:
                print(f"    Iter {iteration+1}/{_BAYES_ITERATIONEN}: best = {beste_y:.4f}")
        self.verlauf.append({
            "methode": "bayesianische_verfeinerung",
            "beste_gewichte": beste_x.tolist(),
            "beste_bewertung": float(beste_y),
        })
        print(f"  [Bayesianisch] Abgeschlossen. Best: {beste_y:.4f}")
        return beste_x, beste_y

    def optimiere(
        self, methode: str = "de", verfeinern: bool = True,
    ) -> "OptimierungsErgebnis":
        if not self.proteine:
            raise RuntimeError("Keine Testproteine geladen.")
        if not any(p.dekoys for p in self.proteine):
            print("  [GewichtsOptimierer] Keine Dekoys vorhanden. Starte Dekoy-Sampling...")
            self.sample_dekoys()
        beste_x, beste_y = (
            self._differentielle_evolution() if methode == "de"
            else self._zufallssuche()
        )
        if verfeinern:
            beste_x, beste_y = self._bayesianische_verfeinerung(beste_x)
        self.beste_gewichte = tuple(float(beste_x[i]) for i in range(5))
        self.beste_bewertung = beste_y
        detail: dict[str, dict[str, float]] = {}
        for protein in self.proteine:
            if not protein.dekoys:
                continue
            energie_rmsd = sorted(
                [(_gewichtete_energie(self.beste_gewichte, de.komponenten), de.rmsd_zum_nativ)
                 for de in protein.dekoys],
                key=lambda x: x[0],
            )
            top5_rmsd = sum(r for _, r in energie_rmsd[:5]) / 5
            best_rmsd = energie_rmsd[0][1] if energie_rmsd else float("inf")
            detail[protein.name] = {
                "top5_avg_rmsd": round(top5_rmsd, 4),
                "best_rmsd": round(best_rmsd, 4),
                "dekoys": len(protein.dekoys),
            }
        konfidenz: tuple[float, float] = self._berechne_konfidenz()
        return OptimierungsErgebnis(
            beste_gewichte=self.beste_gewichte,
            durchschnittliche_rmsd=beste_y,
            konfidenz_intervall=konfidenz,
            detail_pro_protein=detail,
            verlauf=list(self.verlauf),
        )

    def _zufallssuche(self, versuche: int = 500) -> tuple[np.ndarray, float]:
        print(f"\n  [Zufallssuche] Starte {versuche} Versuche...")
        beste_x = np.array(STANDARD_GEWICHTE)
        beste_y = float("inf")
        for i in range(versuche):
            probe = np.array([np.random.uniform(GEWICHT_GRENZEN[j][0], GEWICHT_GRENZEN[j][1])
                              for j in range(5)])
            wert = self._zielfunktion(probe)
            if wert < beste_y:
                beste_y, beste_x = wert, probe.copy()
            if (i+1) % 100 == 0:
                print(f"    Versuch {i+1}/{versuche}: best = {beste_y:.4f}")
        self.verlauf.append({
            "methode": "zufallssuche", "versuche": versuche,
            "beste_gewichte": beste_x.tolist(), "beste_bewertung": float(beste_y),
        })
        return beste_x, beste_y

    def _berechne_konfidenz(self, bootstrap_n: int = 200) -> tuple[float, float]:
        if not self.proteine:
            return (0.0, 0.0)
        rmsd_proteine: list[float] = []
        for protein in self.proteine:
            if not protein.dekoys:
                continue
            energie_rmsd = sorted(
                [(_gewichtete_energie(self.beste_gewichte, de.komponenten), de.rmsd_zum_nativ)
                 for de in protein.dekoys],
                key=lambda x: x[0],
            )
            rmsd_proteine.append(sum(r for _, r in energie_rmsd[:5]) / 5)
        if not rmsd_proteine:
            return (0.0, 0.0)
        bootstrap_mittel: list[float] = []
        n_prot: int = len(rmsd_proteine)
        for _ in range(bootstrap_n):
            stichprobe: np.ndarray = np.random.choice(rmsd_proteine, size=n_prot, replace=True)
            bootstrap_mittel.append(float(np.mean(stichprobe)))
        bootstrap_mittel.sort()
        alpha: float = 0.05
        return (round(bootstrap_mittel[int(bootstrap_n * alpha / 2)], 4),
                round(bootstrap_mittel[int(bootstrap_n * (1 - alpha / 2))], 4))


# ═══════════════════════════════════════════════════════════════════════
#  5. Ergebnisdatenklasse
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OptimierungsErgebnis:
    beste_gewichte: tuple[float, float, float, float, float]
    durchschnittliche_rmsd: float
    konfidenz_intervall: tuple[float, float]
    detail_pro_protein: dict[str, dict[str, float]]
    verlauf: list[dict[str, object]]

    def ausgabe(self) -> str:
        zeilen: list[str] = [
            "\n" + "=" * 60,
            "  OPTIMIERUNGSERGEBNIS — GewichtsOptimierer",
            "=" * 60, "",
            f"  Durchschnittliches RMSD: {self.durchschnittliche_rmsd:.4f} Å",
            f"  95%-Konfidenzintervall: [{self.konfidenz_intervall[0]:.4f}, {self.konfidenz_intervall[1]:.4f}] Å",
            "", "  Optimierte Gewichte:",
        ]
        for name, wert in zip(GEWICHT_NAMEN, self.beste_gewichte):
            standard = STANDARD_GEWICHTE[
                ["w_lj","w_hbond","w_solvation","w_rama","w_repulsive"].index(name)
            ]
            zeilen.append(f"    {name:15s} = {wert:.4f}  (REF2015: {standard:.4f}, Δ = {wert-standard:+.4f})")
        zeilen.append("")
        for prot_name, metrik in self.detail_pro_protein.items():
            zeilen.append(f"    {prot_name:25s}: Top5-RMSD = {metrik['top5_avg_rmsd']:.2f} Å, "
                          f"Best-RMSD = {metrik['best_rmsd']:.2f} Å ({metrik['dekoys']} Dekoys)")
        if self.verlauf:
            zeilen.append(f"\n  Letzte Methode: {self.verlauf[-1].get('methode', '?')}")
        zeilen.append("=" * 60)
        return "\n".join(zeilen)

    def to_dict(self) -> dict[str, object]:
        return {
            "beste_gewichte": {name: round(w, 6) for name, w in zip(GEWICHT_NAMEN, self.beste_gewichte)},
            "durchschnittliche_rmsd_angstroem": round(self.durchschnittliche_rmsd, 4),
            "konfidenz_intervall_95": [round(self.konfidenz_intervall[0], 4), round(self.konfidenz_intervall[1], 4)],
            "detail_pro_protein": self.detail_pro_protein,
            "verlauf": self.verlauf,
        }

    def speichere_json(self, datei_pfad: str | Path) -> None:
        pfad: Path = Path(datei_pfad)
        with pfad.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        print(f"  [Ergebnis] Gespeichert: {pfad}")


# ═══════════════════════════════════════════════════════════════════════
#  6. Hauptprogramm (Demo / Selbsttest)
# ═══════════════════════════════════════════════════════════════════════

def _simuliere_optimierung() -> None:
    print("\n" + "=" * 60)
    print("  GEWICHTSOPTIMIERUNG — Simulation")
    print("=" * 60)
    optimierer: GewichtsOptimierer = GewichtsOptimierer(dekoys_protein=20, zufalls_saat=42)
    print("\n  [Setup] Initialisiere Testproteine...")
    optimierer.setze_testproteine_direkt(["AAAAK", "AACAA", "ALALA"])
    optimierer.sample_dekoys(schritte=2000)
    ergebnis: OptimierungsErgebnis = optimierer.optimiere(
        methode="de", verfeinern=_SKLEARN_VERFÜGBAR,
    )
    print(ergebnis.ausgabe())
    ausgabe_pfad: Path = _PROJEKT_PFAD / "params" / "optimierung_ergebnis.json"
    ergebnis.speichere_json(ausgabe_pfad)

if __name__ == "__main__":
    _simuliere_optimierung()
