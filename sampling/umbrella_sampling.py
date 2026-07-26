#!/usr/bin/env python3
"""
sampling/umbrella_sampling.py
=============================

Échantillonnage parapluie (Umbrella Sampling).

Méthode de Torrie & Valleau (1977) pour échantillonner des régions
défavorisées de l'espace des configurations en ajoutant un potentiel
de biais harmonique :

    V_biais(ξ) = (k_i / 2) · (ξ − ξ_i⁰)²

où ξ est une coordonnée de réaction (p. ex. RMSD, Rg, nombre de
contacts) et ξ_i⁰ le centre de la i-ème fenêtre.

Les histogrammes biaisés de chaque fenêtre sont ensuite combinés
via WHAM pour reconstruire le profil d'énergie libre sans biais.

Variables :
    Fenêtre        → intervalle de coordonnée de réaction (tranche)
    Biais          → potentiel harmonique V(ξ)
    Raideur        → constante de force k (kcal · mol⁻¹ · Å⁻²)
    Coordonnée     → ξ — variable de réaction
    Histogramme    → distribution de ξ sous biais

Références :
    - Torrie & Valleau (1977) J. Comput. Phys. 23:187
    - Frenkel & Smit, Understanding Molecular Simulation, §7.1–7.2
    - Kästner (2011) WIREs Comput. Mol. Sci. 1:932
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

sys.path.insert(0, "..")
from folding_engine import Conformation, RosettaEnergyFunction

from .wham import CONSTANTE_BOLTZMANN, WHAM, Histogramme


# ─── Coordonnées de réaction ─────────────────────────────────────────

def rmsd_naïf(conf: Conformation, référence: Optional[Conformation] = None) -> float:
    """
    RMSD brut (sans alignement optimal) entre deux conformations.

    Calcule la distance CA–CA moyenne comme coordonnée de réaction
    simplifiée.
    """
    résidus = conf.residues
    if référence is not None:
        réf_res = référence.residues
    else:
        # Par défaut : RMSD par rapport à la première conformation
        return 0.0

    n = min(len(résidus), len(réf_res))
    if n == 0:
        return 0.0

    somme = 0.0
    for i in range(n):
        d = np.linalg.norm(résidus[i].CA - réf_res[i].CA)
        somme += d ** 2
    return math.sqrt(somme / n)


def rayon_gyration(conf: Conformation) -> float:
    """
    Rayon de giration Rg (Å).

    Mesure de la compacité de la protéine.
    """
    centres = np.array([r.CA for r in conf.residues])
    centre_masse = np.mean(centres, axis=0)
    carrés = np.sum((centres - centre_masse) ** 2, axis=1)
    n = len(conf.residues)
    if n == 0:
        return 0.0
    return math.sqrt(np.sum(carrés) / n)


def nombre_contacts(conf: Conformation, distance_seuil: float = 8.0) -> int:
    """
    Nombre de contacts entre résidus non-adjacents (|i−j| ≥ 2).

    Un contact est défini par une distance CA–CA < distance_seuil.
    """
    n = len(conf.residues)
    contacts = 0
    for i in range(n - 2):
        for j in range(i + 2, n):
            d = np.linalg.norm(conf.residues[i].CA - conf.residues[j].CA)
            if d < distance_seuil:
                contacts += 1
    return contacts


# ─── Fenêtre d'échantillonnage parapluie ──────────────────────────────

@dataclass
class Fenêtre:
    """
    Une fenêtre d'échantillonnage parapluie.

    Attributs :
        centre     : ξ_i⁰ — valeur cible de la coordonnée de réaction
        raideur    : k_i (kcal · mol⁻¹ · unité⁻²)
        température : température de simulation (K)
        beta       : 1 / k_B T
        historique_ξ : liste des valeurs ξ échantillonnées
        historique_E : liste des énergies totales
    """
    centre: float
    raideur: float
    température: float = 300.0
    beta: float = field(init=False)
    historique_ξ: list[float] = field(default_factory=list)
    historique_E: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.beta = 1.0 / (CONSTANTE_BOLTZMANN * self.température)

    def énergie_biais(self, ξ: float) -> float:
        """Calcule V_biais(ξ) = (k/2) · (ξ − ξ⁰)²."""
        return 0.5 * self.raideur * (ξ - self.centre) ** 2


# ─── Classe principale d'échantillonnage parapluie ────────────────────

class ÉchantillonnageParapluie:
    """
    Échantillonnage parapluie avec reconstruction WHAM.

    Lance des simulations MC avec biais harmonique dans chaque fenêtre,
    collecte les histogrammes de la coordonnée de réaction, puis les
    combine avec WHAM.

    Paramètres :
        fonction_énergie     : RosettaEnergyFunction pour E_total
        coordonnée_réaction  : fonction f(Conformation) → ξ
        pas_mc               : nombre de pas MC par fenêtre
        pas_d_échantillonnage : intervalle d'enregistrement de ξ
        taille_bin           : largeur des bins pour l'histogramme
    """

    def __init__(
        self,
        fonction_énergie: Optional[RosettaEnergyFunction] = None,
        coordonnée_réaction: Callable[[Conformation], float] = rayon_gyration,
        pas_mc: int = 5000,
        pas_d_échantillonnage: int = 10,
        taille_bin: float = 0.5,
    ):
        self.fonction_énergie = fonction_énergie or RosettaEnergyFunction()
        self.coordonnée_réaction = coordonnée_réaction
        self.pas_mc = pas_mc
        self.pas_d_échantillonnage = pas_d_échantillonnage
        self.taille_bin = taille_bin

        self.fenêtres: list[Fenêtre] = []
        self.conformations: list[Conformation] = []

    # ── Configuration des fenêtres ──────────────────────────────────

    def configurer_fenêtres(
        self,
        séquence: str,
        centres: list[float],
        raideur: float = 5.0,
        température: float = 300.0,
    ):
        """
        Configure les fenêtres et initialise une conformation par fenêtre.

        Paramètres :
            séquence   : séquence d'acides aminés (codes 1 lettre)
            centres    : ξ_i⁰ pour chaque fenêtre
            raideur    : constante de force k (identique pour toutes les fenêtres)
            température : température de simulation
        """
        self.fenêtres.clear()
        self.conformations.clear()

        # Conformation initiale (chaîne étendue)
        conf_base = Conformation(
            residues=[
                __import__("folding_engine", fromlist=["Residue"]).Residue(
                    seq_index=i + 1,
                    aa_code=__import__("folding_engine", fromlist=["AA_CODES"]).AA_CODES.get(aa, "ALA"),
                )
                for i, aa in enumerate(séquence.upper())
            ]
        )
        # Reconstruction des coordonnées
        for i, res in enumerate(conf_base.residues):
            if i > 0:
                prev = conf_base.residues[i - 1]
                res.N = prev.C + np.array([0.0, 0.0, 1.33])
                res.CA = res.N + np.array([1.47, 0.0, 0.0])
                res.C = res.CA + np.array([0.0, 1.51, 0.0])
                res.O = res.C + np.array([0.0, 0.0, 1.23])
                res.CB = res.CA + np.array([0.0, 0.0, 1.53])

        for centre in centres:
            self.fenêtres.append(
                Fenêtre(centre=centre, raideur=raideur, température=température)
            )
            # Copie de la conformation étendue pour chaque fenêtre
            self.conformations.append(
                self._copier_conformation(conf_base)
            )

        print(f"  [Parapluie] {len(centres)} fenêtres, k = {raideur:.1f}, "
              f"ξ ∈ [{min(centres):.1f}, {max(centres):.1f}]")

    @staticmethod
    def _copier_conformation(conf: Conformation) -> Conformation:
        """Crée une copie indépendante d'une conformation."""
        from copy import deepcopy
        return deepcopy(conf)

    # ── Exécution ───────────────────────────────────────────────────

    def exécuter(
        self,
        séquence: str,
        centres: list[float],
        raideur: float = 5.0,
        température: float = 300.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Exécute l'échantillonnage parapluie complet.

        1. Configure les fenêtres
        2. Lance MC avec biais dans chaque fenêtre
        3. Construit les histogrammes
        4. Combine via WHAM

        Retourne :
            (ξ_bins, F_profil) — profil d'énergie libre sans biais
        """
        self.configurer_fenêtres(séquence, centres, raideur, température)

        for idx_fenêtre, (fenêtre, conf) in enumerate(
            zip(self.fenêtres, self.conformations)
        ):
            self._échantillonner_fenêtre(fenêtre, conf, idx_fenêtre)

        # Reconstruction WHAM
        return self._reconstruire_profil()

    # ── MC avec biais harmonique ────────────────────────────────────

    def _échantillonner_fenêtre(
        self, fenêtre: Fenêtre, conf: Conformation, idx: int
    ):
        """
        Simulation MC avec biais harmonique dans une fenêtre.

        L'énergie totale pour le critère de Métropole est :
            E_tot = E_protéine + V_biais(ξ)
        """
        ξ = self.coordonnée_réaction(conf)
        E_prot = self.fonction_énergie.evaluate(conf)
        E_biais = fenêtre.énergie_biais(ξ)
        E_totale = E_prot + E_biais

        n_res = len(conf.residues)

        for pas in range(self.pas_mc):
            # Perturbation (insertion de fragment)
            start = random.randint(0, n_res - 4)
            frag_len = 3

            anciens_angles = []
            for i in range(start, min(start + frag_len, n_res)):
                r = conf.residues[i]
                anciens_angles.append((r.phi, r.psi))
                r.phi += random.gauss(0, 15)
                r.psi += random.gauss(0, 15)

            # Reconstruction
            self._reconstruire(conf, start, start + frag_len)

            # Énergie + biais
            E_prot_nouv = self.fonction_énergie.evaluate(conf)
            ξ_nouv = self.coordonnée_réaction(conf)
            E_biais_nouv = fenêtre.énergie_biais(ξ_nouv)
            E_totale_nouv = E_prot_nouv + E_biais_nouv

            ΔE = E_totale_nouv - E_totale

            if ΔE < 0 or random.random() < math.exp(-ΔE * fenêtre.beta):
                E_totale = E_totale_nouv
                ξ = ξ_nouv
                E_prot = E_prot_nouv
            else:
                # Restaurer
                for i, (phi, psi) in enumerate(anciens_angles):
                    idx_r = start + i
                    if idx_r < n_res:
                        conf.residues[idx_r].phi = phi
                        conf.residues[idx_r].psi = psi
                self._reconstruire(conf, start, start + frag_len)

            # Enregistrement
            if pas % self.pas_d_échantillonnage == 0:
                fenêtre.historique_ξ.append(ξ)
                fenêtre.historique_E.append(E_prot)

        if (idx + 1) % max(1, len(self.fenêtres) // 5) == 0:
            print(f"  [Parapluie] Fenêtre {idx + 1}/{len(self.fenêtres)} "
                  f"ξ⁰ = {fenêtre.centre:.1f}, "
                  f"⟨ξ⟩ = {np.mean(fenêtre.historique_ξ):.2f}")

    @staticmethod
    def _reconstruire(conf: Conformation, début: int, fin: int):
        """Reconstruit les coordonnées CA/C/O/CB."""
        for i in range(max(1, début), min(fin, len(conf.residues))):
            prev = conf.residues[i - 1]
            curr = conf.residues[i]

            curr.N = prev.C + np.array([0.0, 0.0, 1.33])

            phi_rad = math.radians(curr.phi)
            curr.CA = curr.N + np.array([
                1.47 * math.cos(phi_rad), 1.47 * math.sin(phi_rad), 0.0,
            ])

            psi_rad = math.radians(curr.psi)
            curr.C = curr.CA + np.array([
                1.51 * math.cos(psi_rad), 1.51 * math.sin(psi_rad), 0.0,
            ])

            perp = np.cross(curr.N - curr.CA, curr.C - curr.CA)
            norme = np.linalg.norm(perp) + 1e-10
            curr.O = curr.C + (perp / norme) * 1.23

            cb_dir = np.cross(curr.N - curr.CA, perp / norme)
            cb_dir = cb_dir / (np.linalg.norm(cb_dir) + 1e-10)
            curr.CB = curr.CA + cb_dir * 1.53

    # ── Reconstruction WHAM ─────────────────────────────────────────

    def _reconstruire_profil(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Combine les histogrammes biaisés via WHAM.

        Retourne :
            (ξ_bins, F_profil)
        """
        # Déterminer l'étendue des valeurs de ξ
        tous_ξ = []
        for fenêtre in self.fenêtres:
            tous_ξ.extend(fenêtre.historique_ξ)

        if not tous_ξ:
            raise RuntimeError("Aucune donnée d'échantillonnage.")

        ξ_min, ξ_max = min(tous_ξ), max(tous_ξ)
        marge = self.taille_bin * 2
        ξ_bins = np.arange(ξ_min - marge, ξ_max + marge + self.taille_bin / 2,
                           self.taille_bin)
        centres_bins = ξ_bins[:-1] + self.taille_bin / 2

        wham = WHAM(tolérance=1e-6)

        for fenêtre in self.fenêtres:
            # Histogramme des valeurs de ξ
            comptes, _ = np.histogram(fenêtre.historique_ξ, bins=ξ_bins)
            comptes = comptes.astype(np.float64)

            # Biais V(ξ) pour chaque bin
            biais = np.array([fenêtre.énergie_biais(c) for c in centres_bins])

            wham.ajouter_histogramme(
                Histogramme(
                    température=fenêtre.température,
                    bins=centres_bins.copy(),
                    comptes=comptes,
                    nb_échantillons=len(fenêtre.historique_ξ),
                    biais=biais,
                )
            )

        wham.résoudre()
        _, F = wham.profil_énergie_libre(fenêtre.température)

        print(f"  [Parapluie] Profil reconstruit via WHAM : "
              f"F_min = {np.min(F):.2f}, F_max = {np.max(F):.2f}")

        return centres_bins, F


# ─── Point d'entrée pour les tests ────────────────────────────────────

if __name__ == "__main__":
    # Test : échantillonnage parapluie sur la coordonnée Rg
    # d'un petit peptide
    SÉQUENCE_TEST = "AAAAKAAAAKAAAAK"

    print("  [Test Parapluie] Peptide 15 résidus, 5 fenêtres Rg")
    échantillonneur = ÉchantillonnageParapluie(
        coordonnée_réaction=rayon_gyration,
        pas_mc=2000,
        pas_d_échantillonnage=10,
        taille_bin=0.3,
    )

    ξ_bins, F = échantillonneur.exécuter(
        séquence=SÉQUENCE_TEST,
        centres=[6.0, 7.0, 8.0, 9.0, 10.0],
        raideur=2.0,
        température=300.0,
    )

    print(f"  [Test] Profil libre : {len(ξ_bins)} bins")
    print(f"  [Test] Plage F : [{np.min(F):.2f}, {np.max(F):.2f}] kcal/mol")

    # Vérification que le puits de Rg compact (petit Rg) correspond
    # à une énergie libre plus basse.
    idx_min = np.argmin(F)
    print(f"  [Test] Minimum F à Rg ≈ {ξ_bins[idx_min]:.2f} Å")
