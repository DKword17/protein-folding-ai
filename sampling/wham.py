#!/usr/bin/env python3
"""
sampling/wham.py
================

WHAM — Weighted Histogram Analysis Method (Méthode d'analyse par
histogrammes pondérés).

Algorithme de Kumar et al. (1992) pour combiner plusieurs simulations
(à différentes températures ou sous différents potentiels de biais)
en une estimation unique de l'énergie libre.

Équations d'auto-cohérence WHAM :

    P(ξ) = Σᵢ nᵢ(ξ) / Σⱼ Nⱼ · exp(−βⱼ[Vⱼ(ξ) − fⱼ])

    exp(−β_k f_k) = Σ_ξ P(ξ) · exp(−β_k V_k(ξ))

où :
    nᵢ(ξ)  → histogramme de la simulation i dans le bin ξ
    Nⱼ     → nombre total d'échantillons de la simulation j
    βⱼ     → 1/k_B Tⱼ
    Vⱼ(ξ)  → potentiel de biais de la simulation j à la coordonnée ξ
    fⱼ     → énergie libre (constante de normalisation) de la simulation j

Ces équations sont résolues par itération jusqu'à convergence.

Variables :
    Histogramme     → nᵢ(ξ) — nombre d'observations par bin
    Énergie_libre   → fⱼ — constante de normalisation
    Poids           → poids statistique de chaque observation
    Convergence     → critère d'arrêt (Δf < tolérance)

Références :
    - Kumar et al. (1992) J. Comput. Chem. 13:1011
    - Frenkel & Smit, Understanding Molecular Simulation, §7.3
    - Souaille & Roux (2001) Comput. Phys. Commun. 135:40
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

sys.path.insert(0, "..")

# ─── Constante physique ───────────────────────────────────────────────

CONSTANTE_BOLTZMANN: float = 0.001987  # kcal · mol⁻¹ · K⁻¹


# ─── Structure de données pour un histogramme ─────────────────────────

@dataclass
class Histogramme:
    """
    Histogramme d'une simulation (température ou fenêtre de biais).

    Attributs :
        température   : température de la simulation (K)
        beta          : 1 / k_B T
        biais         : fonction de biais V(ξ) — None si sans biais
        nb_échantillons : nombre total d'échantillons N
        bins          : tableau des centres de bins de la coordonnée ξ
        comptes       : tableau n(ξ) — nombre d'observations par bin
    """
    température: float
    bins: np.ndarray                    # centres de bins
    comptes: np.ndarray                 # n(ξ) par bin
    nb_échantillons: int = 0
    beta: float = field(init=False)
    biais: Optional[np.ndarray] = None  # V(ξ) si umbrella, None si non biaisé

    def __post_init__(self):
        self.beta = 1.0 / (CONSTANTE_BOLTZMANN * self.température)
        if self.nb_échantillons == 0:
            self.nb_échantillons = int(np.sum(self.comptes))


# ─── Classe principale WHAM ───────────────────────────────────────────

class WHAM:
    """
    Résolution auto-cohérente des équations WHAM.

    Combine des histogrammes provenant de simulations à différentes
    températures (REMD) ou de fenêtres d'échantillonnage parapluie
    pour produire une estimation optimale de la densité d'états et
    de l'énergie libre.

    Paramètres :
        tolérance    : critère de convergence pour Δf
        itérations_max : nombre maximum d'itérations auto-cohérentes
    """

    def __init__(self, tolérance: float = 1e-6, itérations_max: int = 10000):
        self.tolérance = tolérance
        self.itérations_max = itérations_max
        self.histogrammes: list[Histogramme] = []
        self.énergies_libres: np.ndarray = field(init=False, default=None)  # f_k
        self.densité_états: np.ndarray = field(init=False, default=None)    # Ω(ξ)
        self.poids: np.ndarray = field(init=False, default=None)            # poids totaux
        self.convergé: bool = False
        self.itérations_effectuées: int = 0

    # ── Ajout d'histogrammes ────────────────────────────────────────

    def ajouter_histogramme(self, histogramme: Histogramme):
        """Ajoute un histogramme à l'analyse WHAM."""
        self.histogrammes.append(histogramme)

    def ajouter_depuis_trajectoire(
        self,
        températures: list[float],
        histogrammes: list[np.ndarray],
        bins: np.ndarray,
    ):
        """
        Ajoute plusieurs histogrammes à partir de données brutes.

        Paramètres :
            températures : liste des températures pour chaque simulation
            histogrammes : liste de tableaux [n_échantillons] par simulation
            bins         : centres de bins communs à tous les histogrammes
        """
        for T, histo in zip(températures, histogrammes):
            if len(histo) != len(bins):
                raise ValueError("Taille de l'histogramme incompatible avec bins.")
            self.ajouter_histogramme(
                Histogramme(température=T, bins=bins, comptes=histo)
            )

    # ── Résolution auto-cohérente ───────────────────────────────────

    def résoudre(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Résout itérativement les équations WHAM.

        Retourne :
            (densité_états, énergies_libres)
                densité_états   : Ω(ξ) — densité d'états sans biais
                énergies_libres : f_k — constantes de normalisation
        """
        if not self.histogrammes:
            raise RuntimeError("Aucun histogramme ajouté à WHAM.")

        nb_bins = len(self.histogrammes[0].bins)
        nb_sim = len(self.histogrammes)

        # Vérification que tous les bins sont identiques
        for h in self.histogrammes:
            if len(h.bins) != nb_bins:
                raise ValueError("Tous les histogrammes doivent avoir le même nombre de bins.")

        # Initialisation : énergies libres à zéro
        f = np.zeros(nb_sim, dtype=np.float64)
        # Fixer f[0] = 0 comme référence (invariance par translation globale)
        f[0] = 0.0
        Ω = np.zeros(nb_bins, dtype=np.float64)

        # Matrice des poids dénominateur : W[k, ξ] = N_k · exp(−β_k[V_k(ξ) − f_k])
        préfacteurs = np.zeros((nb_sim, nb_bins), dtype=np.float64)

        print(f"  [WHAM] {nb_sim} simulations, {nb_bins} bins, "
              f"tolérance = {self.tolérance:.0e}")

        for it in range(self.itérations_max):
            f_précédent = f.copy()

            # Mise à jour du dénominateur
            for k in range(nb_sim):
                h = self.histogrammes[k]
                biais = h.biais if h.biais is not None else np.zeros(nb_bins)
                préfacteurs[k, :] = h.nb_échantillons * np.exp(
                    -h.beta * (biais - f[k])
                )

            # Normalisation : dénominateur total
            dénominateur = np.sum(préfacteurs, axis=0)  # [nb_bins]

            # Nouvelle estimation de Ω(ξ)
            Ω = np.zeros(nb_bins)
            for k in range(nb_sim):
                Ω += self.histogrammes[k].comptes.astype(np.float64)
            Ω /= (dénominateur + 1e-30)

            # Nouvelle estimation de f_k
            for k in range(1, nb_sim):  # k=0 est la référence (f=0)
                h = self.histogrammes[k]
                biais = h.biais if h.biais is not None else np.zeros(nb_bins)
                numerateur = np.sum(Ω * np.exp(-h.beta * biais))
                f_k_nouveau = -(1.0 / h.beta) * math.log(max(numerateur, 1e-30))

                # Sous-relaxation pour éviter les oscillations
                f[k] = 0.5 * f_k_nouveau + 0.5 * f[k]

            # Convergence
            Δf = np.max(np.abs(f - f_précédent))
            if Δf < self.tolérance:
                self.convergé = True
                self.itérations_effectuées = it + 1
                print(f"  [WHAM] Convergence atteinte en {it + 1} itérations. "
                      f"Δf_max = {Δf:.2e}")
                break
        else:
            print(f"  [WHAM] ATTENTION : pas de convergence après "
                  f"{self.itérations_max} itérations. Δf_max = {Δf:.2e}")

        self.énergies_libres = f
        self.densité_états = Ω
        self.poids = préfacteurs

        # Profil d'énergie libre : F(ξ) = −k_B T · ln Ω(ξ)
        return Ω, f

    # ── Profil d'énergie libre ──────────────────────────────────────

    def profil_énergie_libre(
        self, température_référence: Optional[float] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calcule le profil d'énergie libre F(ξ) = −k_B T · ln Ω(ξ).

        Paramètres :
            température_référence : température pour k_B T (défaut = 300 K)

        Retourne :
            (bins, F) — énergie libre par bin en kcal/mol
        """
        if self.densité_états is None:
            self.résoudre()

        T = température_référence or 300.0
        beta = 1.0 / (CONSTANTE_BOLTZMANN * T)

        bins = self.histogrammes[0].bins
        F = -CONSTANTE_BOLTZMANN * T * np.log(self.densité_états + 1e-30)
        F -= np.min(F)  # zéro au minimum

        return bins, F

    # ── Barre d'erreur bootstrap ────────────────────────────────────

    def bootstrap(
        self, n_réplicats: int = 100, température_référence: Optional[float] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimation des barres d'erreur par bootstrap.

        Retourne :
            (bins, F_moy, F_écart)
        """
        if self.densité_états is None:
            self.résoudre()

        T = température_référence or 300.0
        nb_sim = len(self.histogrammes)
        nb_bins = len(self.histogrammes[0].bins)
        profils = np.zeros((n_réplicats, nb_bins))

        for r in range(n_réplicats):
            wham_boot = WHAM(tolérance=self.tolérance, itérations_max=500)

            for k in range(nb_sim):
                h = self.histogrammes[k]
                # Ré-échantillonnage multinomial des comptes
                total = h.nb_échantillons
                p = h.comptes.astype(np.float64) / max(total, 1)
                p = p / np.sum(p)  # normalisation explicite contre les erreurs FP
                nouveaux_comptes = np.random.multinomial(total, p)
                wham_boot.ajouter_histogramme(
                    Histogramme(
                        température=h.température,
                        bins=h.bins.copy(),
                        comptes=nouveaux_comptes,
                        nb_échantillons=total,
                        biais=h.biais.copy() if h.biais is not None else None,
                    )
                )

            wham_boot.résoudre()
            _, F_b = wham_boot.profil_énergie_libre(T)
            profils[r, :] = F_b

        bins = self.histogrammes[0].bins
        F_moy = np.mean(profils, axis=0)
        F_écart = np.std(profils, axis=0)

        return bins, F_moy, F_écart


# ─── Point d'entrée pour les tests ────────────────────────────────────

if __name__ == "__main__":
    # Test : deux histogrammes gaussiens à températures différentes
    print("  [WHAM Test] Deux distributions gaussiennes, T = 300 K et 500 K")

    bins = np.linspace(-4, 4, 41)
    centre = bins[:-1] + (bins[1] - bins[0]) / 2  # centres

    # Simulation 1 (T=300 K) : gaussienne centrée en 0
    σ1 = 0.8
    histo1 = np.exp(-(centre ** 2) / (2 * σ1 ** 2))
    histo1 += np.random.default_rng(42).normal(0, 0.01, len(centre))
    histo1 = np.maximum(histo1, 0.0) * 200

    # Simulation 2 (T=500 K) : gaussienne plus large, même centre
    σ2 = 1.5
    histo2 = np.exp(-(centre ** 2) / (2 * σ2 ** 2))
    histo2 += np.random.default_rng(123).normal(0, 0.01, len(centre))
    histo2 = np.maximum(histo2, 0.0) * 200

    wham = WHAM(tolérance=1e-6)
    wham.ajouter_depuis_trajectoire(
        températures=[300.0, 500.0],
        histogrammes=[histo1, histo2],
        bins=centre,
    )

    Ω, f = wham.résoudre()
    bins_out, F = wham.profil_énergie_libre()

    print(f"  [Test] Énergies libres de normalisation : {f}")
    print(f"  [Test] F_min = {np.min(F):.3f}, F_max = {np.max(F):.3f} kcal/mol")

    # Bootstrap
    _, F_moy, F_std = wham.bootstrap(n_réplicats=50)
    print(f"  [Test] Bootstrap : erreur moyenne = {np.mean(F_std):.4f} kcal/mol")
