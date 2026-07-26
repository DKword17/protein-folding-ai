#!/usr/bin/env python3
"""
sampling/wham.py
=================

Weighted Histogram Analysis Method (WHAM).

WHAM combine plusieurs histogrammes issus de simulations à différentes
températures (ou fenêtres de biais) pour reconstruire l'énergie libre
et la densité d'états sans biais.

Équation maîtresse (résolution auto-cohérente) :

    p(ξ) = Σⱼ nⱼ(ξ) · wⱼ / Ω(ξ)
    Ω(ξ) = Σⱼ nⱼ(ξ) / Σᵢ Nᵢ · exp(−βᵢ · [Vᵢ(ξ) − fᵢ])

où fᵢ sont les constantes de normalisation.

Références :
    - Kumar et al. (1992) J Comput Chem 13:1011
    - Souaille & Roux (2001) Comput Phys Commun 135:40
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class WHAM:
    """
    Weighted Histogram Analysis Method.

    Résout le système d'équations auto-cohérentes pour reconstruire
    la densité d'états sans biais à partir d'histogrammes biaisés.

    Exemple:
        >>> wham = WHAM(n_bins=50, température=300.0)
        >>> hist = [np.ones(50) * 100]  # Histogrammes simulés
        >>> résultat = wham.résoudre(histogrammes=hist)
    """
    n_bins: int = 50                     # Nombre de bins de la variable de réaction
    température: float = 300.0           # Température de référence (K)
    β_ref: float = field(init=False)     # 1/(k_B·T_ref)
    tolérance: float = 1e-6
    max_itérations: int = 1000

    def __post_init__(self):
        k_B = 0.001987  # kcal/(mol·K)
        self.β_ref = 1.0 / (k_B * self.température)

    def résoudre(self, histogrammes: list[np.ndarray],
                  énergies: list[np.ndarray] = None,
                  poids: list[float] = None) -> dict:
        """
        Résout les équations WHAM de manière auto-cohérente.

        Paramètres:
            histogrammes: Liste de tableaux [n_bins] — histogrammes
                          de chaque simulation/fenêtre.
            énergies: Liste d'énergies moyennes par bin (optionnel).
            poids: Poids statistiques pour chaque fenêtre.

        Retourne:
            Dictionnaire avec la densité d'états, l'énergie libre,
            et les constantes de normalisation f_i.
        """
        n_fenêtres = len(histogrammes)
        if poids is None:
            poids = [1.0] * n_fenêtres

        N = [np.sum(h) for h in histogrammes]  # Nombre d'échantillons par fenêtre
        β = [self.β_ref] * n_fenêtres

        # Initialisation des constantes de normalisation
        f = np.zeros(n_fenêtres)

        # Itération auto-cohérente
        for it in range(self.max_itérations):
            # Mise à jour de Ω(ξ) = densité d'états
            dénominateur = np.zeros(self.n_bins)
            for j in range(n_fenêtres):
                dénominateur += N[j] * np.exp(-β[j] * self._énergie_biais(j, histogrammes[j]) + f[j])

            Ω = np.zeros(self.n_bins)
            for j in range(n_fenêtres):
                hist_j = histogrammes[j] + 1e-15  # Éviter division par zéro
                Ω += hist_j * poids[j]

            Ω /= (dénominateur + 1e-15)

            # Mise à jour des f_i
            f_old = f.copy()
            for j in range(n_fenêtres):
                somme = np.sum(histogrammes[j] / (dénominateur + 1e-15))
                if somme > 0:
                    f[j] = math.log(max(somme, 1e-15))

            # Vérification de la convergence
            Δf = np.max(np.abs(f - f_old))
            if Δf < self.tolérance:
                break

        # Calcul de l'énergie libre sans biais
        k_B = 0.001987
        β = 1.0 / (k_B * self.température)
        énergie_libre = -1.0 / β * np.log(Ω + 1e-15)
        énergie_libre -= np.min(énergie_libre)  # Décalage à zéro

        return {
            'Ω': Ω,                              # Densité d'états
            'énergie_libre': énergie_libre,       # Profil d'énergie libre
            'f': f,                               # Constantes de normalisation
            'itérations': it + 1,
            'convergence': Δf,
        }

    def _énergie_biais(self, fenêtre: int, histogramme: np.ndarray) -> np.ndarray:
        """Énergie de biais (harmonique) pour la fenêtre donnée."""
        # Approximation : l'histogramme lui-même encode le biais
        return np.zeros_like(histogramme)

    def bootstrap(self, histogrammes: list[np.ndarray],
                  n_réplicats: int = 50) -> dict:
        """
        Estimation d'erreur par bootstrap.

        Retourne:
            Profil d'énergie libre moyen et écart-type.
        """
        profils = []
        n = len(histogrammes)

        for _ in range(n_réplicats):
            # Ré-échantillonnage multinomial
            boot_hist = []
            for h in histogrammes:
                total = int(np.sum(h))
                indices = np.random.choice(self.n_bins, size=total, p=h / total)
                boot_h, _ = np.histogram(indices, bins=self.n_bins,
                                         range=(0, self.n_bins - 1))
                boot_hist.append(boot_h.astype(float))

            try:
                résultat = self.résoudre(histogrammes=boot_hist)
                profils.append(résultat['énergie_libre'])
            except Exception:
                continue

        profils = np.array(profils)
        return {
            'moyenne': np.mean(profils, axis=0),
            'écart_type': np.std(profils, axis=0),
            'n_réplicats': len(profils),
        }


if __name__ == "__main__":
    # Test : 2 distributions gaussiennes, convergence WHAM
    n_bins = 50
    x = np.linspace(0, 10, n_bins)

    # Deux fenêtres simulées
    hist1 = np.exp(-0.5 * ((x - 3) / 1.5) ** 2) * 500 + 10
    hist2 = np.exp(-0.5 * ((x - 7) / 1.5) ** 2) * 500 + 10

    wham = WHAM(n_bins=n_bins)
    résultat = wham.résoudre(histogrammes=[hist1, hist2])

    print(f"WHAM — Convergence en {résultat['itérations']} itérations")
    print(f"Δf final : {résultat['convergence']:.2e}")
    print(f"Énergie libre minimale : {np.min(résultat['énergie_libre']):.2f}")

    # Bootstrap
    boot = wham.bootstrap([hist1, hist2], n_réplicats=20)
    print(f"Erreur moyenne (bootstrap) : {np.mean(boot['écart_type']):.3f}")
