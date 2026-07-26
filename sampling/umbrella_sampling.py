#!/usr/bin/env python3
"""
sampling/umbrella_sampling.py
==============================

Échantillonnage Parapluie (Umbrella Sampling) avec WHAM.

On applique un potentiel harmonique de biais le long d'une coordonnée
de réaction ξ pour explorer des régions de haute énergie :

    V_bias(ξ) = ½ · k · (ξ − ξ⁰)²

Le profil d'énergie libre sans biais est reconstruit via WHAM.

Références :
    - Torrie & Valleau (1977) J Comput Phys 23:187
    - Kästner (2011) WIREs Comput Mol Sci 1:932
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .wham import WHAM


@dataclass
class Fenêtre:
    """Une fenêtre d'échantillonnage parapluie."""
    centre: float               # ξ⁰ — centre de la fenêtre
    raideur: float              # k — constante de raideur (kcal/mol/Å²)
    échantillons: list[float] = field(default_factory=list)

    def ajouter(self, ξ: float):
        self.échantillons.append(ξ)


class ÉchantillonnageParapluie:
    """
    Échantillonnage parapluie avec reconstruction WHAM.

    Exemple:
        >>> us = ÉchantillonnageParapluie(
        ...     fenêtres=[(6.0, 2.0), (8.0, 2.0)],
        ...     n_bins=50
        ... )
        >>> us.échantillonner(ξ=7.0, fenêtre=0)
        >>> profil = us.reconstruire()
    """

    def __init__(self, fenêtres: list[tuple[float, float]],
                 n_bins: int = 50, température: float = 300.0):
        """
        Paramètres:
            fenêtres: Liste de (centre_ξ, raideur_k)
            n_bins: Résolution de l'histogramme WHAM
            température: Température de simulation (K)
        """
        self.fenêtres = [Fenêtre(centre=c, raideur=k)
                         for c, k in fenêtres]
        self.n_bins = n_bins
        self.température = température

        # Coordonnée de réaction
        self.ξ_min = min(c for c, _ in fenêtres) - 2.0
        self.ξ_max = max(c for c, _ in fenêtres) + 2.0
        self.grille = np.linspace(self.ξ_min, self.ξ_max, n_bins)

    def échantillonner(self, ξ: float, fenêtre: int):
        """
        Ajoute un échantillon à la fenêtre spécifiée.
        En conditions réelles, ceci vient d'une simulation MC/MD.
        """
        if 0 <= fenêtre < len(self.fenêtres):
            self.fenêtres[fenêtre].ajouter(ξ)

    def reconstruire(self) -> dict:
        """
        Reconstruit le profil d'énergie libre sans biais par WHAM.

        Retourne:
            Dictionnaire avec le profil PMF (Potential of Mean Force)
            et la coordonnée de réaction correspondante.
        """
        n_fenêtres = len(self.fenêtres)
        n_bins = self.n_bins
        dx = (self.ξ_max - self.ξ_min) / n_bins

        # Construction des histogrammes biaisés
        histogrammes = []
        for fenêtre in self.fenêtres:
            hist, _ = np.histogram(
                fenêtre.échantillons,
                bins=n_bins,
                range=(self.ξ_min, self.ξ_max)
            )
            histogrammes.append(hist.astype(float))

        # Application du débiais (retrait du potentiel harmonique)
        énergies_biais = []
        for fenêtre in self.fenêtres:
            décalage = self.grille - fenêtre.centre
            énergie_biais = 0.5 * fenêtre.raideur * décalage ** 2
            énergies_biais.append(énergie_biais)

        β = 1.0 / (0.001987 * self.température)

        for j, énergie in enumerate(énergies_biais):
            histogrammes[j] *= np.exp(β * énergie)

        # WHAM
        wham = WHAM(n_bins=n_bins, température=self.température)
        résultat = wham.résoudre(histogrammes=histogrammes)

        return {
            'ξ': self.grille,
            'PMF': résultat['énergie_libre'],
            'itérations_wham': résultat['itérations'],
            'Ω': résultat['Ω'],
        }


# ─── Coordonnées de réaction ────────────────────────────────────────

def rayon_de_gyration(coordonnées: np.ndarray) -> float:
    """
    Calcule le rayon de giration Rg d'un ensemble d'atomes.

    Rg² = (1/N) · Σᵢ |rᵢ − r_cm|²

    Paramètres:
        coordonnées: Tableau [N, 3] des positions atomiques en Å

    Retourne:
        Rayon de giration en Å
    """
    centre_de_masse = np.mean(coordonnées, axis=0)
    déviation = coordonnées - centre_de_masse
    rg_sq = np.mean(np.sum(déviation ** 2, axis=1))
    return math.sqrt(rg_sq)


def nombre_de_contacts(coordonnées: np.ndarray,
                       distance_seuil: float = 6.0) -> int:
    """
    Nombre de paires d'atomes dans un rayon de coupure.

    Paramètres:
        coordonnées: Tableau [N, 3] — positions atomiques en Å
        distance_seuil: Distance de contact en Å

    Retourne:
        Nombre de contacts intra-moléculaires.
    """
    n = len(coordonnées)
    contacts = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = np.linalg.norm(coordonnées[i] - coordonnées[j])
            if d < distance_seuil:
                contacts += 1
    return contacts


if __name__ == "__main__":
    # Test : 5 fenêtres sur Rg (6–10 Å), reconstruction WHAM
    centres = [6.0, 7.0, 8.0, 9.0, 10.0]
    raideurs = [2.0] * 5

    us = ÉchantillonnageParapluie(
        fenêtres=list(zip(centres, raideurs)),
        n_bins=50
    )

    # Échantillons simulés : gaussienne autour de chaque centre
    for i, centre in enumerate(centres):
        for _ in range(200):
            ξ = random.gauss(centre, 0.8)
            us.échantillonner(ξ, fenêtre=i)

    import random  # noqa
    résultat = us.reconstruire()

    print(f"Échantillonnage Parapluie — {len(centres)} fenêtres")
    print(f"WHAM : {résultat['itérations_wham']} itérations")
    print(f"PMF min : {np.min(résultat['PMF']):.2f} kcal/mol")
    print(f"PMF max : {np.max(résultat['PMF']):.2f} kcal/mol")
