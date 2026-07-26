#!/usr/bin/env python3
"""
sampling/replica_exchange.py
=============================

Échange de Répliques (Replica Exchange Molecular Dynamics — REMD).

Algorithme de Monte Carlo parallèle en température. On simule N répliques
du système à des températures T₁ < T₂ < ... < T_N. Périodiquement, on
tente d'échanger les configurations entre répliques adjacentes, ce qui
permet au système de franchir les barrières d'énergie.

Critère d'échange (Metropolis) :

    P_acc = min(1, exp[(βᵢ − βⱼ)(Eᵢ − Eⱼ)])

où β = 1/(k_B·T).

Références :
    - Swendsen & Wang (1986) Phys Rev Lett 57:2607
    - Sugita & Okamoto (1999) Chem Phys Lett 314:141
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Réplique:
    """Une réplique du système à une température donnée."""
    température: float              # Température (K)
    énergie: float = 0.0            # Énergie potentielle courante (kcal/mol)
    accepté: int = 0                # Nombre d'échanges acceptés
    proposé: int = 0                # Nombre d'échanges proposés


class ÉchangeDeRépliques:
    """
    Algorithme d'échange de répliques (REMD).

    Exemple:
        >>> remd = ÉchangeDeRépliques(
        ...     n_répliques=3,
        ...     températures=[300.0, 400.0, 500.0]
        ... )
        >>> remd.étape_locale(énergies=[-14.5, -10.2, -8.1])
        >>> remd.échange()
    """

    def __init__(self, n_répliques: int = 3,
                 températures: list[float] = None):
        if températures is None:
            températures = [300.0, 350.0, 400.0]

        self.n_répliques = n_répliques
        self.bêtas = [1.0 / (0.001987 * T) for T in températures]
        self.répliques = [
            Réplique(température=T)
            for T in températures
        ]
        self.historique_énergies: list[list[float]] = []
        self.matrices_échange: list[float] = []

    def étape_locale(self, énergies: list[float]):
        """
        Met à jour l'énergie de chaque réplique (simule un pas MC local).

        Paramètres:
            énergies: Liste des énergies pour chaque réplique.
        """
        for réplique, énergie in zip(self.répliques, énergies):
            réplique.énergie = énergie

        self.historique_énergies.append(list(énergies))

    def échange(self) -> dict:
        """
        Tente d'échanger les configurations entre répliques adjacentes.

        L'échange suit un schéma pair → impair, puis impair → pair,
        pour garantir la réversibilité détaillée.

        Retourne:
            Dictionnaire avec les taux d'acceptation.
        """
        n = self.n_répliques
        stats = {'échanges': 0, 'total': 0}

        for offset in [0, 1]:  # Pair puis impair
            for i in range(offset, n - 1, 2):
                β_i = self.bêtas[i]
                β_j = self.bêtas[i + 1]
                E_i = self.répliques[i].énergie
                E_j = self.répliques[i + 1].énergie

                # Calcul du log du rapport de Metropolis
                Δ = (β_i - β_j) * (E_i - E_j)

                self.répliques[i].proposé += 1
                self.répliques[i + 1].proposé += 1
                stats['total'] += 1

                if Δ >= 0 or random.random() < math.exp(Δ):
                    # Échange accepté
                    self.répliques[i].énergie, self.répliques[i + 1].énergie = \
                        E_j, E_i
                    self.répliques[i].accepté += 1
                    self.répliques[i + 1].accepté += 1
                    stats['échanges'] += 1

        taux = stats['échanges'] / max(stats['total'], 1)
        self.matrices_échange.append(taux)
        return stats

    def diagnostique(self) -> dict:
        """
        Diagnostique de convergence du REMD.

        Retourne:
            Dictionnaire avec les taux d'acceptation par paire et
            le nombre total de cycles.
        """
        diag = {
            'cycles': len(self.historique_énergies),
            'taux_acceptation': [],
        }
        for i in range(self.n_répliques - 1):
            r = self.répliques[i]
            taux = r.accepté / max(r.proposé, 1)
            diag['taux_acceptation'].append(
                f"T{r.température:.0f}K↔T{self.répliques[i+1].température:.0f}K: "
                f"{taux*100:.1f}%"
            )
        return diag

    def énergie_minimale(self) -> tuple[float, float]:
        """Énergie la plus basse et température correspondante."""
        idx = min(range(self.n_répliques),
                  key=lambda i: self.répliques[i].énergie)
        return self.répliques[idx].énergie, self.répliques[idx].température


if __name__ == "__main__":
    # Test : 3 répliques, 300/400/500 K, 200 cycles d'échange
    remd = ÉchangeDeRépliques(n_répliques=3,
                              températures=[300.0, 400.0, 500.0])

    for _ in range(200):
        # Simulation d'énergies issues d'un repliement
        E = [300 * (1 - math.exp(-i / 100)) + random.gauss(0, 2)
             for i in range(3)]
        remd.étape_locale([E[0], E[1], E[2]])
        remd.échange()

    diag = remd.diagnostique()
    print(f"Diagnostique REMD ({diag['cycles']} cycles) :")
    for t in diag['taux_acceptation']:
        print(f"  {t}")

    E_min, T_min = remd.énergie_minimale()
    print(f"Énergie minimale : {E_min:.1f} kcal/mol à {T_min:.0f} K")
