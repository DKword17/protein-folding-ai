"""
sampling/__init__.py
====================

Module d'échantillonnage du paysage énergétique.

Ce package implémente trois méthodes avancées d'échantillonnage
pour le repliement de protéines :

    1. Échange de répliques (REMD) — échantillonnage multi-température
       avec échange périodique de conformations.
    2. WHAM (Weighted Histogram Analysis Method) — extraction de
       l'énergie libre par combinaison optimale d'histogrammes.
    3. Échantillonnage parapluie — biais harmonique le long d'une
       coordonnée de réaction.

Toutes les méthodes reposent sur RosettaEnergyFunction et Conformation
du moteur de repliement principal (folding_engine.py).

Références :
    - Frenkel & Smit, Understanding Molecular Simulation, 2e éd.
    - Kumar et al. (1992) J. Comput. Chem. 13:1011 — WHAM original
    - Sugita & Okamoto (1999) Chem. Phys. Lett. 314:141 — REMD original
    - Torrie & Valleau (1977) J. Comput. Phys. 23:187 — échantillonnage
      parapluie

Auteur : Jean-Luc Mercier
         Algorithmes d'échantillonnage, Institut Pasteur, Paris
"""

from .replica_exchange import ÉchangeDeRépliques
from .wham import WHAM
from .umbrella_sampling import ÉchantillonnageParapluie

__all__ = [
    "ÉchangeDeRépliques",
    "WHAM",
    "ÉchantillonnageParapluie",
]
