"""
Échantillonnage — collective-variable sampling methods for protein folding.

Modules:
    replica_exchange — Échange de Répliques (parallel tempering)
    wham            — Weighted Histogram Analysis Method
    umbrella_sampling — Umbrella sampling with WHAM
"""
from .replica_exchange import ÉchangeDeRépliques
from .wham import WHAM
from .umbrella_sampling import ÉchantillonnageParapluie
