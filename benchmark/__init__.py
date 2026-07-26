"""
benchmark
=========

Benchmarking and validation tools for the protein-folding-ai engine.

Modules
-------
casp_scorer
    CASP-standard structure quality metrics: GDT_TS, RMSD, lDDT, TM-score.
decoy_cluster
    SPICKER-based clustering of decoy ensembles.
rg_sasa
    Radius of gyration and solvent-accessible surface area.
"""

from benchmark.casp_scorer import (
    compute_rmsd,
    compute_gdt_ts,
    compute_lddt,
    compute_tm_score,
    score_structure,
    StructureScore,
    parse_pdb,
    kabsch_superimpose,
)
from benchmark.decoy_cluster import (
    spicker_cluster,
    compute_pairwise_rmsd,
    Cluster,
    ClusteringResult,
    get_representatives,
)
from benchmark.rg_sasa import (
    compute_rg,
    compute_sasa,
    compute_rg_from_conformation,
    compute_sasa_from_conformation,
)

__all__ = [
    # casp_scorer
    'compute_rmsd',
    'compute_gdt_ts',
    'compute_lddt',
    'compute_tm_score',
    'score_structure',
    'StructureScore',
    'parse_pdb',
    'kabsch_superimpose',
    # decoy_cluster
    'spicker_cluster',
    'compute_pairwise_rmsd',
    'Cluster',
    'ClusteringResult',
    'get_representatives',
    # rg_sasa
    'compute_rg',
    'compute_sasa',
    'compute_rg_from_conformation',
    'compute_sasa_from_conformation',
]
