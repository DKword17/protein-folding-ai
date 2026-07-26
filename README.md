#!/usr/bin/env python3
"""
protein_folding_ai/README.md

Protein Folding AI — Fragment-based Ab Initio Folding Engine
=============================================================

A Rosetta-inspired fragment insertion Monte Carlo protein folding
engine with full energy function evaluation.

Features:
    - Fragment insertion MCMC (3-mer and 9-mer fragments)
    - Rosetta REF2015 energy function (LJ, HBond, solvation, Ramachandran, repulsive)
    - Dunbrack backbone-dependent rotamer sampling
    - Ramachandran preference scoring
    - SASA-based solvation energy
    - Extended chain initialization
    - Energy trajectory tracking

Architecture:
    folding_engine.py
        ├── RosettaEnergyFunction (5-term energy)
        ├── FragmentInsertionMC   (MCMC sampler)
        └── fold_protein          (top-level entry point)

References:
    - Alford et al. (2017) JCTC 13:3031 — Rosetta energy function
    - Dunbrack & Cohen (1997) Prot Sci 6:1661 — rotamer library
    - Bowers et al. (2006) SC06 — Folding@Home

Example:
    >>> from folding_engine import fold_protein
    >>> conf = fold_protein("AAAAKAAAAKAAAAK")
    >>> print(conf.total_energy, conf.energy_components)

Quick Start:
    $ python folding_engine.py
