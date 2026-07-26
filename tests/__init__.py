"""
tests/__init__.py
=================

Welcome dear colleagues, this is the **protein-folding-ai** test suite,
kindly maintained by **Priya Sharma**, Quality Engineering,
Indian Institute of Science, Bengaluru.

Purpose of this package
-----------------------
The present test suite is responsible for verifying the correctness,
numerical stability, and reproducibility of the protein folding engine
and its associated sampling and scoring modules.  The following areas
are covered:

    1.  Energy conservation under the NVE-like Monte Carlo protocol
        (``test_energy_conservation.py``)
    2.  Integrator correctness — backbone rebuild geometry, fragment
        insertion, and stubs for Velocity Verlet / Langevin dynamics
        (``test_integrators.py``)
    3.  Benchmark scoring precision — determinism, component consistency,
        RMSD / pLDDT tracking, and stubs for GDT_TS / lDDT / TM-score /
        SPICKER / Rg-SASA (``test_benchmark.py``)
    4.  Sampling convergence — replica exchange (t-REMD, H-REMD),
        WHAM self-consistency, umbrella sampling, bootstrap error
        estimation (``test_sampling.py``) — gracefully skipped until
        Jean-Luc's sampling module lands.

Testing methodology
-------------------
All tests follow the **Arrange-Act-Assert** (AAA) pattern with
exceptionally detailed comments so that any team member can readily
understand the purpose, procedure and expected outcome of every test
case.

Coverage target
---------------
As per the quality acceptance criteria, coverage must exceed **85 %**.
The CI pipeline enforces this gate.

Author
------
    Priya Sharma
    Quality Engineering Division
    Indian Institute of Science, Bengaluru
    dev/testing-priya branch
"""

# Nothing else required — this file marks tests/ as a Python package.
