# Protein Folding AI — Development Notes

## Project Overview

De novo protein folding engine based on Rosetta REF2015 energy function
with fragment insertion Monte Carlo sampling. The baseline `folding_engine.py`
is operational; this directory collects supporting modules.

## Repository

**URL: https://github.com/DKword17/protein-folding-ai**

Design scope:
- Energy landscape sampling (replica exchange, WHAM, umbrella sampling)
- GPU-accelerated pairwise potentials (CUDA)
- Force field parameter calibration
- Benchmark scoring (GDT_TS, RMSD, lDDT, TM-score)
- CI and regression testing

## Development Approach

Code is Python with performance-critical sections in C++/CUDA.
All branches merge into `main` via pull requests.

### Key Modules (planned)

| Module | Purpose |
|--------|---------|
| `sampling/` | Replica exchange MC, WHAM free energy, umbrella sampling |
| `kinetics/` | Velocity Verlet integrator, Langevin thermostat |
| `kernel/` | CUDA pairwise potential kernels |
| `params/` | Force field weight optimization via differential evolution |
| `benchmark/` | CASP scoring, decoy clustering |
| `tests/` | NVE conservation, integrator verification, benchmark accuracy |

## Verification

- NVE ensemble: 1000 steps ΔE < 0.01%
- GPU pair energy: < 1 ms per step (200-residue protein)
- 5 test proteins: mean RMSD < 4 Å from extended chain
- All CI tests pass (coverage > 85%)
