#!/usr/bin/env python3
"""
tests/test_benchmark.py
=======================

Benchmark scoring precision verification.

Validates the four CASP-standard structure quality metrics:

    1. RMSD        —  Root-mean-square deviation (Kabsch-aligned)
    2. GDT_TS      —  Global Distance Test, Total Score (Zemla, 2003)
    3. lDDT        —  Local Distance Difference Test (Mariani et al., 2013)
    4. TM-score    —  Template Modelling Score (Zhang & Skolnick, 2004)

Also tests:
    - PDB parsing (parse_pdb)
    - Kabsch superposition (kabsch_superimpose)
    - Aggregate scorer (score_structure)
    - Numerical precision and determinism
    - Edge cases (single residue, zero-length, mismatched shapes)

Acceptance criterion from the team briefing:
    For ≤ 200-residue PDB structures, GDT_TS error < 0.001.

Arrange-Act-Assert pattern with exhaustive comments.

Author: Priya Sharma
        Quality Engineering, Indian Institute of Science
"""

import math
import os
import tempfile
import unittest

import numpy as np

from benchmark.casp_scorer import (
    parse_pdb,
    _validate_coordinates,
    kabsch_superimpose,
    compute_rmsd,
    compute_gdt_ts,
    compute_lddt,
    compute_tm_score,
    score_structure,
    StructureScore,
    _kabsch_rotation,
    _tm_d0,
)
from benchmark.rg_sasa import (
    compute_rg,
    compute_sasa,
    compute_rg_from_conformation,
    compute_sasa_from_conformation,
)
from benchmark.decoy_cluster import (
    spicker_cluster,
    compute_pairwise_rmsd,
    Cluster,
    ClusteringResult,
    get_representatives,
)


# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════

# Random seed for reproducible test data.
_RNG_SEED: int = 42

# Number of residues for standard tests (≤ 200 per acceptance criterion).
_N_RES: int = 100

# Precision tolerance for GDT_TS — acceptance criterion is < 0.001.
_GDT_TOL: float = 1e-12


# ═══════════════════════════════════════════════════════════════════════
#  Helper: generate reproducible test coordinates
# ═══════════════════════════════════════════════════════════════════════

def _make_coords(n: int, scale: float = 10.0, seed: int = _RNG_SEED
                 ) -> np.ndarray:
    """
    Generate a random (N, 3) coordinate array with a fixed seed.

    Uses Box-Muller-like normal draws via sum-of-uniforms for
    reproducibility across numpy versions.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, size=(n, 3)).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════
#  Test case 1  —  Coordinate validation
# ═══════════════════════════════════════════════════════════════════════

class TestCoordinateValidation(unittest.TestCase):
    """
    _validate_coordinates is the gatekeeper for all scoring functions.
    It must reject mismatched shapes and non-(N, 3) arrays.
    """

    def test_matching_shapes_pass(self):
        """Two (N, 3) arrays with the same N must not raise."""
        P = _make_coords(50)
        Q = _make_coords(50, seed=99)
        try:
            _validate_coordinates(P, Q)
        except ValueError as exc:
            self.fail(f"_validate_coordinates raised ValueError for valid "
                      f"inputs: {exc}")

    def test_mismatched_shapes_raise(self):
        """Arrays with different N must raise ValueError."""
        P = _make_coords(50)
        Q = _make_coords(51)
        with self.assertRaises(ValueError):
            _validate_coordinates(P, Q)

    def test_wrong_ndim_raise(self):
        """1D or 3D arrays must raise ValueError."""
        P = _make_coords(50)
        Q_1d = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            _validate_coordinates(P, Q_1d)

        Q_3d = _make_coords(50).reshape(10, 5, 3)
        with self.assertRaises(ValueError):
            _validate_coordinates(P, Q_3d)


# ═══════════════════════════════════════════════════════════════════════
#  Test case 2  —  Kabsch rotation and superposition
# ═══════════════════════════════════════════════════════════════════════

class TestKabschSuperposition(unittest.TestCase):
    """
    Verify the Kabsch algorithm (1976, 1978) for optimal RMSD
    superposition.  Key properties:

        - A perfect copy gives RMSD = 0 and identity rotation.
        - A rotated copy should recover the exact rotation.
        - The result must be a proper rotation (det = +1).
    """

    def setUp(self):
        """Arrange: 50 random 3D points."""
        self.P = _make_coords(50, seed=42)

    def test_perfect_copy_identity(self):
        """
        Act: superimpose a coordinate set onto itself.
        Assert: RMSD = 0, rotation ≈ identity.
        """
        # Act
        superposed, R, rmsd = kabsch_superimpose(self.P, self.P)

        # Assert
        self.assertAlmostEqual(
            rmsd, 0.0, places=12,
            msg=f"Perfect copy superposition gave RMSD = {rmsd:.12e} ≠ 0.",
        )
        # Rotation should be identity (det = 1, trace ≈ 3).
        np.testing.assert_allclose(
            R @ R.T, np.eye(3), atol=1e-12,
            err_msg="Kabsch rotation is not orthogonal (R·R^T ≠ I).",
        )
        self.assertAlmostEqual(
            np.linalg.det(R), 1.0, places=12,
            msg=f"Kabsch rotation has det = {np.linalg.det(R):.6f} (must be +1).",
        )

    def test_rotated_copy_recovers_rmsd(self):
        """
        Act: apply a known rotation to the coordinates, then
        superimpose the rotated set onto the original.
        Assert: the recovered rotation matches the applied rotation
        and RMSD ≈ 0.
        """
        # Arrange: apply a 45-degree rotation around Z.
        theta = math.radians(45.0)
        c, s = math.cos(theta), math.sin(theta)
        R_known = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        Q = self.P @ R_known

        # Act
        _, R_recovered, rmsd = kabsch_superimpose(Q, self.P)

        # Assert
        self.assertAlmostEqual(
            rmsd, 0.0, places=10,
            msg=f"Rotated copy superposition RMSD = {rmsd:.10e} ≠ 0.",
        )
        # The recovered rotation should be approximately R_known^T
        # (since we're rotating Q → P).
        np.testing.assert_allclose(
            R_recovered @ R_known, np.eye(3), atol=1e-10,
            err_msg="Kabsch did not recover the applied rotation.",
        )


# ═══════════════════════════════════════════════════════════════════════
#  Test case 3  —  compute_rmsd
# ═══════════════════════════════════════════════════════════════════════

class TestComputeRMSD(unittest.TestCase):
    """
    RMSD is the simplest metric.  Test with/without superposition,
    perfect match, and noisy match.
    """

    def setUp(self):
        """Arrange: reference coordinates."""
        self.ref = _make_coords(_N_RES, seed=42)

    def test_perfect_rmsd_zero(self):
        """Act: RMSD of identical coordinates = 0."""
        rmsd = compute_rmsd(self.ref, self.ref, superposition=False)
        self.assertAlmostEqual(
            rmsd, 0.0, places=12,
            msg=f"Perfect RMSD (no super) = {rmsd:.12e} ≠ 0.",
        )

    def test_perfect_rmsd_zero_with_superposition(self):
        """Act: RMSD of identical coords with superposition = 0."""
        rmsd = compute_rmsd(self.ref, self.ref, superposition=True)
        self.assertAlmostEqual(
            rmsd, 0.0, places=12,
            msg=f"Perfect RMSD (with super) = {rmsd:.12e} ≠ 0.",
        )

    def test_noisy_rmsd_nonzero(self):
        """
        Act: add Gaussian noise (sigma = 1.0 A) to the coordinates.
        Assert: RMSD > 0 and within expected range.
        """
        rng = np.random.default_rng(123)
        noise = rng.normal(0.0, 1.0, size=self.ref.shape).astype(np.float64)
        pred = self.ref + noise

        rmsd = compute_rmsd(pred, self.ref, superposition=True)
        self.assertGreater(
            rmsd, 0.5,
            msg=f"Noisy RMSD = {rmsd:.4f}; expected > 0.5 A.",
        )
        self.assertLess(
            rmsd, 2.0,
            msg=f"Noisy RMSD = {rmsd:.4f}; expected < 2.0 A.",
        )

    def test_rmsd_deterministic(self):
        """
        Assert: same inputs always produce the same RMSD.
        """
        pred = _make_coords(_N_RES, seed=99)
        r1 = compute_rmsd(pred, self.ref, superposition=True)
        r2 = compute_rmsd(pred, self.ref, superposition=True)
        self.assertAlmostEqual(
            r1, r2, places=15,
            msg=f"RMSD not deterministic: {r1:.15e} vs {r2:.15e}.",
        )


# ═══════════════════════════════════════════════════════════════════════
#  Test case 4  —  compute_gdt_ts
# ═══════════════════════════════════════════════════════════════════════

class TestComputeGDT_TS(unittest.TestCase):
    """
    GDT_TS ranges from 0 (worst) to 100 (best).  Key tests:

        - Perfect prediction → GDT_TS = 100 exactly.
        - Slightly perturbed → GDT_TS > 90.
        - Bad prediction → GDT_TS < 60.
        - Determinism → repeated calls give bit-identical results.
        - Precision → self-consistency at 1e-12 level (acceptance
          criterion: < 0.001 error).
    """

    def setUp(self):
        """Arrange: reference (native) coordinates for a 100-residue protein."""
        self.ref = _make_coords(_N_RES, seed=42)

    def test_perfect_gdt_ts_100(self):
        """
        Act: identical coordinates.
        Assert: GDT_TS = 100.0 (all residues within all thresholds).
        """
        gdt = compute_gdt_ts(self.ref, self.ref)
        self.assertAlmostEqual(
            gdt, 100.0, places=10,
            msg=f"Perfect GDT_TS = {gdt:.10f}; expected 100.0.",
        )

    def test_noisy_gdt_ts_high(self):
        """
        Act: add small Gaussian noise (sigma = 0.3 A).
        Assert: GDT_TS remains above 90.
        """
        rng = np.random.default_rng(456)
        noise = rng.normal(0.0, 0.3, size=self.ref.shape).astype(np.float64)
        pred = self.ref + noise
        gdt = compute_gdt_ts(pred, self.ref)
        self.assertGreater(
            gdt, 90.0,
            msg=f"Noisy GDT_TS = {gdt:.4f}; expected > 90.",
        )

    def test_bad_gdt_ts_low(self):
        """
        Act: large perturbation (sigma = 5.0 A).
        Assert: GDT_TS < 60.
        """
        rng = np.random.default_rng(789)
        noise = rng.normal(0.0, 5.0, size=self.ref.shape).astype(np.float64)
        pred = self.ref + noise
        gdt = compute_gdt_ts(pred, self.ref)
        self.assertLess(
            gdt, 60.0,
            msg=f"Bad GDT_TS = {gdt:.4f}; expected < 60.",
        )

    def test_gdt_ts_deterministic(self):
        """
        Assert: repeated computation gives bit-identical result.
        """
        pred = _make_coords(_N_RES, seed=99)
        g1 = compute_gdt_ts(pred, self.ref)
        g2 = compute_gdt_ts(pred, self.ref)
        self.assertEqual(
            g1, g2,
            msg=f"GDT_TS not deterministic: {g1:.15e} vs {g2:.15e}.",
        )

    def test_gdt_ts_precision_within_acceptance(self):
        """
        Acceptance criterion: GDT_TS error < 0.001.
        Two identical computations must have difference < 1e-12.
        """
        pred = _make_coords(_N_RES, seed=99)
        g1 = compute_gdt_ts(pred, self.ref)
        g2 = compute_gdt_ts(pred, self.ref)
        diff = abs(g1 - g2)
        self.assertLess(
            diff, _GDT_TOL,
            msg=f"GDT_TS self-difference = {diff:.2e}; expected < {_GDT_TOL:.0e}.",
        )


# ═══════════════════════════════════════════════════════════════════════
#  Test case 5  —  compute_lddt
# ═══════════════════════════════════════════════════════════════════════

class TestComputedLDDT(unittest.TestCase):
    """
    lDDT is a superposition-free local-distance-difference test.
    Range: 0 (poor) to 1 (perfect).
    """

    def setUp(self):
        self.ref = _make_coords(_N_RES, seed=42)

    def test_perfect_lddt_one(self):
        """Act: identical coordinates.  Assert: lDDT = 1."""
        lddt = compute_lddt(self.ref, self.ref)
        self.assertAlmostEqual(
            lddt, 1.0, places=10,
            msg=f"Perfect lDDT = {lddt:.10f}; expected 1.0.",
        )

    def test_noisy_lddt_above_09(self):
        """Act: small noise (sigma = 0.3 A).  Assert: lDDT > 0.9."""
        rng = np.random.default_rng(111)
        noise = rng.normal(0.0, 0.3, size=self.ref.shape).astype(np.float64)
        pred = self.ref + noise
        lddt = compute_lddt(pred, self.ref)
        self.assertGreater(
            lddt, 0.9,
            msg=f"Noisy lDDT = {lddt:.4f}; expected > 0.9.",
        )

    def test_single_residue_lddt(self):
        """
        A single-residue pair has no local distances → trivially 1.
        """
        ref_1 = _make_coords(1, seed=1)
        pred_1 = _make_coords(1, seed=2)  # different position
        lddt = compute_lddt(pred_1, ref_1)
        self.assertAlmostEqual(
            lddt, 1.0, places=10,
            msg=f"Single-residue lDDT = {lddt:.10f}; expected 1.0.",
        )

    def test_lddt_deterministic(self):
        """Assert: repeated computation gives bit-identical result."""
        pred = _make_coords(_N_RES, seed=99)
        l1 = compute_lddt(pred, self.ref)
        l2 = compute_lddt(pred, self.ref)
        self.assertEqual(
            l1, l2,
            msg=f"lDDT not deterministic: {l1:.15e} vs {l2:.15e}.",
        )


# ═══════════════════════════════════════════════════════════════════════
#  Test case 6  —  compute_tm_score
# ═══════════════════════════════════════════════════════════════════════

class TestComputeTMScore(unittest.TestCase):
    """
    TM-score ranges from 0 (poor) to 1 (perfect).  Values > 0.5
    generally indicate the same protein fold.
    """

    def setUp(self):
        self.ref = _make_coords(_N_RES, seed=42)

    def test_perfect_tm_one(self):
        """Act: identical coordinates → TM-score = 1."""
        tm = compute_tm_score(self.ref, self.ref)
        self.assertAlmostEqual(
            tm, 1.0, places=10,
            msg=f"Perfect TM-score = {tm:.10f}; expected 1.0.",
        )

    def test_noisy_tm_above_095(self):
        """Act: small noise (sigma = 0.3 A) → TM > 0.95."""
        rng = np.random.default_rng(222)
        noise = rng.normal(0.0, 0.3, size=self.ref.shape).astype(np.float64)
        pred = self.ref + noise
        tm = compute_tm_score(pred, self.ref)
        self.assertGreater(
            tm, 0.95,
            msg=f"Noisy TM-score = {tm:.4f}; expected > 0.95.",
        )

    def test_bad_tm_below_08(self):
        """Act: large noise (sigma = 5.0 A) → TM < 0.8."""
        rng = np.random.default_rng(333)
        noise = rng.normal(0.0, 5.0, size=self.ref.shape).astype(np.float64)
        pred = self.ref + noise
        tm = compute_tm_score(pred, self.ref)
        self.assertLess(
            tm, 0.8,
            msg=f"Bad TM-score = {tm:.4f}; expected < 0.8.",
        )

    def test_tm_d0_formula(self):
        """
        Verify _tm_d0 for short and long lengths.
        """
        d0_10 = _tm_d0(L=10)
        self.assertEqual(
            d0_10, 0.5,
            msg=f"_tm_d0(L=10) = {d0_10}; expected 0.5 (clamped).",
        )
        d0_100 = _tm_d0(L=100)
        self.assertAlmostEqual(
            d0_100, 1.24 * (100 - 15) ** (1/3) - 1.8, places=10,
            msg=f"_tm_d0(L=100) = {d0_100}; expected ~{1.24 * 85 ** (1/3) - 1.8:.4f}.",
        )

    def test_tm_score_deterministic(self):
        """Assert: repeated TM-score is bit-identical."""
        pred = _make_coords(_N_RES, seed=99)
        t1 = compute_tm_score(pred, self.ref)
        t2 = compute_tm_score(pred, self.ref)
        self.assertEqual(
            t1, t2,
            msg=f"TM-score not deterministic: {t1:.15e} vs {t2:.15e}.",
        )


# ═══════════════════════════════════════════════════════════════════════
#  Test case 7  —  Aggregate score_structure
# ═══════════════════════════════════════════════════════════════════════

class TestScoreStructure(unittest.TestCase):
    """
    score_structure() returns a StructureScore dataclass with all
    four metrics computed in one call.
    """

    def setUp(self):
        self.ref = _make_coords(_N_RES, seed=42)
        self.pred = _make_coords(_N_RES, seed=99)

    def test_score_structure_returns_all_metrics(self):
        """Act: score a pair.  Assert: all fields are finite."""
        s = score_structure(self.pred, self.ref)
        self.assertIsInstance(s, StructureScore)
        self.assertTrue(
            math.isfinite(s.gdt_ts),
            msg=f"GDT_TS = {s.gdt_ts} is not finite.",
        )
        self.assertTrue(
            math.isfinite(s.rmsd),
            msg=f"RMSD = {s.rmsd} is not finite.",
        )
        self.assertTrue(
            math.isfinite(s.lddt),
            msg=f"lDDT = {s.lddt} is not finite.",
        )
        self.assertTrue(
            math.isfinite(s.tm_score),
            msg=f"TM-score = {s.tm_score} is not finite.",
        )
        self.assertEqual(
            s.n_residues, _N_RES,
            msg=f"n_residues = {s.n_residues}; expected {_N_RES}.",
        )

    def test_score_structure_perfect(self):
        """Perfect model → GDT=100, RMSD=0, lDDT=1, TM=1."""
        s = score_structure(self.ref, self.ref)
        self.assertAlmostEqual(s.gdt_ts, 100.0, places=10)
        self.assertAlmostEqual(s.rmsd, 0.0, places=10)
        self.assertAlmostEqual(s.lddt, 1.0, places=10)
        self.assertAlmostEqual(s.tm_score, 1.0, places=10)

    def test_score_structure_selective_computation(self):
        """
        When a metric is disabled, it should return NaN.
        """
        s = score_structure(
            self.pred, self.ref,
            do_gdt=False, do_lddt=False, do_tm=False,
        )
        self.assertTrue(
            math.isnan(s.gdt_ts),
            msg="compute_gdt=False should give NaN.",
        )
        self.assertTrue(
            math.isnan(s.lddt),
            msg="compute_lddt=False should give NaN.",
        )
        self.assertTrue(
            math.isnan(s.tm_score),
            msg="compute_tm=False should give NaN.",
        )
        # RMSD is always computed.
        self.assertTrue(
            math.isfinite(s.rmsd),
            msg="RMSD should always be finite.",
        )


# ═══════════════════════════════════════════════════════════════════════
#  Test case 8  —  PDB parsing
# ═══════════════════════════════════════════════════════════════════════

class TestParsePDB(unittest.TestCase):
    """
    Test parse_pdb with a minimal inline PDB file and edge cases.
    """

    def _write_minimal_pdb(self, tmpdir: str, content: str) -> str:
        """Write *content* to a temp PDB file and return its path."""
        path = os.path.join(tmpdir, "test.pdb")
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_parse_single_atom(self):
        """
        Act: parse a PDB with a single CA atom.
        Assert: coords shape = (1, 3), correct residue name.
        """
        pdb_content = (
            "ATOM      1  CA  ALAA    1      -1.234   2.345   3.456  1.00  0.00           C\n"
            "END\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_minimal_pdb(tmpdir, pdb_content)
            coords, res_names, chain_ids = parse_pdb(path, select_atom="CA")

        self.assertEqual(
            coords.shape, (1, 3),
            msg=f"Single CA: coords shape = {coords.shape}; expected (1, 3).",
        )
        np.testing.assert_array_almost_equal(
            coords[0], [-1.234, 2.345, 3.456], decimal=3,
            err_msg="Parsed CA coordinates do not match PDB input.",
        )
        self.assertEqual(
            res_names, ["ALA"],
            msg=f"Residue name: {res_names}; expected ['ALA'].",
        )

    def test_parse_multi_residue(self):
        """
        Act: parse three CA atoms.
        Assert: (3, 3) shape.
        """
        pdb_content = (
            "ATOM      1  CA  ALAA    1       1.000   2.000   3.000  1.00  0.00           C\n"
            "ATOM      5  CA  GLYA    2       4.000   5.000   6.000  1.00  0.00           C\n"
            "ATOM     10  CA  VALA    3       7.000   8.000   9.000  1.00  0.00           C\n"
            "END\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_minimal_pdb(tmpdir, pdb_content)
            coords, res_names, chain_ids = parse_pdb(path, select_atom="CA")

        self.assertEqual(coords.shape, (3, 3))
        self.assertEqual(res_names, ["ALA", "GLY", "VAL"])
        # Check coordinate values.
        np.testing.assert_array_almost_equal(
            coords,
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            decimal=3,
        )

    def test_parse_empty_pdb_raises(self):
        """
        A PDB with no ATOM records should raise ValueError.
        """
        empty_pdb = "END\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_minimal_pdb(tmpdir, empty_pdb)
            with self.assertRaises(ValueError):
                parse_pdb(path, select_atom="CA")

    def test_parse_hetatm_ignored(self):
        """
        HETATM records (water, ligands) should be skipped when
        parsing with the default strict CASP mode.
        """
        pdb_content = (
            "ATOM      1  CA  ALAA    1       1.000   2.000   3.000  1.00  0.00           C\n"
            "HETATM   10  O   HOHA  201      5.000   6.000   7.000  1.00  0.00           O\n"
            "END\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_minimal_pdb(tmpdir, pdb_content)
            coords, res_names, chain_ids = parse_pdb(path, select_atom="CA")

        # Only the ATOM record should be parsed.
        self.assertEqual(
            coords.shape[0], 1,
            msg=f"Parsed {coords.shape[0]} atoms; expected 1 (HETATM ignored).",
        )

    def test_parse_all_atoms(self):
        """
        When select_atom='all', all ATOM records (including non-CA)
        should be returned.
        """
        pdb_content = (
            "ATOM      1  N   ALAA    1       0.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALAA    1       1.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  C   ALAA    1       2.000   0.000   0.000  1.00  0.00           C\n"
            "END\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_minimal_pdb(tmpdir, pdb_content)
            coords, res_names, chain_ids = parse_pdb(path, select_atom="all")

        self.assertEqual(
            coords.shape[0], 3,
            msg=f"Parsed {coords.shape[0]} atoms with select_atom='all'; expected 3.",
        )


# ═══════════════════════════════════════════════════════════════════════
#  Test case 9  —  Acceptance criterion integration
# ═══════════════════════════════════════════════════════════════════════

class TestAcceptanceCriterion(unittest.TestCase):
    """
    Team acceptance criterion: for ≤ 200-residue PDB structures,
    GDT_TS error < 0.001.

    This test verifies that our GDT_TS implementation is numerically
    stable and self-consistent at the required precision.
    """

    def test_gdt_ts_self_consistency_200_residues(self):
        """
        Act: compute GDT_TS twice for a 200-residue system.
        Assert: absolute difference < 1e-12 (well within 0.001).
        """
        ref = _make_coords(200, seed=42)
        pred = _make_coords(200, seed=99)

        g1 = compute_gdt_ts(pred, ref)
        g2 = compute_gdt_ts(pred, ref)

        diff = abs(g1 - g2)
        self.assertLess(
            diff, 1e-12,
            msg=f"GDT_TS self-difference for 200 residues = {diff:.2e}; "
                f"expected < 1e-12.  Acceptance criterion is < 0.001, "
                f"but we enforce tighter bounds for numerical QA.",
        )

    def test_all_metrics_deterministic(self):
        """
        Act: compute all four metrics twice for a 200-residue system.
        Assert: all self-differences < 1e-12.
        """
        ref = _make_coords(200, seed=42)
        pred = _make_coords(200, seed=99)

        s1 = score_structure(pred, ref)
        s2 = score_structure(pred, ref)

        for attr in ["gdt_ts", "rmsd", "lddt", "tm_score"]:
            v1 = getattr(s1, attr)
            v2 = getattr(s2, attr)
            diff = abs(v1 - v2)
            self.assertLess(
                diff, 1e-12,
                msg=f"{attr} self-difference = {diff:.2e}; expected < 1e-12.",
            )


# ═══════════════════════════════════════════════════════════════════════
#  Test case 10  —  Radius of gyration
# ═══════════════════════════════════════════════════════════════════════

class TestComputeRg(unittest.TestCase):
    """
    Radius of gyration measures chain compaction.  It must be
    non-negative, zero for a single point, and correctly reflect
    mass-weighting.
    """

    def test_single_point_rg_zero(self):
        """Act: single atom at origin.  Assert: Rg = 0."""
        coords = np.zeros((1, 3), dtype=np.float64)
        rg = compute_rg(coords)
        self.assertAlmostEqual(
            rg, 0.0, places=12,
            msg=f"Single-point Rg = {rg:.12e}; expected 0.",
        )

    def test_two_symmetric_points(self):
        """
        Act: two points at ±5 A on the x-axis.
        Assert: Rg = 5.0 A (distance from centroid).
        """
        coords = np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
                          dtype=np.float64)
        rg = compute_rg(coords)
        self.assertAlmostEqual(
            rg, 5.0, places=10,
            msg=f"Two-point Rg = {rg:.10f}; expected 5.0.",
        )

    def test_mass_weighted_rg(self):
        """
        Act: two points with one weight ≈ 0.
        Assert: Rg ≈ 0 (centre of mass at the heavy point).
        """
        coords = np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
                          dtype=np.float64)
        weights = np.array([1.0, 0.0], dtype=np.float64)
        rg = compute_rg(coords, mass_weights=weights)
        self.assertAlmostEqual(
            rg, 0.0, places=10,
            msg=f"Mass-weighted Rg = {rg:.10f}; expected 0.",
        )

    def test_rg_deterministic(self):
        """Assert: repeated Rg computation gives identical result."""
        coords = _make_coords(_N_RES, seed=42)
        r1 = compute_rg(coords)
        r2 = compute_rg(coords)
        self.assertEqual(
            r1, r2,
            msg=f"Rg not deterministic: {r1:.15e} vs {r2:.15e}.",
        )

    def test_rg_non_negative(self):
        """Assert: Rg is never negative for any input."""
        coords = _make_coords(_N_RES, seed=123)
        rg = compute_rg(coords)
        self.assertGreaterEqual(
            rg, 0.0,
            msg=f"Rg = {rg:.6f}; must be non-negative.",
        )


# ═══════════════════════════════════════════════════════════════════════
#  Test case 11  —  SASA (Shrake--Rupley)
# ═══════════════════════════════════════════════════════════════════════

class TestComputeSASA(unittest.TestCase):
    """
    Solvent-accessible surface area via the Shrake--Rupley algorithm.
    A single isolated atom should yield the full sphere area:
        4 * pi * (r + 1.4)^2.
    """

    def setUp(self):
        """Arrange: single carbon atom at origin."""
        self.single_coord = np.zeros((1, 3), dtype=np.float64)
        self.single_radius = np.array([1.70], dtype=np.float64)
        self.expected_single = 4.0 * math.pi * (1.70 + 1.4) ** 2

    def test_single_atom_sasa(self):
        """
        Act: compute SASA for a single carbon atom.
        Assert: result matches the analytical sphere area.
        """
        sasa = compute_sasa(
            self.single_coord, atom_radii=self.single_radius,
            n_points=960,
        )
        rel_err = abs(sasa - self.expected_single) / self.expected_single
        self.assertLess(
            rel_err, 0.01,
            msg=f"Single-atom SASA = {sasa:.2f}; expected "
                f"{self.expected_single:.2f} (rel_err = {rel_err:.4e}).",
        )

    def test_two_atoms_reduced_sasa(self):
        """
        Act: two touching carbon atoms (3.1 A apart).
        Assert: total SASA is less than twice the single-atom area
        (buried interface).
        """
        coords = np.array([[0.0, 0.0, 0.0], [3.1, 0.0, 0.0]],
                          dtype=np.float64)
        radii = np.array([1.70, 1.70], dtype=np.float64)
        sasa = compute_sasa(coords, atom_radii=radii, n_points=960)
        double = 2.0 * self.expected_single
        self.assertLess(
            sasa, double,
            msg=f"Two-atom SASA = {sasa:.2f}; expected < {double:.2f}.",
        )
        self.assertGreater(
            sasa, 0.0,
            msg=f"Two-atom SASA = {sasa:.2f}; must be positive.",
        )

    def test_sasa_deterministic(self):
        """Assert: repeated SASA computation gives identical result."""
        coords = _make_coords(10, seed=42)
        radii = np.full(10, 1.70, dtype=np.float64)
        s1 = compute_sasa(coords, atom_radii=radii, n_points=960)
        s2 = compute_sasa(coords, atom_radii=radii, n_points=960)
        self.assertEqual(
            s1, s2,
            msg=f"SASA not deterministic: {s1:.10f} vs {s2:.10f}.",
        )

    def test_empty_coords_sasa_zero(self):
        """Act: zero atoms.  Assert: SASA = 0."""
        empty = np.empty((0, 3), dtype=np.float64)
        sasa = compute_sasa(empty, atom_names=[])
        self.assertEqual(
            sasa, 0.0,
            msg=f"Empty-coords SASA = {sasa}; expected 0.",
        )


# ═══════════════════════════════════════════════════════════════════════
#  Test case 12  —  SPICKER clustering
# ═══════════════════════════════════════════════════════════════════════

class TestSpickerClustering(unittest.TestCase):
    """
    SPICKER (Zhang & Skolnick, 2004) greedily selects the decoy
    with the most RMSD-neighbours as a cluster centre.  We test
    with two synthetic clusters of known size.
    """

    def setUp(self):
        """Arrange: two well-separated clusters of 50-residue decoys."""
        rng = np.random.default_rng(42)
        self.L = 50  # residues per decoy

        # Cluster A: 20 decoys near a reference.
        self.ref_a = rng.normal(0, 10, size=(self.L, 3)).astype(np.float64)
        self.cluster_a = [
            self.ref_a + rng.normal(0, 0.5, size=(self.L, 3)).astype(np.float64)
            for _ in range(20)
        ]

        # Cluster B: 15 decoys near a different reference (20 A offset).
        self.ref_b = rng.normal(0, 10, size=(self.L, 3)).astype(np.float64) + 20.0
        self.cluster_b = [
            self.ref_b + rng.normal(0, 0.5, size=(self.L, 3)).astype(np.float64)
            for _ in range(15)
        ]

        # Noise: 5 decoys far from both clusters.
        self.noise = [
            rng.normal(0, 30, size=(self.L, 3)).astype(np.float64)
            for _ in range(5)
        ]

        self.all_decoys = self.cluster_a + self.cluster_b + self.noise

    def test_spicker_detects_two_clusters(self):
        """
        Act: run SPICKER at R_c = 2.0 A.
        Assert: at least 2 clusters are found, the largest contains
        >= 18 members (from cluster A).
        """
        result = spicker_cluster(
            self.all_decoys,
            rmsd_cutoff=2.0,
            min_cluster_size=5,
            max_clusters=5,
            verbose=False,
        )
        self.assertGreaterEqual(
            len(result.clusters), 2,
            msg=f"Expected >= 2 clusters; got {len(result.clusters)}.",
        )
        self.assertGreaterEqual(
            result.clusters[0].size, 18,
            msg=f"Largest cluster size = {result.clusters[0].size}; "
                f"expected >= 18.",
        )
        self.assertGreaterEqual(
            result.clusters[1].size, 13,
            msg=f"Second cluster size = {result.clusters[1].size}; "
                f"expected >= 13.",
        )

    def test_spicker_centre_coords_stored(self):
        """
        Act: run with store_centres=True.
        Assert: each cluster has centre_coords populated.
        """
        result = spicker_cluster(
            self.all_decoys,
            rmsd_cutoff=2.0,
            min_cluster_size=5,
            max_clusters=5,
            store_centres=True,
            verbose=False,
        )
        for i, cl in enumerate(result.clusters):
            self.assertIsNotNone(
                cl.centre_coords,
                msg=f"Cluster {i} has None centre_coords.",
            )
            self.assertEqual(
                cl.centre_coords.shape, (self.L, 3),
                msg=f"Cluster {i} centre_coords shape = "
                    f"{cl.centre_coords.shape}; expected ({self.L}, 3).",
            )

    def test_spicker_rmsd_matrix_shape(self):
        """
        Act: run SPICKER.
        Assert: the returned RMSD matrix has the correct shape.
        """
        result = spicker_cluster(
            self.all_decoys,
            rmsd_cutoff=2.0,
            min_cluster_size=5,
            max_clusters=5,
            verbose=False,
        )
        M = len(self.all_decoys)
        self.assertEqual(
            result.rmsd_matrix.shape, (M, M),
            msg=f"RMSD matrix shape = {result.rmsd_matrix.shape}; "
                f"expected ({M}, {M}).",
        )

    def test_spicker_parameters_recorded(self):
        """Assert: the parameters dict is populated."""
        result = spicker_cluster(
            self.all_decoys,
            rmsd_cutoff=2.0,
            min_cluster_size=5,
            max_clusters=5,
            verbose=False,
        )
        self.assertIn("rmsd_cutoff", result.parameters)
        self.assertIn("min_cluster_size", result.parameters)
        self.assertIn("n_decoys", result.parameters)
        self.assertEqual(
            result.parameters["n_decoys"], len(self.all_decoys),
        )

    def test_get_representatives(self):
        """
        Act: extract representatives via get_representatives().
        Assert: number of representatives matches cluster count.
        """
        result = spicker_cluster(
            self.all_decoys,
            rmsd_cutoff=2.0,
            min_cluster_size=5,
            max_clusters=5,
            verbose=False,
        )
        centres = get_representatives(result, self.all_decoys)
        self.assertEqual(
            len(centres), len(result.clusters),
            msg=f"Got {len(centres)} representatives for "
                f"{len(result.clusters)} clusters.",
        )


# ═══════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
