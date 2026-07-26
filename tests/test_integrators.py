#!/usr/bin/env python3
"""
test_integrators.py
====================

Integration tests for backbone rebuild geometry, fragment insertion,
and continuum-mechanics integrators.

Tests cover:
  1. RebuildBondGeometry  — 5 tests checking N-CA, CA-C, C-N(prev),
     C=O, and CA-CB bond lengths are within ±0.01 Å of ideal values.
  2. RebuildContinuity    — 1 test verifying a +0.1° phi perturbation
     yields a CA displacement below 0.1 Å.
  3. RebuildBoundaryConditions — 2 tests: single-residue no-crash and
     2-residue chain continuity.
  4. FragmentInsertionIntegrator — 2 tests: insertion changes dihedrals
     and rebuild after insertion yields valid bond lengths.
  5. VelocityVerletStub   — skipped when kinetics.verlet unavailable.
  6. LangevinStub         — skipped when kinetics.langevin unavailable.

Author: Priya Sharma, Quality Engineering, Indian Institute of Science, Bengaluru
"""

import unittest

import numpy as np

from folding_engine import Residue, Conformation, FragmentInsertionMC


def _make_conf(sequence="AAA"):
    """
    Build a minimal conformation with backbone coordinates rebuilt from
    idealised dihedral angles (phi=-60, psi=-45 for every residue).

    Args:
        sequence: One-letter amino acid codes (default "AAA").

    Returns:
        Conformation with valid backbone coordinates for residues
        at index 1 onward (index 0 has zero coordinates).
    """
    residues = [
        Residue(i + 1, "ALA", phi=-60.0, psi=-45.0) for i in range(len(sequence))
    ]
    conf = Conformation(residues=residues)
    sampler = FragmentInsertionMC.__new__(FragmentInsertionMC)
    for i in range(1, len(conf.residues)):
        FragmentInsertionMC._rebuild_backbone(sampler, conf, i, i + 1)
    return conf


# ======================================================================
# 1. TestRebuildBondGeometry
# ======================================================================

class TestRebuildBondGeometry(unittest.TestCase):
    """Verify that _rebuild_backbone produces idealised bond lengths."""

    def setUp(self):
        self.conf = _make_conf("AAA")
        self.residues = self.conf.residues

    # -- 1a. N-CA = 1.47 Å --------------------------------------------

    def test_n_ca_bond_length(self):
        """
        Kindly ensure the N-CA bond length equals 1.47 Å (±0.01) for
        every rebuilt residue.  The CA position is constructed as

            CA = N + [1.47*cos(phi), 1.47*sin(phi), 0]

        so the Euclidean distance should be 1.47.
        """
        for res in self.residues[1:]:
            d = np.linalg.norm(res.CA - res.N)
            self.assertAlmostEqual(
                d, 1.47, delta=0.01,
                msg=f"Residue {res.seq_index}: N-CA distance = {d:.4f} Å, "
                    f"expected 1.47 ± 0.01 Å"
            )

    # -- 1b. CA-C = 1.51 Å --------------------------------------------

    def test_ca_c_bond_length(self):
        """
        Kindly ensure the CA-C bond length equals 1.51 Å (±0.01) for
        every rebuilt residue.  The C position is constructed as

            C = CA + [1.51*cos(psi), 1.51*sin(psi), 0]

        so the Euclidean distance should be 1.51.
        """
        for res in self.residues[1:]:
            d = np.linalg.norm(res.C - res.CA)
            self.assertAlmostEqual(
                d, 1.51, delta=0.01,
                msg=f"Residue {res.seq_index}: CA-C distance = {d:.4f} Å, "
                    f"expected 1.51 ± 0.01 Å"
            )

    # -- 1c. C-N(prev) = 1.33 Å (peptide bond) -----------------------

    def test_c_n_prev_bond_length(self):
        """
        Please verify the peptide bond length C_i - N_{i+1} equals
        1.33 Å (±0.01).  Residue N is placed at prev.C + [0, 0, 1.33].
        """
        for i in range(1, len(self.residues)):
            d = np.linalg.norm(self.residues[i].N - self.residues[i - 1].C)
            self.assertAlmostEqual(
                d, 1.33, delta=0.01,
                msg=f"Residue {self.residues[i].seq_index}: C(prev)-N distance = "
                    f"{d:.4f} Å, expected 1.33 ± 0.01 Å"
            )

    # -- 1d. C=O = 1.23 Å --------------------------------------------

    def test_co_bond_length(self):
        """
        Kindly ensure the carbonyl C=O bond length equals 1.23 Å (±0.01)
        for every rebuilt residue.  O is built as

            O = C + perp * 1.23

        where perp is the unit vector perpendicular to the N-CA-C plane.
        """
        for res in self.residues[1:]:
            d = np.linalg.norm(res.O - res.C)
            self.assertAlmostEqual(
                d, 1.23, delta=0.01,
                msg=f"Residue {res.seq_index}: C=O distance = {d:.4f} Å, "
                    f"expected 1.23 ± 0.01 Å"
            )

    # -- 1e. CA-CB = 1.53 Å ------------------------------------------

    def test_ca_cb_bond_length(self):
        """
        Please verify the CA-CB bond length equals 1.53 Å (±0.01) for
        every rebuilt residue.  CB is built as

            CB = CA + cb_dir * 1.53

        where cb_dir = cross(N-CA, perp) normalised.
        """
        for res in self.residues[1:]:
            d = np.linalg.norm(res.CB - res.CA)
            self.assertAlmostEqual(
                d, 1.53, delta=0.01,
                msg=f"Residue {res.seq_index}: CA-CB distance = {d:.4f} Å, "
                    f"expected 1.53 ± 0.01 Å"
            )


# ======================================================================
# 2. TestRebuildContinuity
# ======================================================================

class TestRebuildContinuity(unittest.TestCase):
    """
    Verify that a small perturbation to one dihedral angle yields a
    proportionally small displacement in the atom positions, confirming
    numerical continuity of the rebuild procedure.
    """

    def test_phi_perturbation_ca_displacement(self):
        """
        Apply a +0.1° perturbation to phi of residue 2, then rebuild
        the backbone.  The CA displacement should remain below 0.1 Å,
        demonstrating that the rebuild is continuous with respect to
        small dihedral changes.
        """
        conf = _make_conf("AAAA")
        sampler = FragmentInsertionMC.__new__(FragmentInsertionMC)

        # Record reference CA position of residue 2 (index 1)
        ca_ref = conf.residues[1].CA.copy()

        # Perturb phi by +0.1°
        conf.residues[1].phi += 0.1

        # Rebuild from residue 1 onward to propagate the change
        FragmentInsertionMC._rebuild_backbone(sampler, conf, 1, len(conf.residues))

        ca_new = conf.residues[1].CA
        displacement = np.linalg.norm(ca_new - ca_ref)

        self.assertLess(
            displacement, 0.1,
            msg=f"CA displacement after +0.1° phi perturbation = "
                f"{displacement:.6f} Å, expected < 0.1 Å"
        )


# ======================================================================
# 3. TestRebuildBoundaryConditions
# ======================================================================

class TestRebuildBoundaryConditions(unittest.TestCase):
    """
    Ensure _rebuild_backbone handles edge cases without crashing and
    produces sensible geometry at chain boundaries.
    """

    def test_single_residue_no_crash(self):
        """
        A single-residue conformation has no preceding residue to
        provide a C atom anchor.  _rebuild_backbone should exit
        gracefully without raising an exception.
        """
        conf = _make_conf("A")
        sampler = FragmentInsertionMC.__new__(FragmentInsertionMC)
        try:
            FragmentInsertionMC._rebuild_backbone(sampler, conf, 0, 1)
        except Exception as exc:
            self.fail(
                f"_rebuild_backbone raised {type(exc).__name__} on a "
                f"single-residue conformation: {exc}"
            )

    def test_two_residue_continuity(self):
        """
        For a 2-residue chain, the peptide bond between residue 1 and
        residue 2 should have the standard C-N bond length of 1.33 Å
        (±0.01), confirming that the rebuild works correctly at the
        minimal chain length.
        """
        conf = _make_conf("AA")
        sampler = FragmentInsertionMC.__new__(FragmentInsertionMC)
        FragmentInsertionMC._rebuild_backbone(sampler, conf, 1, 2)

        d = np.linalg.norm(conf.residues[1].N - conf.residues[0].C)
        self.assertAlmostEqual(
            d, 1.33, delta=0.01,
            msg=f"2-residue chain: C-N peptide bond = {d:.4f} Å, "
                f"expected 1.33 ± 0.01 Å"
        )


# ======================================================================
# 4. TestFragmentInsertionIntegrator
# ======================================================================

class TestFragmentInsertionIntegrator(unittest.TestCase):
    """
    Verify that FragmentInsertionMC._insert_fragment modifies backbone
    dihedrals and that subsequent coordinate rebuild yields valid
    bond geometry.
    """

    def setUp(self):
        self.conf = _make_conf("AAAAA")
        self.sampler = FragmentInsertionMC.__new__(FragmentInsertionMC)

    def test_insert_fragment_changes_dihedrals(self):
        """
        After calling _insert_fragment on a contiguous stretch of
        residues, at least one of the phi or psi values should differ
        from its original value, confirming the fragment library
        sampling produces a conformational change.
        """
        start = 1
        frag_len = 3

        # Record original dihedrals
        orig = [
            (self.conf.residues[i].phi, self.conf.residues[i].psi)
            for i in range(start, start + frag_len)
        ]

        # Insert fragment (random dihedral perturbation)
        FragmentInsertionMC._insert_fragment(self.sampler, self.conf, start, frag_len)

        # Check at least one dihedral changed
        changed = False
        for i, (phi_orig, psi_orig) in enumerate(orig):
            idx = start + i
            if (abs(self.conf.residues[idx].phi - phi_orig) > 1e-6 or
                    abs(self.conf.residues[idx].psi - psi_orig) > 1e-6):
                changed = True
                break

        self.assertTrue(
            changed,
            msg="_insert_fragment did not change any dihedral angles "
                f"over residues {start}–{start + frag_len - 1}"
        )

    def test_rebuild_after_insertion_valid_bonds(self):
        """
        After fragment insertion, rebuild the backbone coordinates and
        verify that all standard bond lengths (N-CA, CA-C, C-N_prev,
        C=O, CA-CB) fall within ±0.01 Å of their ideal values for the
        rebuilt region.
        """
        start = 1
        frag_len = 3

        # Insert fragment and rebuild
        FragmentInsertionMC._insert_fragment(self.sampler, self.conf, start, frag_len)
        FragmentInsertionMC._rebuild_backbone(
            self.sampler, self.conf, start, start + frag_len
        )

        # Check bond lengths in the rebuilt region
        for idx in range(start, min(start + frag_len, len(self.conf.residues))):
            res = self.conf.residues[idx]

            # N-CA = 1.47
            d_n_ca = np.linalg.norm(res.CA - res.N)
            self.assertAlmostEqual(
                d_n_ca, 1.47, delta=0.01,
                msg=f"Residue {res.seq_index} after insertion+rebuild: "
                    f"N-CA = {d_n_ca:.4f} Å, expected 1.47 ± 0.01 Å"
            )

            # CA-C = 1.51
            d_ca_c = np.linalg.norm(res.C - res.CA)
            self.assertAlmostEqual(
                d_ca_c, 1.51, delta=0.01,
                msg=f"Residue {res.seq_index} after insertion+rebuild: "
                    f"CA-C = {d_ca_c:.4f} Å, expected 1.51 ± 0.01 Å"
            )

            # C=O = 1.23
            d_c_o = np.linalg.norm(res.O - res.C)
            self.assertAlmostEqual(
                d_c_o, 1.23, delta=0.01,
                msg=f"Residue {res.seq_index} after insertion+rebuild: "
                    f"C=O = {d_c_o:.4f} Å, expected 1.23 ± 0.01 Å"
            )

            # CA-CB = 1.53
            d_ca_cb = np.linalg.norm(res.CB - res.CA)
            self.assertAlmostEqual(
                d_ca_cb, 1.53, delta=0.01,
                msg=f"Residue {res.seq_index} after insertion+rebuild: "
                    f"CA-CB = {d_ca_cb:.4f} Å, expected 1.53 ± 0.01 Å"
            )

        # Peptide bond C_i - N_{i+1} = 1.33 for adjacent pairs
        # where both residues are within the rebuilt region
        for idx in range(start, min(start + frag_len, len(self.conf.residues)) - 1):
            d_c_n = np.linalg.norm(
                self.conf.residues[idx + 1].N - self.conf.residues[idx].C
            )
            self.assertAlmostEqual(
                d_c_n, 1.33, delta=0.01,
                msg=f"Peptide bond {idx}-{idx + 1} after insertion+rebuild: "
                    f"C-N = {d_c_n:.4f} Å, expected 1.33 ± 0.01 Å"
            )


# ======================================================================
# 5. TestVelocityVerletStub
# ======================================================================

class TestVelocityVerletStub(unittest.TestCase):
    """
    Stub for Velocity Verlet integrator tests.
    Skipped when the kinetics.verlet module is not available.
    """

    def test_velocity_verlet_import(self):
        """
        Attempt to import the Velocity Verlet integrator from
        kinetics.verlet.  If the module is absent, the test is
        skipped gracefully.
        """
        try:
            from kinetics.verlet import VelocityVerletIntegrator  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest(
                "kinetics.verlet module not available — "
                "Velocity Verlet integrator tests skipped"
            )


# ======================================================================
# 6. TestLangevinStub
# ======================================================================

class TestLangevinStub(unittest.TestCase):
    """
    Stub for Langevin dynamics integrator tests.
    Skipped when the kinetics.langevin module is not available.
    """

    def test_langevin_import(self):
        """
        Attempt to import the Langevin integrator from
        kinetics.langevin.  If the module is absent, the test is
        skipped gracefully.
        """
        try:
            from kinetics.langevin import LangevinIntegrator  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest(
                "kinetics.langevin module not available — "
                "Langevin integrator tests skipped"
            )


# ======================================================================
# Entry Point
# ======================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
