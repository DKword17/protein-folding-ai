#!/usr/bin/env python3
"""
protein_folding_ai/tests/test_integrators.py
=============================================

Integration-level tests for the fragment-insertion Monte Carlo
backbone rebuild routine and related coordinate-construction logic.

Tests verify:
    - Idealised bond geometry after _rebuild_backbone
    - Continuity / numerical stability under small dihedral perturbations
    - Boundary conditions (single-residue, 2-residue chains)
    - Fragment insertion mutates phi/psi while preserving bond geometry
    - Stub/skip for future kinetic integrators (velocity Verlet, Langevin)

Every assert carries a descriptive error message to expedite debugging.

Author: Priya Sharma, Quality Engineering,
        Indian Institute of Science, Bengaluru
"""

import math
import random
import unittest

import numpy as np

from folding_engine import (
    FragmentInsertionMC,
    Conformation,
    Residue,
    RosettaEnergyFunction,
)


# ─── Helper: build a small test conformation ──────────────────────────

def _build_conf(sequence="AAA", phi=-60.0, psi=-45.0):
    """
    Construct a *Conformation* from *sequence* (one-letter codes) and
    uniform backbone dihedrals, then run the fragment-insertion MC
    backbone rebuild so that every residue after the first has valid
    coordinates.

    Returns
    -------
    (Conformation, FragmentInsertionMC)
        The built conformation and the sampler instance used to rebuild it.
    """
    residues = [
        Residue(i + 1, "ALA", phi=phi, psi=psi)
        for i in range(len(sequence))
    ]
    conf = Conformation(residues=residues)

    # Create a sampler with a dummy energy function — we only need
    # access to _rebuild_backbone and _insert_fragment.
    sampler = FragmentInsertionMC.__new__(FragmentInsertionMC)
    FragmentInsertionMC.__init__(sampler, lambda c: 0)  # dummy energy-fn

    # Rebuild coordinates sequentially so each residue's C is available
    # for the N of the next residue.
    for i in range(1, len(conf.residues)):
        FragmentInsertionMC._rebuild_backbone(sampler, conf, i, i + 1)

    return conf, sampler


# ─── 1. TestRebuildBondGeometry ──────────────────────────────────────

class TestRebuildBondGeometry(unittest.TestCase):
    """
    Verify that every peptide-backbone and side-chain bond set by
    ``_rebuild_backbone`` matches its idealised target length.
    """

    _BOND_TOL  = 0.01   # Å — tolerance for all bond-length checks
    _LEN_N_CA  = 1.47   # Å
    _LEN_CA_C  = 1.51   # Å
    _LEN_C_N   = 1.33   # Å (peptide bond)
    _LEN_C_O   = 1.23   # Å
    _LEN_CA_CB = 1.53   # Å

    @classmethod
    def setUpClass(cls):
        """Build a 3-residue all-alanine chain once for all tests."""
        cls.conf, cls.sampler = _build_conf("AAA", phi=-60.0, psi=-45.0)
        # Residues that were rebuilt: indices 1 and 2 (0-indexed).
        # Index 0 remains at default (zero) coordinates.
        cls.built_indices = [1, 2]

    # -- convenience helpers -------------------------------------------

    def _check_bond(self, res_idx, atom_a, atom_b, expected):
        """Assert |atom_a - atom_b| ≈ *expected* for one residue."""
        res = self.conf.residues[res_idx]
        a = getattr(res, atom_a)
        b = getattr(res, atom_b)
        actual = np.linalg.norm(a - b)
        self.assertAlmostEqual(
            actual, expected, delta=self._BOND_TOL,
            msg=(
                f"Residue {res.seq_index} ({atom_a}–{atom_b}): "
                f"expected {expected:.3f} Å, got {actual:.4f} Å"
            ),
        )

    # -- individual test methods ---------------------------------------

    def test_N_CA_bond_length(self):
        """
        N–CA bond length must be 1.47 Å for every rebuilt residue.
        -----------------------------------------------------------------
        The rebuild code places CA at N + [1.47·cos(φ), 1.47·sin(φ), 0].
        """
        for idx in self.built_indices:
            self._check_bond(idx, "N", "CA", self._LEN_N_CA)

    def test_CA_C_bond_length(self):
        """
        CA–C bond length must be 1.51 Å for every rebuilt residue.
        -----------------------------------------------------------------
        The rebuild code places C at CA + [1.51·cos(ψ), 1.51·sin(ψ), 0].
        """
        for idx in self.built_indices:
            self._check_bond(idx, "CA", "C", self._LEN_CA_C)

    def test_C_N_peptide_bond_length(self):
        """
        C–N peptide bond must be 1.33 Å between successive residues.
        -----------------------------------------------------------------
        For residue pair (i-1 → i), this is the distance from residue
        (i-1).C to residue(i).N.  The rebuild code sets
        residue(i).N = residue(i-1).C + [0, 0, 1.33].
        """
        residues = self.conf.residues
        for i in range(1, len(residues)):
            dist = np.linalg.norm(residues[i].N - residues[i - 1].C)
            self.assertAlmostEqual(
                dist, self._LEN_C_N, delta=self._BOND_TOL,
                msg=(
                    f"Peptide bond C({i})–N({i + 1}): "
                    f"expected {self._LEN_C_N:.3f} Å, got {dist:.4f} Å"
                ),
            )

    def test_C_O_bond_length(self):
        """
        C=O bond length must be 1.23 Å for every rebuilt residue.
        -----------------------------------------------------------------
        The rebuild code places O = C + perp · 1.23 where perp is a
        unit vector orthogonal to the N–CA–C plane.
        """
        for idx in self.built_indices:
            self._check_bond(idx, "C", "O", self._LEN_C_O)

    def test_CA_CB_bond_length(self):
        """
        CA–CB bond length must be 1.53 Å for every rebuilt residue.
        -----------------------------------------------------------------
        The rebuild code places CB = CA + cb_dir · 1.53 where cb_dir is
        a unit vector orthogonal to both N–CA and the plane normal.
        """
        for idx in self.built_indices:
            self._check_bond(idx, "CA", "CB", self._LEN_CA_CB)


# ─── 2. TestRebuildContinuity ────────────────────────────────────────

class TestRebuildContinuity(unittest.TestCase):
    """
    A small perturbation to one residue's phi should produce only a
    minimal change in the CA coordinates after re-running the backbone
    rebuild — verifying numerical continuity of the reconstruction.
    """

    _PHI_PERTURB = 0.1    # degrees
    _RMSD_LIMIT  = 0.1    # Å — upper bound for CA RMS displacement

    def test_continuity_under_small_phi_change(self):
        """
        Perturb residue 2 (0-indexed: 1) phi by +0.1°, rebuild, and
        verify that the CA RMS displacement across all residues stays
        below 0.1 Å.
        -----------------------------------------------------------------
        A structurally sound rebuild should respond only linearly to a
        tiny dihedral change; large displacements would indicate a bug
        in the coordinate propagation logic.
        """
        conf, sampler = _build_conf("AAA", phi=-60.0, psi=-45.0)

        # --- Arrange: capture baseline CA positions -------------------
        ca_before = [np.copy(r.CA) for r in conf.residues]

        # --- Act: perturb residue 2 (1-indexed) phi -------------------
        conf.residues[1].phi += self._PHI_PERTURB   # -60.0 → -59.9

        # Rebuild from the perturbed residue to the chain end.
        # Each call rebuilds one residue using the *previous* residue's C,
        # which may itself have been just updated — iterate sequentially.
        for i in range(2, len(conf.residues) + 1):
            sampler._rebuild_backbone(conf, i, i + 1)

        # --- Assert: RMS displacement across all CA atoms -------------
        ca_after = [r.CA for r in conf.residues]
        displacements = [
            np.linalg.norm(after - before)
            for before, after in zip(ca_before, ca_after)
        ]
        rms = math.sqrt(sum(d * d for d in displacements) / len(displacements))

        self.assertLess(
            rms, self._RMSD_LIMIT,
            msg=(
                f"CA RMS displacement after {self._PHI_PERTURB}° phi "
                f"perturbation: {rms:.5f} Å (limit: {self._RMSD_LIMIT} Å). "
                f"Individual displacements: {[f'{d:.5f}' for d in displacements]}"
            ),
        )


# ─── 3. TestRebuildBoundaryConditions ────────────────────────────────

class TestRebuildBoundaryConditions(unittest.TestCase):
    """
    Edge-case behaviour of ``_rebuild_backbone`` at chain boundaries.
    """

    def test_single_residue_does_not_crash(self):
        """
        A single-residue chain must not raise when ``_rebuild_backbone``
        is called on it — the internal loop should be empty and return
        without error.
        -----------------------------------------------------------------
        For a 1-residue chain, max(1, start) ≥ 1 and min(end, 1) ≤ 1,
        so the range is always empty.
        """
        conf = Conformation(residues=[Residue(1, "ALA", phi=-60.0, psi=-45.0)])
        sampler = FragmentInsertionMC.__new__(FragmentInsertionMC)
        FragmentInsertionMC.__init__(sampler, lambda c: 0)

        # This should be a no-op, not an exception.
        try:
            FragmentInsertionMC._rebuild_backbone(sampler, conf, 0, 1)
        except Exception as exc:
            self.fail(
                f"_rebuild_backbone raised {type(exc).__name__} on a "
                f"single-residue chain: {exc}"
            )

    def test_two_residue_peptide_bond_length(self):
        """
        A 2-residue chain must respect the 1.33 Å C–N peptide bond.
        -----------------------------------------------------------------
        This is the minimal chain where a peptide bond exists, and the
        rebuild code must handle it without off-by-one errors.
        """
        conf, sampler = _build_conf("AA", phi=-60.0, psi=-45.0)
        residues = conf.residues

        dist = np.linalg.norm(residues[1].N - residues[0].C)
        self.assertAlmostEqual(
            dist, 1.33, delta=0.01,
            msg=(
                f"2-residue chain C(1)–N(2) bond: "
                f"expected 1.33 Å, got {dist:.4f} Å"
            ),
        )


# ─── 4. TestFragmentInsertionIntegrator ──────────────────────────────

class TestFragmentInsertionIntegrator(unittest.TestCase):
    """
    The fragment-insertion integrator mutates backbone dihedrals and
    requires a subsequent rebuild; both steps must leave covalent bond
    geometry intact.
    """

    def setUp(self):
        """Build a fresh 3-residue chain and sampler for each test."""
        self.conf, self.sampler = _build_conf("AAA", phi=-60.0, psi=-45.0)
        # Seed for reproducibility of the random fragment draw.
        random.seed(42)

    def _original_phi_psi(self):
        """Return lists of (phi, psi) before the fragment insertion."""
        return [(r.phi, r.psi) for r in self.conf.residues]

    def test_fragment_insertion_changes_phi_psi(self):
        """
        ``_insert_fragment`` on residue 1, length 3 must alter at least
        one backbone dihedral in the chain.
        -----------------------------------------------------------------
        Fragment insertion draws new phi/psi values from the Ramachandran
        library with Gaussian noise; with the seeded RNG these should
        differ from the initial uniform (-60°, -45°).
        """
        # --- Arrange: capture pre-insertion dihedrals -----------------
        pre = self._original_phi_psi()

        # --- Act: insert a 3-residue fragment starting at residue 1 ----
        self.sampler._insert_fragment(self.conf, 0, 3)

        # --- Assert: at least one phi or psi changed ------------------
        post = [(r.phi, r.psi) for r in self.conf.residues]
        any_change = any(
            abs(p[0] - q[0]) > 1e-9 or abs(p[1] - q[1]) > 1e-9
            for p, q in zip(pre, post)
        )
        self.assertTrue(
            any_change,
            msg=(
                "Fragment insertion did not modify any backbone dihedral. "
                "Either the RNG seed produced a degenerate sample or "
                "_insert_fragment failed to mutate the conformation."
            ),
        )

    def test_fragment_insertion_preserves_bond_geometry(self):
        """
        After fragment insertion followed by a full backbone rebuild,
        N–CA and CA–C bond lengths must still match ideal values.
        -----------------------------------------------------------------
        The integrator relies on ``_rebuild_backbone`` to reconstruct
        coordinates from the new dihedrals; this test confirms that the
        post-rebuild geometry remains chemically correct.
        """
        # --- Act: insert fragment and rebuild -------------------------
        self.sampler._insert_fragment(self.conf, 0, 3)
        for i in range(1, len(self.conf.residues)):
            self.sampler._rebuild_backbone(self.conf, i, i + 1)

        # --- Assert: bond lengths are preserved -----------------------
        # Rebuilt residues are indices 1 and 2 (0-indexed).
        for idx in [1, 2]:
            res = self.conf.residues[idx]

            n_ca = np.linalg.norm(res.CA - res.N)
            self.assertAlmostEqual(
                n_ca, 1.47, delta=0.01,
                msg=(
                    f"After fragment insertion + rebuild, residue "
                    f"{res.seq_index} N–CA = {n_ca:.4f} Å (expected 1.47)"
                ),
            )

            ca_c = np.linalg.norm(res.C - res.CA)
            self.assertAlmostEqual(
                ca_c, 1.51, delta=0.01,
                msg=(
                    f"After fragment insertion + rebuild, residue "
                    f"{res.seq_index} CA–C = {ca_c:.4f} Å (expected 1.51)"
                ),
            )


# ─── 5. TestVelocityVerletStub ───────────────────────────────────────

class TestVelocityVerletStub(unittest.TestCase):
    """
    Placeholder for velocity-Verlet integration (kinetics.verlet).
    Skipped until the dev/engine-dmitry branch merges.
    """

    def test_velocity_verlet_not_available(self):
        """
        Attempt to import ``kinetics.verlet``; skip the test with a
        descriptive message when the module is absent (expected until
        dev/engine-dmitry is merged).
        """
        try:
            import kinetics.verlet  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest(
                "kinetics.verlet is not available until the "
                "dev/engine-dmitry branch merges."
            )


# ─── 6. TestLangevinStub ─────────────────────────────────────────────

class TestLangevinStub(unittest.TestCase):
    """
    Placeholder for Langevin-dynamics integration (kinetics.langevin).
    Skipped until the module lands on the main branch.
    """

    def test_langevin_not_available(self):
        """
        Attempt to import ``kinetics.langevin``; skip when the module
        does not exist.
        """
        try:
            import kinetics.langevin  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest(
                "kinetics.langevin is not yet available — "
                "expected to land before the next release cycle."
            )


# ─── Entry Point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
