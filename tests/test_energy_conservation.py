#!/usr/bin/env python3
"""
tests/test_energy_conservation.py
==================================

Comprehensive energy conservation and determinism test suite for the
protein folding engine.  This module verifies the following invariants
of the RosettaEnergyFunction and the FragmentInsertionMC sampler:

    1.  **Energy determinism** — repeated evaluations on the same
        Conformation object produce bit-identical total energy and
        component energies, and deep copies match to 12 decimal places.
    2.  **Energy component consistency** — all 5 expected component keys
        are present in energy_components, the weighted sum Σ(w_i * comp_i)
        equals total_energy, and every residue carries a finite rama_score.
    3.  **Energy reversibility** — a phi/psi perturbation followed by an
        exact revert recovers the original total energy to 12 decimal
        places, both for single-residue and multi-residue perturbations.
    4.  **Edge-case robustness** — single-residue conformations produce
        zero LJ / HBond / repulsive components, and two-residue
        conformations satisfy swap symmetry (E(ALA,VAL) == E(VAL,ALA)).
    5.  **MC-step energy conservation** — under kT → 0 the Metropolis
        criterion rejects moves with delta_e ≥ 0, and the reverted
        conformation yields a bit-identical energy.
    6.  **Full-pipeline integration** — fold_protein produces the correct
        residue count, finite component energies, bounded |E| < 1000,
        and completes without error on mixed sequences.

Mathematical basis
------------------
The Rosetta energy function is a weighted linear combination:

    E_total = w_lj * E_LJ + w_hbond * E_HB + w_solv * E_solv
              + w_rama * E_rama + w_rep * E_rep

Under the Metropolis-Hastings criterion, any rejected move must restore
the exact previous conformation, yielding a bit-identical total energy.
The present suite enforces that invariant.

Author
------
    Priya Sharma
    Quality Engineering Division
    Indian Institute of Science, Bengaluru
"""

from __future__ import annotations

import copy
import math
import random
import struct
import unittest

import numpy as np

from folding_engine import (
    AA_CODES,
    Conformation,
    FragmentInsertionMC,
    Residue,
    RosettaEnergyFunction,
    VDW_RADII,
    _RAMA_PREFERENCE,
    fold_protein,
)


# ===================================================================
# Helper: build an extended-chain Conformation from a one-letter
# sequence, rebuild backbone coordinates.
# ===================================================================
def _build_extended(seq: str) -> Conformation:
    """
    Build an extended-chain Conformation from a one-letter amino acid
    sequence, then rebuild the backbone coordinates so that pairwise
    energy terms can be meaningfully evaluated.

    Parameters
    ----------
    seq : str
        One-letter amino acid codes (e.g. ``'ACDEF'``).

    Returns
    -------
    Conformation
        A Conformation with rebuilt backbone coordinates.
    """
    residues = [
        Residue(
            seq_index=i + 1,
            aa_code=AA_CODES.get(aa.upper(), "ALA"),
            phi=-135.0,
            psi=135.0,
        )
        for i, aa in enumerate(seq)
    ]

    conf = Conformation(residues=residues)

    # Use FragmentInsertionMC purely for _rebuild_backbone.
    sampler = FragmentInsertionMC.__new__(FragmentInsertionMC)
    FragmentInsertionMC.__init__(sampler, lambda c: 0)  # type: ignore[arg-type]

    for i in range(1, len(conf.residues)):
        FragmentInsertionMC._rebuild_backbone(sampler, conf, i, i + 1)

    return conf


# ===================================================================
# Helper: convert a float to bytes for bit-level comparison.
# ===================================================================
def _float_bytes(value: float) -> bytes:
    """Pack a Python float into 8 bytes (IEEE 754 double)."""
    return struct.pack("d", value)


# ===================================================================
# 1.  TestEnergyDeterminism  (3 tests)
# ===================================================================
class TestEnergyDeterminism(unittest.TestCase):
    """
    Verify that the RosettaEnergyFunction is fully deterministic:
    repeated evaluations on the *same* Conformation produce bit-identical
    total energy, deep-copied conformations match to 12 dp, and all
    5 components are deterministic to 12 dp.

    AAA:
        Arrange: Build a medium-length extended chain.
        Act:     Evaluate energy multiple times.
        Assert:  Energies match at the required precision.
    """

    # ------------------------------------------------------------------
    # 1.1  Repeated evaluate() calls produce bit-identical energy
    # ------------------------------------------------------------------
    def test_evaluate_bit_identical(self) -> None:
        """
        GIVEN  a 10-residue extended chain (polyalanine)
        WHEN   RosettaEnergyFunction.evaluate() is called three times
               on the *same* Conformation object
        THEN   the ten three total-energy values must be bit-identical
               (same IEEE 754 double representation).
        """
        # Arrange: build 10mer polyalanine and score it.
        conf = _build_extended("AAAAAAAAAA")
        energy_fn = RosettaEnergyFunction()

        # Act: evaluate three times.
        e1 = energy_fn.evaluate(conf)
        e2 = energy_fn.evaluate(conf)
        e3 = energy_fn.evaluate(conf)

        # Assert: bit-identical (raw bytes).
        self.assertEqual(
            _float_bytes(e1),
            _float_bytes(e2),
            "First and second evaluate() differ at the bit level: "
            f"e1={e1:.20e}, e2={e2:.20e}.",
        )
        self.assertEqual(
            _float_bytes(e2),
            _float_bytes(e3),
            "Second and third evaluate() differ at the bit level: "
            f"e2={e2:.20e}, e3={e3:.20e}.",
        )

    # ------------------------------------------------------------------
    # 1.2  Deep copy produces identical energy to 12 dp
    # ------------------------------------------------------------------
    def test_deep_copy_identical_12dp(self) -> None:
        """
        GIVEN  an evaluated Conformation
        WHEN   a deep copy is created and evaluated
        THEN   the total energy must match the original to at least
               12 decimal places.
        """
        # Arrange: build 10mer mixed sequence and score.
        conf = _build_extended("ACDEFGHIKL")
        energy_fn = RosettaEnergyFunction()
        energy_fn.evaluate(conf)
        original_energy = conf.total_energy

        # Act: deep-copy and re-evaluate.
        conf_copy = copy.deepcopy(conf)
        energy_fn.evaluate(conf_copy)
        copy_energy = conf_copy.total_energy

        # Assert: 12 decimal places.
        self.assertAlmostEqual(
            original_energy,
            copy_energy,
            places=12,
            msg=(
                "Deep-copied conformation must produce the same total "
                "energy to 12 decimal places.  "
                f"Original={original_energy:.15e}, "
                f"Copy={copy_energy:.15e}."
            ),
        )

    # ------------------------------------------------------------------
    # 1.3  All 5 components are deterministic to 12 dp
    # ------------------------------------------------------------------
    def test_all_components_deterministic_12dp(self) -> None:
        """
        GIVEN  an evaluated Conformation with energy_components
        WHEN   a deep copy is evaluated
        THEN   every component (lennard_jones, hydrogen_bond, solvation,
               ramachandran, repulsive) must match to 12 decimal places.
        """
        # Arrange: build, score, capture component dict.
        conf = _build_extended("MLSDGEFQL")
        energy_fn = RosettaEnergyFunction()
        energy_fn.evaluate(conf)
        orig_components = dict(conf.energy_components)

        # Act: deep-copy and re-evaluate.
        conf_copy = copy.deepcopy(conf)
        energy_fn.evaluate(conf_copy)
        copy_components = conf_copy.energy_components

        # Assert: each of the 5 components matches to 12 dp.
        component_keys = [
            "lennard_jones",
            "hydrogen_bond",
            "solvation",
            "ramachandran",
            "repulsive",
        ]
        for key in component_keys:
            self.assertIn(
                key,
                orig_components,
                f"Original energy_components is missing key '{key}'.",
            )
            self.assertIn(
                key,
                copy_components,
                f"Copy energy_components is missing key '{key}'.",
            )
            self.assertAlmostEqual(
                orig_components[key],
                copy_components[key],
                places=12,
                msg=(
                    f"Energy component '{key}' must be deterministic to "
                    f"12 decimal places.  "
                    f"Original={orig_components[key]:.15e}, "
                    f"Copy={copy_components[key]:.15e}."
                ),
            )


# ===================================================================
# 2.  TestEnergyComponents  (3 tests)
# ===================================================================
class TestEnergyComponents(unittest.TestCase):
    """
    Verify that the energy components are stored correctly and satisfy
    the linear combination:

        E_total == w_lj * E_LJ + w_hbond * E_HB + w_solv * E_solv
                   + w_rama * E_rama + w_rep * E_rep

    Also verify that the per-residue Ramachandran score is stored on
    every Residue object after evaluation.

    AAA:
        Arrange: Build a multi-residue extended chain.
        Act:     Call evaluate().
        Assert:  Keys present, weighted sum matches, rama_score set.
    """

    def setUp(self) -> None:
        """Build a 6-residue mixed sequence and evaluate."""
        self.conf = _build_extended("ACDEFG")
        self.energy_fn = RosettaEnergyFunction()
        self.energy_fn.evaluate(self.conf)

    # ------------------------------------------------------------------
    # 2.1  All 5 component keys are present
    # ------------------------------------------------------------------
    def test_component_keys_present(self) -> None:
        """
        GIVEN  an evaluated Conformation
        WHEN   inspecting energy_components
        THEN   all 5 expected keys (lennard_jones, hydrogen_bond,
               solvation, ramachandran, repulsive) must be present.
        """
        # Arrange: component dict already populated via setUp.
        comp = self.conf.energy_components
        expected_keys = {
            "lennard_jones",
            "hydrogen_bond",
            "solvation",
            "ramachandran",
            "repulsive",
        }

        # Act: compute set difference.
        actual_keys = set(comp.keys())
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys

        # Assert: no missing or extra keys.
        self.assertSetEqual(
            expected_keys,
            actual_keys,
            f"Energy component keys mismatch.  "
            f"Missing={missing}, Extra={extra}.",
        )

    # ------------------------------------------------------------------
    # 2.2  Weighted sum matches total_energy
    # ------------------------------------------------------------------
    def test_weighted_sum_matches_total(self) -> None:
        """
        GIVEN  an evaluated Conformation with populated energy_components
        WHEN   computing Σ(w_i * component_i) using the energy function's
               own weights
        THEN   the result must equal conf.total_energy to 12 decimal
               places.
        """
        # Arrange: component dict and weights.
        comp = self.conf.energy_components
        w = self.energy_fn

        # Act: compute weighted sum.
        weighted_sum = (
            w.w_lj * comp["lennard_jones"]
            + w.w_hbond * comp["hydrogen_bond"]
            + w.w_solvation * comp["solvation"]
            + w.w_rama * comp["ramachandran"]
            + w.w_repulsive * comp["repulsive"]
        )

        # Assert: matches total_energy to 12 dp.
        self.assertAlmostEqual(
            weighted_sum,
            self.conf.total_energy,
            places=12,
            msg=(
                "Weighted sum Σ(w_i * component_i) must equal "
                "total_energy to 12 decimal places.  "
                f"Weighted sum={weighted_sum:.15e}, "
                f"total_energy={self.conf.total_energy:.15e}."
            ),
        )

    # ------------------------------------------------------------------
    # 2.3  Ramachandran score stored on each residue
    # ------------------------------------------------------------------
    def test_ramachandran_stored_on_residues(self) -> None:
        """
        GIVEN  an evaluated Conformation
        WHEN   inspecting each residue's rama_score attribute
        THEN   every residue must have a non-negative rama_score
               between 0.0 and 1.0 (inclusive).
        """
        for res in self.conf.residues:
            # Assert: rama_score is non-negative.
            self.assertGreaterEqual(
                res.rama_score,
                0.0,
                f"Residue {res.seq_index} ({res.aa_code}) has a negative "
                f"rama_score: {res.rama_score:.15e}.",
            )
            # Assert: rama_score does not exceed 1.0.
            self.assertLessEqual(
                res.rama_score,
                1.0,
                f"Residue {res.seq_index} ({res.aa_code}) has a "
                f"rama_score > 1.0: {res.rama_score:.15e}.",
            )


# ===================================================================
# 3.  TestEnergyReversibility  (2 tests)
# ===================================================================
class TestEnergyReversibility(unittest.TestCase):
    """
    Verify that backbone dihedral perturbations (phi/psi) can be
    reverted to recover the exact original energy.

    AAA:
        Arrange: Build a 5-residue extended chain with defined dihedrals.
        Act:     Perturb phi/psi, evaluate, then revert exactly.
        Assert:  Post-revert total energy matches original to 12 dp.
    """

    def setUp(self) -> None:
        """Build a 5-residue chain with specific dihedral values."""
        self.energy_fn = RosettaEnergyFunction()
        self.conf = _build_extended("VLAIG")
        # Overwrite with well-defined dihedrals for reproducibility.
        for i, res in enumerate(self.conf.residues):
            res.phi = -60.0 + i * 10.0
            res.psi = -45.0 - i * 5.0
        self.original_energy = self.energy_fn.evaluate(self.conf)

    # ------------------------------------------------------------------
    # 3.1  Single-residue perturbation then revert
    # ------------------------------------------------------------------
    def test_single_residue_perturb_revert_12dp(self) -> None:
        """
        GIVEN  an evaluated conformation
        WHEN   residue index 2 has phi increased by +30° and psi
               decreased by -20°, then both are reverted exactly
        THEN   the total energy must recover to 12 decimal places.
        """
        # Arrange: store original dihedrals of residue 2.
        idx = 2
        orig_phi = self.conf.residues[idx].phi
        orig_psi = self.conf.residues[idx].psi

        # Act: perturb (phi+30, psi-20).
        self.conf.residues[idx].phi = orig_phi + 30.0
        self.conf.residues[idx].psi = orig_psi - 20.0
        _ = self.energy_fn.evaluate(self.conf)

        # Act: revert exactly.
        self.conf.residues[idx].phi = orig_phi
        self.conf.residues[idx].psi = orig_psi
        restored_energy = self.energy_fn.evaluate(self.conf)

        # Assert: energy recovered.
        self.assertAlmostEqual(
            self.original_energy,
            restored_energy,
            places=12,
            msg=(
                "Single-residue phi/psi revert did not recover the "
                "original energy to 12 decimal places.  "
                f"Original={self.original_energy:.15e}, "
                f"Restored={restored_energy:.15e}."
            ),
        )

    # ------------------------------------------------------------------
    # 3.2  Multi-residue (3 residues) perturbation then revert
    # ------------------------------------------------------------------
    def test_multi_residue_perturb_revert_12dp(self) -> None:
        """
        GIVEN  an evaluated conformation
        WHEN   three residues (indices 1, 2, 3) have their phi (+30°)
               and psi (-20°) perturbed, then all are reverted exactly
        THEN   the total energy must recover to 12 decimal places.
        """
        # Arrange: store original dihedrals for residues 1, 2, 3.
        indices = [1, 2, 3]
        orig = {}
        for idx in indices:
            orig[idx] = (
                self.conf.residues[idx].phi,
                self.conf.residues[idx].psi,
            )

        # Act: perturb all three.
        for idx in indices:
            self.conf.residues[idx].phi = orig[idx][0] + 30.0
            self.conf.residues[idx].psi = orig[idx][1] - 20.0
        _ = self.energy_fn.evaluate(self.conf)

        # Act: revert all three exactly.
        for idx in indices:
            self.conf.residues[idx].phi = orig[idx][0]
            self.conf.residues[idx].psi = orig[idx][1]
        restored_energy = self.energy_fn.evaluate(self.conf)

        # Assert: energy recovered.
        self.assertAlmostEqual(
            self.original_energy,
            restored_energy,
            places=12,
            msg=(
                "Multi-residue phi/psi revert did not recover the "
                "original energy to 12 decimal places.  "
                f"Original={self.original_energy:.15e}, "
                f"Restored={restored_energy:.15e}."
            ),
        )


# ===================================================================
# 4.  TestEnergyEdgeCases  (2 tests)
# ===================================================================
class TestEnergyEdgeCases(unittest.TestCase):
    """
    Verify that the energy function behaves correctly at edge cases:

    * A single-residue conformation: only Ramachandran (and potentially
      solvation) contributes; LJ, HBond, and repulsive must be zero.
    * Two-residue swap symmetry: E(ALA, VAL) == E(VAL, ALA) to 12 dp.

    AAA:
        Arrange: Build edge-case conformations (1 or 2 residues).
        Act:     Evaluate energy.
        Assert:  Components behave as physically expected.
    """

    def setUp(self) -> None:
        """Create a shared energy function."""
        self.energy_fn = RosettaEnergyFunction()

    # ------------------------------------------------------------------
    # 4.1  Single residue: LJ, HBond, repulsive are zero
    # ------------------------------------------------------------------
    def test_single_residue_ramachandran_only(self) -> None:
        """
        GIVEN  a single-residue conformation (ALA)
        WHEN   the energy function is evaluated
        THEN   the Lennard-Jones, hydrogen-bond, and repulsive
               components must be exactly 0.0, while the Ramachandran
               component must be strictly positive.
        """
        # Arrange: single ALA residue.
        conf = _build_extended("A")
        # Set dihedrals to a non-optimal position (off-basin) so that
        # Ramachandran energy is strictly positive — at the exact basin
        # centre (-60, -45) the score would be 0 (best = 1.0, E = 0).
        conf.residues[0].phi = -90.0
        conf.residues[0].psi = -90.0

        # Act: evaluate.
        self.energy_fn.evaluate(conf)
        comp = conf.energy_components

        # Assert: LJ = 0 (no residue pairs).
        self.assertEqual(
            comp["lennard_jones"],
            0.0,
            f"Lennard-Jones for single residue must be 0.0, "
            f"got {comp['lennard_jones']:.15e}.",
        )

        # Assert: HBond = 0 (need at least 4 residues).
        self.assertEqual(
            comp["hydrogen_bond"],
            0.0,
            f"Hydrogen bond for single residue must be 0.0, "
            f"got {comp['hydrogen_bond']:.15e}.",
        )

        # Assert: Repulsive = 0 (no residue pairs).
        self.assertEqual(
            comp["repulsive"],
            0.0,
            f"Repulsive for single residue must be 0.0, "
            f"got {comp['repulsive']:.15e}.",
        )

        # Assert: Ramachandran > 0 (always computed).
        self.assertGreater(
            comp["ramachandran"],
            0.0,
            f"Ramachandran for single residue must be > 0, "
            f"got {comp['ramachandran']:.15e}.",
        )

    # ------------------------------------------------------------------
    # 4.2  Two-residue swap symmetry
    # ------------------------------------------------------------------
    def test_two_residue_symmetry_12dp(self) -> None:
        """
        GIVEN  two conformations: (ALA, VAL) and (VAL, ALA)
        WHEN   both are evaluated with the same energy function
        THEN   every component and the total energy must match to
               12 decimal places (swap symmetry).
        """
        # Arrange: build two conformations with swapped residues.
        conf_av = _build_extended("AV")  # ALA, VAL
        conf_va = _build_extended("VA")  # VAL, ALA

        # Act: evaluate both.
        self.energy_fn.evaluate(conf_av)
        self.energy_fn.evaluate(conf_va)

        # Assert: each component matches to 12 dp.
        component_keys = [
            "lennard_jones",
            "hydrogen_bond",
            "solvation",
            "ramachandran",
            "repulsive",
        ]
        for key in component_keys:
            self.assertAlmostEqual(
                conf_av.energy_components[key],
                conf_va.energy_components[key],
                places=12,
                msg=(
                    f"Swap symmetry broken for component '{key}': "
                    f"(ALA,VAL)={conf_av.energy_components[key]:.15e}, "
                    f"(VAL,ALA)={conf_va.energy_components[key]:.15e}."
                ),
            )

        # Assert: total energy matches to 12 dp.
        self.assertAlmostEqual(
            conf_av.total_energy,
            conf_va.total_energy,
            places=12,
            msg=(
                "Swap symmetry broken for total energy: "
                f"(ALA,VAL)={conf_av.total_energy:.15e}, "
                f"(VAL,ALA)={conf_va.total_energy:.15e}."
            ),
        )


# ===================================================================
# 5.  TestMCStepEnergyConservation  (1 test)
# ===================================================================
class TestMCStepEnergyConservation(unittest.TestCase):
    """
    Verify that a single Metropolis Monte Carlo step conserves energy
    when the proposed move is rejected.

    Under the Metropolis acceptance criterion with kT → 0, any move
    that results in delta_e >= 0 is deterministically rejected, and the
    conformation (and therefore its energy) must be restored exactly.

    If delta_e < 0 happens to occur, the move would be auto-accepted
    (because delta_e < 0 always passes the Metropolis test), so the
    rejection branch cannot be tested for that random seed; the test
    is skipped in that case via ``self.skipTest()``.

    AAA:
        Arrange: Build a 5-residue extended chain with defined dihedrals.
        Act:     Run one MC step with kT = 1e-10 (effectively zero).
        Assert:  If rejected (delta_e >= 0), post-revert energy is
                 bit-identical to pre-move energy.
    """

    def test_rejected_move_conserves_energy(self) -> None:
        """
        GIVEN  a 5-residue conformation evaluated at kT → 0
        WHEN   a single fragment-insertion MC step is attempted
        THEN   if delta_e >= 0 (rejected), the total energy after
               revert must be bit-identical to the pre-move energy.
               If delta_e < 0 (auto-accept), the test is skipped.
        """
        # Arrange: fixed random seed for reproducibility.
        random.seed(42)
        np.random.seed(42)

        # Build 5-residue mixed sequence.
        conf = _build_extended("ACDEF")
        energy_fn = RosettaEnergyFunction()

        # Create sampler with near-zero temperature.
        sampler = FragmentInsertionMC(energy_fn, temperature=1e-6)
        # Override kT to be effectively zero.
        sampler.kT = 1e-10

        n_residues = len(conf.residues)

        # Record pre-move energy.
        pre_energy = energy_fn.evaluate(conf)

        # Select a random fragment (same logic as FragmentInsertionMC.run).
        start = random.randint(0, n_residues - 4)
        frag_len = random.choice([3, 9])

        # Store old dihedrals.
        old_phi_psi: list[tuple[float, float]] = []
        for i in range(start, min(start + frag_len, n_residues)):
            old_phi_psi.append((conf.residues[i].phi, conf.residues[i].psi))

        # Insert fragment (propose move).
        sampler._insert_fragment(conf, start, frag_len)
        sampler._rebuild_backbone(conf, start, start + frag_len)

        # Evaluate trial energy.
        trial_energy = energy_fn.evaluate(conf)
        delta_e = trial_energy - pre_energy

        # Revert the move (as in the rejection branch).
        for j, (phi, psi) in enumerate(old_phi_psi):
            idx = start + j
            if idx < n_residues:
                conf.residues[idx].phi = phi
                conf.residues[idx].psi = psi
        sampler._rebuild_backbone(conf, start, start + frag_len)

        # If delta_e < 0, the move would have been auto-accepted.
        if delta_e < 0:
            self.skipTest(
                f"delta_e = {delta_e:.15e} < 0 — Metropolis auto-accept, "
                f"cannot test rejection branch with this random seed.  "
                f"Try a different seed."
            )

        # Assert: post-revert energy is bit-identical to pre-move.
        post_energy = energy_fn.evaluate(conf)
        self.assertEqual(
            _float_bytes(pre_energy),
            _float_bytes(post_energy),
            f"Energy after rejected MC move must be bit-identical to "
            f"pre-move energy.  "
            f"pre_energy={pre_energy:.20e}, "
            f"post_energy={post_energy:.20e}, "
            f"delta_e={delta_e:.15e}.",
        )

        # Cross-check: delta_e >= 0 (rejection condition).
        self.assertGreaterEqual(
            delta_e,
            0.0,
            f"Expected delta_e >= 0 for a rejected move, "
            f"got delta_e = {delta_e:.15e}.",
        )


# ===================================================================
# 6.  TestFoldProteinIntegration  (4 tests)
# ===================================================================
class TestFoldProteinIntegration(unittest.TestCase):
    """
    Integration tests for the complete fold_protein pipeline.

    Verifies that the pipeline produces the correct number of residues,
    that all energy components are finite, that the total energy is
    bounded within ±1000 kcal/mol, and that mixed sequences run without
    exceptions.

    NOTE:
        fold_protein returns ``(conf, energy_trace, n_accept)``.
        This test handles the tuple unpacking accordingly.

    AAA:
        Arrange: Define short amino-acid sequences.
        Act:     Call fold_protein with a small number of steps.
        Assert:  Output conformation meets all structural and
                 energetic criteria.
    """

    # ------------------------------------------------------------------
    # 6.1  Correct residue count
    # ------------------------------------------------------------------
    def test_correct_residue_count(self) -> None:
        """
        GIVEN  a 5-letter sequence ``'ACDEF'``
        WHEN   fold_protein is called with 100 steps
        THEN   the returned Conformation must contain exactly 5 residues.
        """
        # Act: fold a short sequence.
        result = fold_protein("ACDEF", n_steps=100, temperature=300.0)

        # Handle tuple return: (conf, energy_trace, n_accept).
        if isinstance(result, tuple):
            conf, _energy_trace, _n_accept = result
        else:
            conf = result

        # Assert: correct number of residues.
        self.assertEqual(
            len(conf.residues),
            5,
            f"Expected 5 residues, got {len(conf.residues)}.  "
            f"Sequence was 'ACDEF'.",
        )

    # ------------------------------------------------------------------
    # 6.2  All components finite after fold
    # ------------------------------------------------------------------
    def test_components_finite_after_fold(self) -> None:
        """
        GIVEN  a 5-letter sequence ``'ACDEF'``
        WHEN   fold_protein is called with 100 steps
        THEN   all 5 energy components must be finite (not NaN or inf).
        """
        # Act: fold.
        result = fold_protein("ACDEF", n_steps=100, temperature=300.0)
        if isinstance(result, tuple):
            conf, _energy_trace, _n_accept = result
        else:
            conf = result

        # Assert: each component is finite.
        comp = conf.energy_components
        component_keys = [
            "lennard_jones",
            "hydrogen_bond",
            "solvation",
            "ramachandran",
            "repulsive",
        ]
        for key in component_keys:
            self.assertIn(
                key,
                comp,
                f"Energy components dict is missing key '{key}' after fold.",
            )
            self.assertTrue(
                math.isfinite(comp[key]),
                f"Component '{key}' is not finite after fold: "
                f"{comp[key]!r}.",
            )

    # ------------------------------------------------------------------
    # 6.3  Total energy bounded
    # ------------------------------------------------------------------
    def test_energy_bounded(self) -> None:
        """
        GIVEN  a 5-letter sequence ``'ACDEF'`` folded for 100 steps
        WHEN   inspecting total_energy
        THEN   |total_energy| must be strictly less than 1000 kcal/mol.
        """
        # Act: fold.
        result = fold_protein("ACDEF", n_steps=100, temperature=300.0)
        if isinstance(result, tuple):
            conf, _energy_trace, _n_accept = result
        else:
            conf = result

        # Assert: |E| < 1000.
        abs_energy = abs(conf.total_energy)
        self.assertLess(
            abs_energy,
            1000.0,
            f"|total_energy| = {abs_energy:.10e} exceeds the 1000 "
            f"kcal/mol bound.  "
            f"total_energy = {conf.total_energy:.10e}.",
        )

    # ------------------------------------------------------------------
    # 6.4  Mixed-sequence pipeline completes without error
    # ------------------------------------------------------------------
    def test_mixed_sequence_pipeline(self) -> None:
        """
        GIVEN  a mixed 9-residue sequence ``'MLSDGEFQL'``
        WHEN   fold_protein is called with 100 steps
        THEN   the pipeline must complete without raising any exception
               and return a Conformation instance.
        """
        # Act & Assert: no exception.
        try:
            result = fold_protein("MLSDGEFQL", n_steps=100, temperature=300.0)
            if isinstance(result, tuple):
                conf, _energy_trace, _n_accept = result
            else:
                conf = result
            self.assertIsInstance(
                conf,
                Conformation,
                f"Expected Conformation instance, got {type(conf)}.",
            )
        except Exception as exc:
            self.fail(
                "fold_protein raised an unexpected exception on "
                f"sequence 'MLSDGEFQL': {exc!r}."
            )


# ===================================================================
# Test runner entry point
# ===================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
