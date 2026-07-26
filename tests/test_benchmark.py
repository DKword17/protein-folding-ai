#!/usr/bin/env python3
"""
tests/test_benchmark.py
=======================

Benchmark scoring precision test suite for the protein folding engine.

This module verifies:
    1. Determinism of the RosettaEnergyFunction — repeated evaluations
       on the same conformation produce bit-identical results.
    2. Numerical precision — small perturbations produce small energy
       changes, and revert operations recover the original score.
    3. Conformation metadata integrity — RMSD and pLDDT set/get, deep
       copy preservation, and boundary conditions.
    4. Weighted-score consistency — the sum of weighted component terms
       matches the reported total energy, and all weights are in (0, 5).
    5. Sequence-length scaling — longer chains produce larger energies.
    6. Ramachandran score precision — every standard AA yields a finite
       backbone preference score.
    7. Full-pipeline benchmark — fold_protein produces finite energies
       and the correct residue count.
    8. Stub tests for external benchmark tools (GDT_TS, lDDT, TM-score,
       SPICKER, Rg-SASA) — gracefully skipped until implemented.
    9. Edge cases — single-residue conformations and overlapping
       residues both produce finite (non-exploding) energies.

All tests follow the AAA (Arrange-Act-Assert) pattern with detailed
comments so that every team member can readily understand the purpose,
procedure and expected outcome.

Author
------
    Priya Sharma
    Quality Engineering Division
    Indian Institute of Science, Bengaluru
"""

from __future__ import annotations

import copy
import math
import struct
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Imports from the production folding engine
# ---------------------------------------------------------------------------
from folding_engine import (
    AA_CODES,
    Conformation,
    FragmentInsertionMC,
    Residue,
    RosettaEnergyFunction,
    fold_protein,
)


# ---------------------------------------------------------------------------
# Helper: build an extended-chain Conformation from a one-letter sequence
# ---------------------------------------------------------------------------
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
    # Step 1: create one Residue per character; default phi = -135, psi = +135
    #         gives an extended (beta-strand-like) backbone.
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

    # Step 2: create a minimal MC sampler purely to call _rebuild_backbone.
    #         The energy function is a no-op (lambda returning 0) because
    #         we only need the geometry builder.
    sampler = FragmentInsertionMC.__new__(FragmentInsertionMC)
    FragmentInsertionMC.__init__(sampler, lambda c: 0)  # type: ignore[arg-type]

    # Step 3: rebuild backbone coordinates from residue i-1 to i.
    for i in range(1, len(conf.residues)):
        FragmentInsertionMC._rebuild_backbone(sampler, conf, i, i + 1)

    return conf


# ===================================================================
# 1.  TestScoreDeterminism  (4 tests)
# ===================================================================
class TestScoreDeterminism(unittest.TestCase):
    """
    Verify that the RosettaEnergyFunction is fully deterministic:
    repeated evaluations of the *same* conformation produce bit-identical
    total energies and component energies.
    """

    def test_repeated_eval_bit_identical(self) -> None:
        """
        GIVEN  a medium-length extended chain (10mer of alanines)
        WHEN   the RosettaEnergyFunction.evaluate() is called twice
               in succession on the *same* Conformation object
        THEN   the two returned total_energy values must be bit-identical.
        """
        # Arrange: build a 10-residue polyalanine and score it.
        conf = _build_extended("AAAAAAAAAA")
        energy_fn = RosettaEnergyFunction()

        # Act: evaluate twice, capturing the raw float bytes.
        e1 = energy_fn.evaluate(conf)
        e2 = energy_fn.evaluate(conf)

        # Assert: bit-identical (float representation).
        self.assertEqual(
            e1.tobytes() if isinstance(e1, np.generic) else struct.pack("d", e1),
            e2.tobytes() if isinstance(e2, np.generic) else struct.pack("d", e2),
            "Repeated evaluate() calls must be bit-identical. "
            f"Got e1={e1:.20e}, e2={e2:.20e}.",
        )

    def test_deep_copy_identical_12dp(self) -> None:
        """
        GIVEN  an evaluated Conformation
        WHEN   a deep copy is created and evaluated
        THEN   the total energies must match to at least 12 decimal places.
        """
        # Arrange: build and score a 10mer.
        conf = _build_extended("ACDEFGHIKL")
        energy_fn = RosettaEnergyFunction()
        energy_fn.evaluate(conf)
        original_energy = conf.total_energy

        # Act: deep-copy and re-evaluate.
        conf_copy = copy.deepcopy(conf)
        energy_fn.evaluate(conf_copy)
        copy_energy = conf_copy.total_energy

        # Assert: match to 12 decimal places.
        self.assertAlmostEqual(
            original_energy,
            copy_energy,
            places=12,
            msg=(
                "Deep-copied conformation must produce the same total energy "
                "to 12 decimal places. "
                f"Original={original_energy:.15e}, Copy={copy_energy:.15e}."
            ),
        )

    def test_components_deterministic_12dp(self) -> None:
        """
        GIVEN  an evaluated Conformation with non-empty energy_components
        WHEN   a deep copy is evaluated
        THEN   every component must match to at least 12 decimal places.
        """
        # Arrange: build, score, and capture component dict.
        conf = _build_extended("MLSDGEFQL")
        energy_fn = RosettaEnergyFunction()
        energy_fn.evaluate(conf)
        orig_components = dict(conf.energy_components)

        # Act: deep-copy and re-evaluate.
        conf_copy = copy.deepcopy(conf)
        energy_fn.evaluate(conf_copy)
        copy_components = conf_copy.energy_components

        # Assert: each component matches to 12 dp.
        component_keys = ["lennard_jones", "hydrogen_bond", "solvation",
                          "ramachandran", "repulsive"]
        for key in component_keys:
            self.assertIn(key, orig_components,
                          f"Original components missing key '{key}'.")
            self.assertIn(key, copy_components,
                          f"Copy components missing key '{key}'.")
            self.assertAlmostEqual(
                orig_components[key],
                copy_components[key],
                places=12,
                msg=(
                    f"Energy component '{key}' must be deterministic to "
                    f"12 decimal places. "
                    f"Original={orig_components[key]:.15e}, "
                    f"Copy={copy_components[key]:.15e}."
                ),
            )

    def test_all_20_AAs_finite(self) -> None:
        """
        GIVEN  one extended chain for each of the 20 standard amino acids
        WHEN   RosettaEnergyFunction.evaluate() is called
        THEN   the total energy and all five components must be finite
               (not NaN, not inf, not -inf).
        """
        # Arrange: the full set of 20 standard one-letter codes.
        one_letter_codes = list("ACDEFGHIKLMNPQRSTVWY")
        energy_fn = RosettaEnergyFunction()

        for aa in one_letter_codes:
            with self.subTest(aa=aa):
                # Build a 5mer of the same amino acid.
                conf = _build_extended(aa * 5)

                # Act: evaluate.
                energy_fn.evaluate(conf)

                # Assert: total energy is finite.
                self.assertTrue(
                    math.isfinite(conf.total_energy),
                    f"Total energy must be finite for {aa}-5mer. "
                    f"Got {conf.total_energy}.",
                )

                # Assert: all components are finite.
                for comp_name, comp_val in conf.energy_components.items():
                    self.assertTrue(
                        math.isfinite(comp_val),
                        f"Component '{comp_name}' must be finite for "
                        f"{aa}-5mer. Got {comp_val}.",
                    )


# ===================================================================
# 2.  TestScorePrecision  (3 tests)
# ===================================================================
class TestScorePrecision(unittest.TestCase):
    """
    Verify that the energy function responds proportionally to small
    perturbations and that revert operations exactly recover the original
    score.
    """

    def test_phi_pert_01_deg_deltaE_lt_001(self) -> None:
        """
        GIVEN  an evaluated extended chain
        WHEN   the phi angle of a single residue is perturbed by +0.1°
        THEN   the absolute change in total energy must be < 0.01 kcal/mol.
        """
        # Arrange: build a 10mer mixed sequence and score baseline.
        conf = _build_extended("ACDEFGHIKL")
        energy_fn = RosettaEnergyFunction()
        energy_fn.evaluate(conf)
        baseline = conf.total_energy

        # Act: perturb phi at residue index 4 (5th residue) by +0.1°.
        conf.residues[4].phi += 0.1
        energy_fn.evaluate(conf)
        perturbed = conf.total_energy

        # Assert: |ΔE| < 0.01.
        delta_e = abs(perturbed - baseline)
        self.assertLess(
            delta_e,
            0.01,
            f"A 0.1° phi perturbation must change total energy by < 0.01 "
            f"kcal/mol. Baseline={baseline:.10e}, Perturbed={perturbed:.10e}, "
            f"ΔE={delta_e:.10e}.",
        )

    def test_pert_then_revert_recovers_12dp(self) -> None:
        """
        GIVEN  an evaluated extended chain
        WHEN   phi is perturbed and then reverted to its original value
        THEN   the total energy must recover to 12 decimal places.
        """
        # Arrange: build, score, and store original phi.
        conf = _build_extended("MLSDGEFQL")
        energy_fn = RosettaEnergyFunction()
        energy_fn.evaluate(conf)
        baseline = conf.total_energy
        original_phi = conf.residues[3].phi

        # Act: perturb, evaluate, revert, evaluate.
        conf.residues[3].phi = original_phi + 5.0  # 5° perturbation
        energy_fn.evaluate(conf)

        conf.residues[3].phi = original_phi  # revert exactly
        energy_fn.evaluate(conf)
        recovered = conf.total_energy

        # Assert: recovered energy matches baseline to 12 dp.
        self.assertAlmostEqual(
            baseline,
            recovered,
            places=12,
            msg=(
                "Reverting a phi perturbation must recover the original "
                "total energy to 12 decimal places. "
                f"Baseline={baseline:.15e}, Recovered={recovered:.15e}."
            ),
        )

    def test_180_flip_finite(self) -> None:
        """
        GIVEN  an evaluated extended chain
        WHEN   phi of one residue is flipped by 180°
        THEN   the total energy must remain finite (not NaN, inf, or -inf).
        """
        # Arrange: build and score baseline.
        conf = _build_extended("ACDEFGHIKL")
        energy_fn = RosettaEnergyFunction()
        energy_fn.evaluate(conf)

        # Act: flip phi by 180° at residue index 2.
        conf.residues[2].phi += 180.0
        energy_fn.evaluate(conf)
        flipped_energy = conf.total_energy

        # Assert: energy is finite.
        self.assertTrue(
            math.isfinite(flipped_energy),
            f"A 180° phi flip must still yield a finite total energy. "
            f"Got {flipped_energy}.",
        )


# ===================================================================
# 3.  TestConformationRMSD  (3 tests)
# ===================================================================
class TestConformationRMSD(unittest.TestCase):
    """
    Verify that the RMSD attribute on Conformation can be set, retrieved,
    deep-copied, and that its default value is None.
    """

    def test_set_retrieve_2345(self) -> None:
        """
        GIVEN  a fresh Conformation
        WHEN   rmsd is set to 2.345
        THEN   reading rmsd back must return 2.345.
        """
        # Arrange: build minimal conformation.
        conf = _build_extended("ALA")

        # Act: set RMSD.
        conf.rmsd = 2.345

        # Assert: retrieve and compare.
        self.assertEqual(
            conf.rmsd,
            2.345,
            f"RMSD set to 2.345 must be retrievable as 2.345. "
            f"Got {conf.rmsd}.",
        )

    def test_deep_copy_preserved(self) -> None:
        """
        GIVEN  a Conformation with RMSD set to a known value
        WHEN   a deep copy is made
        THEN   the copy's RMSD must equal the original's RMSD.
        """
        # Arrange: build and set RMSD.
        conf = _build_extended("ALA")
        conf.rmsd = 1.234

        # Act: deep copy.
        conf_copy = copy.deepcopy(conf)

        # Assert: RMSD preserved.
        self.assertEqual(
            conf.rmsd,
            conf_copy.rmsd,
            f"Deep copy must preserve RMSD. "
            f"Original={conf.rmsd}, Copy={conf_copy.rmsd}.",
        )

    def test_default_none(self) -> None:
        """
        GIVEN  a freshly created Conformation
        WHEN   reading its rmsd attribute
        THEN   it must be None (not 0.0).
        """
        # Arrange: fresh Conformation from helper.
        conf = _build_extended("ALA")

        # Act & Assert: default RMSD is None.
        self.assertIsNone(
            conf.rmsd,
            f"Default RMSD must be None. Got {conf.rmsd}.",
        )


# ===================================================================
# 4.  TestConformationPLDDT  (3 tests)
# ===================================================================
class TestConformationPLDDT(unittest.TestCase):
    """
    Verify that the pLDDT attribute on Conformation can be set, retrieved,
    respects the [0.0, 100.0] boundary, and defaults to None.
    """

    def test_set_retrieve_853(self) -> None:
        """
        GIVEN  a fresh Conformation
        WHEN   plddt is set to 85.3
        THEN   reading plddt back must return 85.3.
        """
        # Arrange: build minimal conformation.
        conf = _build_extended("ALA")

        # Act: set pLDDT.
        conf.plddt = 85.3

        # Assert: retrieve and compare.
        self.assertEqual(
            conf.plddt,
            85.3,
            f"pLDDT set to 85.3 must be retrievable as 85.3. "
            f"Got {conf.plddt}.",
        )

    def test_boundaries_00_1000(self) -> None:
        """
        GIVEN  a fresh Conformation
        WHEN   plddt is set to 0.0 and then to 100.0
        THEN   both values must be stored and retrieved correctly.
        """
        # Arrange: build minimal conformation.
        conf = _build_extended("ALA")

        # Act & Assert: lower boundary.
        conf.plddt = 0.0
        self.assertEqual(
            conf.plddt,
            0.0,
            f"pLDDT lower boundary (0.0) must be retrievable. "
            f"Got {conf.plddt}.",
        )

        # Act & Assert: upper boundary.
        conf.plddt = 100.0
        self.assertEqual(
            conf.plddt,
            100.0,
            f"pLDDT upper boundary (100.0) must be retrievable. "
            f"Got {conf.plddt}.",
        )

    def test_default_none(self) -> None:
        """
        GIVEN  a freshly created Conformation
        WHEN   reading its plddt attribute
        THEN   it must be None (not 0.0).
        """
        # Arrange: fresh Conformation from helper.
        conf = _build_extended("ALA")

        # Act & Assert: default pLDDT is None.
        self.assertIsNone(
            conf.plddt,
            f"Default pLDDT must be None. Got {conf.plddt}.",
        )


# ===================================================================
# 5.  TestWeightedScoreConsistency  (2 tests)
# ===================================================================
class TestWeightedScoreConsistency(unittest.TestCase):
    """
    Verify that the energy function's internal weight-sum relationship
    holds: the reported total_energy must equal the dot product of the
    weight vector and the component vector.  Also verify all weights
    are within the (0, 5) interval.
    """

    def test_sum_w_times_component_matches_total(self) -> None:
        """
        GIVEN  an evaluated Conformation with energy_components populated
        WHEN   computing Σ(w_i * component_i) using the RosettaEnergyFunction
               instance's weights
        THEN   the result must equal conf.total_energy to 12 decimal places.
        """
        # Arrange: build extended chain and evaluate.
        conf = _build_extended("ACDEFGHIKLMNPQRSTVWY")
        energy_fn = RosettaEnergyFunction()
        energy_fn.evaluate(conf)

        # Act: compute weighted sum manually.
        w = {
            "lennard_jones": energy_fn.w_lj,
            "hydrogen_bond": energy_fn.w_hbond,
            "solvation": energy_fn.w_solvation,
            "ramachandran": energy_fn.w_rama,
            "repulsive": energy_fn.w_repulsive,
        }
        weighted_sum = sum(
            w[key] * conf.energy_components[key]
            for key in w
        )

        # Assert: match total_energy to 12 dp.
        self.assertAlmostEqual(
            weighted_sum,
            conf.total_energy,
            places=12,
            msg=(
                "Weighted sum of components (Σ w_i * component_i) must "
                "equal total_energy to 12 decimal places. "
                f"Weighted sum={weighted_sum:.15e}, "
                f"total_energy={conf.total_energy:.15e}."
            ),
        )

    def test_weights_in_0_to_5(self) -> None:
        """
        GIVEN  a default RosettaEnergyFunction instance
        WHEN   inspecting all five weight attributes
        THEN   each weight must be strictly greater than 0 and strictly
               less than 5.
        """
        # Arrange: create default energy function.
        energy_fn = RosettaEnergyFunction()

        # Collect all weight values.
        weights = {
            "w_lj": energy_fn.w_lj,
            "w_hbond": energy_fn.w_hbond,
            "w_solvation": energy_fn.w_solvation,
            "w_rama": energy_fn.w_rama,
            "w_repulsive": energy_fn.w_repulsive,
        }

        # Assert: each weight ∈ (0, 5).
        for name, val in weights.items():
            self.assertGreater(
                val,
                0.0,
                f"Weight '{name}' must be > 0. Got {val}.",
            )
            self.assertLess(
                val,
                5.0,
                f"Weight '{name}' must be < 5. Got {val}.",
            )


# ===================================================================
# 6.  TestScoreSequenceLengthScaling  (1 test)
# ===================================================================
class TestScoreSequenceLengthScaling(unittest.TestCase):
    """
    Verify that longer sequences produce larger-magnitude (more positive)
    total energies than shorter ones when both are in the extended
    conformation, and that all energies are finite.
    """

    def test_20mer_greater_than_5mer_energy(self) -> None:
        """
        GIVEN  an extended 5-residue polyalanine and an extended
               20-residue polyalanine
        WHEN   both are scored with RosettaEnergyFunction
        THEN   the 20mer total_energy must be greater (more positive)
               than the 5mer total_energy, and both must be finite.
        """
        # Arrange: build two chains of different lengths.
        conf_5 = _build_extended("AAAAA")
        conf_20 = _build_extended("A" * 20)
        energy_fn = RosettaEnergyFunction()

        # Act: evaluate both.
        energy_fn.evaluate(conf_5)
        energy_fn.evaluate(conf_20)

        # Assert: both finite.
        self.assertTrue(
            math.isfinite(conf_5.total_energy),
            f"5mer total energy must be finite. Got {conf_5.total_energy}.",
        )
        self.assertTrue(
            math.isfinite(conf_20.total_energy),
            f"20mer total energy must be finite. Got {conf_20.total_energy}.",
        )

        # Assert: 20mer energy > 5mer energy (more residues → more terms).
        self.assertGreater(
            conf_20.total_energy,
            conf_5.total_energy,
            (
                "20mer total energy should be greater (more positive) than "
                "5mer total energy for extended chains. "
                f"5mer={conf_5.total_energy:.10e}, "
                f"20mer={conf_20.total_energy:.10e}."
            ),
        )


# ===================================================================
# 7.  TestRamachandranScorePrecision  (2 tests)
# ===================================================================
class TestRamachandranScorePrecision(unittest.TestCase):
    """
    Verify that the Ramachandran energy term produces finite scores for
    every standard amino acid and that even a single residue yields a
    finite backbone preference score.
    """

    def test_all_AAs_produce_finite_rama(self) -> None:
        """
        GIVEN  one extended tetra-peptide for each of the 20 standard AAs
        WHEN   RosettaEnergyFunction.evaluate() is called
        THEN   the 'ramachandran' component must be finite for every AA.
        """
        # Arrange: all 20 standard one-letter codes.
        one_letter_codes = list("ACDEFGHIKLMNPQRSTVWY")
        energy_fn = RosettaEnergyFunction()

        for aa in one_letter_codes:
            with self.subTest(aa=aa):
                # Build a 4mer of the same residue.
                conf = _build_extended(aa * 4)

                # Act: evaluate.
                energy_fn.evaluate(conf)

                # Assert: Ramachandran component is finite.
                rama_score = conf.energy_components.get("ramachandran", None)
                self.assertIsNotNone(
                    rama_score,
                    f"Energy components must include 'ramachandran' for {aa}.",
                )
                self.assertTrue(
                    math.isfinite(rama_score),
                    f"Ramachandran score must be finite for {aa}-4mer. "
                    f"Got {rama_score}.",
                )

    def test_single_residue_finite(self) -> None:
        """
        GIVEN  a Conformation with exactly one residue (Alanine)
        WHEN   RosettaEnergyFunction.evaluate() is called
        THEN   the 'ramachandran' component must be finite.
        """
        # Arrange: single-residue conformation.
        conf = _build_extended("A")
        energy_fn = RosettaEnergyFunction()

        # Act: evaluate.
        energy_fn.evaluate(conf)

        # Assert: Ramachandran component is finite.
        rama_score = conf.energy_components.get("ramachandran", None)
        self.assertIsNotNone(
            rama_score,
            "Energy components must include 'ramachandran' for a single residue.",
        )
        self.assertTrue(
            math.isfinite(rama_score),
            f"Ramachandran score must be finite for a single residue. "
            f"Got {rama_score}.",
        )


# ===================================================================
# 8.  TestFoldProteinBenchmark  (2 tests)
# ===================================================================
class TestFoldProteinBenchmark(unittest.TestCase):
    """
    Run the full fold_protein pipeline on a small test sequence and
    verify that the produced Conformation has finite energies and the
    correct number of residues.
    """

    def test_scores_finite(self) -> None:
        """
        GIVEN  a short test sequence
        WHEN   fold_protein() is called with a small number of steps
        THEN   the returned Conformation must have finite total_energy
               and finite energy_components.
        """
        # Arrange: define a small folding target.
        test_seq = "AAAAK"

        # Act: run a minimal folding simulation.
        conf = fold_protein(test_seq, n_steps=500, temperature=300.0)

        # Assert: total energy is finite.
        self.assertTrue(
            math.isfinite(conf.total_energy),
            f"Total energy after fold_protein must be finite. "
            f"Got {conf.total_energy}.",
        )

        # Assert: each component is finite.
        for comp_name, comp_val in conf.energy_components.items():
            self.assertTrue(
                math.isfinite(comp_val),
                f"Component '{comp_name}' after fold_protein must be "
                f"finite. Got {comp_val}.",
            )

    def test_correct_residue_count(self) -> None:
        """
        GIVEN  a test sequence of known length
        WHEN   fold_protein() is called
        THEN   the number of residues in the output Conformation must
               match the length of the input sequence.
        """
        # Arrange: define a test sequence.
        test_seq = "MLSDGEFQL"
        expected_length = len(test_seq)

        # Act: run folding.
        conf = fold_protein(test_seq, n_steps=500, temperature=300.0)

        # Assert: residue count matches.
        actual_length = len(conf.residues)
        self.assertEqual(
            actual_length,
            expected_length,
            f"fold_protein must produce {expected_length} residues "
            f"(matching the input sequence), but got {actual_length}.",
        )


# ===================================================================
# 9.  TestGDT_TSStub  (1 test)
# ===================================================================
class TestGDT_TSStub(unittest.TestCase):
    """
    Stub test for GDT_TS (Global Distance Test – Total Score).
    The benchmark module is not yet implemented, so this test is
    gracefully skipped.
    """

    def test_import_gdt_ts_stub(self) -> None:
        """
        GIVEN  that the protein_folding_ai.benchmark.casp_scorer module
               does not exist
        WHEN   attempting to import it
        THEN   a ModuleNotFoundError is raised and the test is skipped
               with a clear message.
        """
        try:
            # Attempt to import the not-yet-implemented benchmark module.
            from protein_folding_ai.benchmark import casp_scorer  # type: ignore[import-unused]
            # If import somehow succeeds, the test fails because the stub
            # should not exist yet.
            self.fail(
                "Expected ModuleNotFoundError — benchmark.casp_scorer "
                "module is not yet implemented. "
                "Kindly ensure the benchmark/ package is created before "
                "removing this skip."
            )
        except ModuleNotFoundError:
            raise unittest.SkipTest(
                "GDT_TS benchmark module (benchmark.casp_scorer) is not "
                "available yet. This test will be enabled once the module "
                "lands — please verify the implementation."
            )


# ===================================================================
# 10.  TestLDDTStub  (1 test)
# ===================================================================
class TestLDDTStub(unittest.TestCase):
    """
    Stub test for lDDT (local Distance Difference Test).
    Gracefully skipped until the benchmark module is implemented.
    """

    def test_import_lddt_stub(self) -> None:
        """
        GIVEN  that the protein_folding_ai.benchmark.lddt module does
               not exist
        WHEN   attempting to import it
        THEN   a ModuleNotFoundError is raised and the test is skipped.
        """
        try:
            from protein_folding_ai.benchmark import lddt  # type: ignore[import-unused]
            self.fail(
                "Expected ModuleNotFoundError — benchmark.lddt "
                "module is not yet implemented."
            )
        except ModuleNotFoundError:
            raise unittest.SkipTest(
                "lDDT benchmark module (benchmark.lddt) is not "
                "available yet. Please verify when the module lands."
            )


# ===================================================================
# 11.  TestTMScoreStub  (1 test)
# ===================================================================
class TestTMScoreStub(unittest.TestCase):
    """
    Stub test for TM-score (Template Modeling score).
    Gracefully skipped until the benchmark module is implemented.
    """

    def test_import_tm_score_stub(self) -> None:
        """
        GIVEN  that the protein_folding_ai.benchmark.tm_score module
               does not exist
        WHEN   attempting to import it
        THEN   a ModuleNotFoundError is raised and the test is skipped.
        """
        try:
            from protein_folding_ai.benchmark import tm_score  # type: ignore[import-unused]
            self.fail(
                "Expected ModuleNotFoundError — benchmark.tm_score "
                "module is not yet implemented."
            )
        except ModuleNotFoundError:
            raise unittest.SkipTest(
                "TM-score benchmark module (benchmark.tm_score) is not "
                "available yet. The same has been noted and will be "
                "enabled upon implementation."
            )


# ===================================================================
# 12.  TestSPICKERStub  (1 test)
# ===================================================================
class TestSPICKERStub(unittest.TestCase):
    """
    Stub test for SPICKER (clustering-based model selection).
    Gracefully skipped until the benchmark module is implemented.
    """

    def test_import_spicker_stub(self) -> None:
        """
        GIVEN  that the protein_folding_ai.benchmark.spicker module
               does not exist
        WHEN   attempting to import it
        THEN   a ModuleNotFoundError is raised and the test is skipped.
        """
        try:
            from protein_folding_ai.benchmark import spicker  # type: ignore[import-unused]
            self.fail(
                "Expected ModuleNotFoundError — benchmark.spicker "
                "module is not yet implemented."
            )
        except ModuleNotFoundError:
            raise unittest.SkipTest(
                "SPICKER benchmark module (benchmark.spicker) is not "
                "available yet. Kindly verify once implemented."
            )


# ===================================================================
# 13.  TestRgSASAStub  (1 test)
# ===================================================================
class TestRgSASAStub(unittest.TestCase):
    """
    Stub test for Rg (radius of gyration) and SASA (solvent-accessible
    surface area) analysis.  Gracefully skipped until the benchmark
    module is implemented.
    """

    def test_import_rg_sasa_stub(self) -> None:
        """
        GIVEN  that the protein_folding_ai.benchmark.rg_sasa module
               does not exist
        WHEN   attempting to import it
        THEN   a ModuleNotFoundError is raised and the test is skipped.
        """
        try:
            from protein_folding_ai.benchmark import rg_sasa  # type: ignore[import-unused]
            self.fail(
                "Expected ModuleNotFoundError — benchmark.rg_sasa "
                "module is not yet implemented."
            )
        except ModuleNotFoundError:
            raise unittest.SkipTest(
                "Rg-SASA benchmark module (benchmark.rg_sasa) is not "
                "available yet. Please verify when the module lands."
            )


# ===================================================================
# 14.  TestScoreEdgeCases  (2 tests)
# ===================================================================
class TestScoreEdgeCases(unittest.TestCase):
    """
    Verify that the energy function handles edge-case conformations
    gracefully — specifically single-residue systems and overlapping
    residues — without producing NaN or infinite energies.
    """

    def test_single_residue_only_rama_nonzero(self) -> None:
        """
        GIVEN  a Conformation with exactly one residue (Alanine)
        WHEN   RosettaEnergyFunction.evaluate() is called
        THEN   the pairwise terms (lennard_jones, hydrogen_bond, repulsive)
               must be exactly zero (no pairs to evaluate), while the
               ramachandran term must be non-zero (it depends only on
               the single residue's phi/psi).
        """
        # Arrange: single-residue conformation.
        conf = _build_extended("A")
        energy_fn = RosettaEnergyFunction()

        # Act: evaluate.
        energy_fn.evaluate(conf)

        # Assert: pairwise terms are zero.
        self.assertEqual(
            conf.energy_components.get("lennard_jones", None),
            0.0,
            "Lennard-Jones energy must be zero for a single residue "
            f"(no pairs). Got {conf.energy_components.get('lennard_jones')}.",
        )
        self.assertEqual(
            conf.energy_components.get("hydrogen_bond", None),
            0.0,
            "Hydrogen bond energy must be zero for a single residue "
            f"(no donor-acceptor pairs). Got {conf.energy_components.get('hydrogen_bond')}.",
        )
        self.assertEqual(
            conf.energy_components.get("repulsive", None),
            0.0,
            "Repulsive energy must be zero for a single residue "
            f"(no pairs). Got {conf.energy_components.get('repulsive')}.",
        )

        # Assert: Ramachandran term is non-zero.
        rama = conf.energy_components.get("ramachandran", None)
        self.assertIsNotNone(
            rama,
            "Energy components must include 'ramachandran'.",
        )
        self.assertNotEqual(
            rama,
            0.0,
            f"Ramachandran score should be non-zero for a single residue "
            f"with phi=-135, psi=135. Got {rama}.",
        )

        # Assert: total energy is finite.
        self.assertTrue(
            math.isfinite(conf.total_energy),
            f"Total energy must be finite for a single residue. "
            f"Got {conf.total_energy}.",
        )

    def test_overlapping_residues_finite(self) -> None:
        """
        GIVEN  a dipeptide with CB atoms placed 0.5 Å apart
               (well within VDW radius for any residue pair)
        WHEN   RosettaEnergyFunction.evaluate() is called
        THEN   the total energy and all components must be finite
               (no explosion from the repulsive term).
        """
        # Arrange: build a dipeptide, then manually place the two CB
        #         atoms extremely close (0.5 Å) to trigger strong
        #         repulsive and LJ forces.
        conf = _build_extended("AA")

        # Place first residue's CB at the origin.
        conf.residues[0].CB = np.array([0.0, 0.0, 0.0], dtype=float)
        # Place second residue's CB just 0.5 Å away along X.
        conf.residues[1].CB = np.array([0.5, 0.0, 0.0], dtype=float)

        energy_fn = RosettaEnergyFunction()

        # Act: evaluate.
        energy_fn.evaluate(conf)

        # Assert: total energy is finite.
        self.assertTrue(
            math.isfinite(conf.total_energy),
            f"Total energy with overlapping residues (0.5 Å) must be "
            f"finite. Got {conf.total_energy}.",
        )

        # Assert: each component is finite.
        for comp_name, comp_val in conf.energy_components.items():
            self.assertTrue(
                math.isfinite(comp_val),
                f"Component '{comp_name}' with overlapping residues "
                f"(0.5 Å) must be finite. Got {comp_val}.",
            )


# ===================================================================
# Test suite entry point
# ===================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
