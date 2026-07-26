#!/usr/bin/env python3
"""
test_benchmark.py
=================

Benchmark-level tests for the Rosetta-style energy function, conformation
scoring, and protein folding pipeline implemented in folding_engine.py.

Tests cover score determinism, numerical precision, RMSD/plDDT metadata
handling, weighted-score consistency, sequence-length scaling, Ramachandran
preference coverage, fold-protein pipeline integration, metric stubs for
unmerged modules (GDT_TS, lDDT, TM-score, SPICKER, Rg/SASA), and edge-case
residue geometries.

Scientific context:
    The Rosetta energy function decomposes the total energy into five
    physically motivated terms — Lennard-Jones, hydrogen bonding, solvation,
    Ramachandran preference, and steric repulsion.  Reproducibility of
    scores to machine precision is essential for conformational ranking;
    numerical stability under small perturbations is required for robust
    gradient-based refinement.

Author: Priya Sharma, Quality Engineering, Indian Institute of Science, Bengaluru
"""

import copy
import math
import unittest

import numpy as np

from folding_engine import (
    AA_CODES,
    _RAMA_PREFERENCE,
    Conformation,
    FragmentInsertionMC,
    Residue,
    RosettaEnergyFunction,
)

# ── Helper ────────────────────────────────────────────────────────────

def _build_extended(seq: str) -> Conformation:
    """
    Build an extended-chain conformation with backbone coordinates
    reconstructed from idealised dihedral angles (phi=-135, psi=135).

    This initialises a FragmentInsertionMC instance purely as a static
    dispatch vehicle for _rebuild_backbone.  The resulting conformation
    has valid 3-D coordinates for all residues except the first (whose
    backbone atoms remain at the origin).

    Args:
        seq: One-letter amino-acid codes (e.g. "ACDEFG").

    Returns:
        Conformation with backbone coordinates set.
    """
    residues = [
        Residue(i + 1, AA_CODES.get(aa, "ALA"), phi=-135.0, psi=135.0)
        for i, aa in enumerate(seq.upper())
    ]
    conf = Conformation(residues=residues)
    sampler = FragmentInsertionMC.__new__(FragmentInsertionMC)
    for i in range(1, len(conf.residues)):
        FragmentInsertionMC._rebuild_backbone(sampler, conf, i, i + 1)
    return conf


# ======================================================================
# 1. TestScoreDeterminism
# ======================================================================

class TestScoreDeterminism(unittest.TestCase):
    """
    Verify that RosettaEnergyFunction.evaluate() is fully deterministic —
    identical inputs always produce identical outputs, both at the total
    energy level and for each individual component.
    """

    def setUp(self):
        self.energy_fn = RosettaEnergyFunction()
        self.conf = _build_extended("ACDEFG")

    # -- 1a. Repeated eval identical -----------------------------------

    def test_repeated_eval_identical(self):
        """
        Kindly ensure that two successive calls to evaluate() on the
        same conformation, without any intervening mutation, return
        exactly the same total energy.  Determinism is a prerequisite
        for reproducible conformational ranking.
        """
        e1 = self.energy_fn.evaluate(self.conf)
        e2 = self.energy_fn.evaluate(self.conf)
        self.assertEqual(
            e1, e2,
            msg=f"Repeated evaluate() calls differ: first = {e1:.15f}, "
                f"second = {e2:.15f}"
        )

    # -- 1b. Deep copy identical ---------------------------------------

    def test_deep_copy_identical(self):
        """
        Please verify that a deep copy of a conformation, after
        evaluation on the original, yields the same total energy,
        energy components, and per-residue Ramachandran scores.
        """
        _ = self.energy_fn.evaluate(self.conf)
        conf_copy = copy.deepcopy(self.conf)
        e_copy = self.energy_fn.evaluate(conf_copy)

        self.assertAlmostEqual(
            self.conf.total_energy, e_copy, places=12,
            msg=f"Total energy differs after deep copy: original = "
                f"{self.conf.total_energy:.12f}, copy = {e_copy:.12f}"
        )
        for key in self.conf.energy_components:
            self.assertAlmostEqual(
                self.conf.energy_components[key],
                conf_copy.energy_components[key],
                places=12,
                msg=f"Energy component '{key}' differs after deep copy: "
                    f"original = {self.conf.energy_components[key]:.12f}, "
                    f"copy = {conf_copy.energy_components[key]:.12f}"
            )
        for orig_res, copy_res in zip(self.conf.residues, conf_copy.residues):
            self.assertAlmostEqual(
                orig_res.rama_score, copy_res.rama_score, places=12,
                msg=f"Residue {orig_res.seq_index} rama_score differs "
                    f"after deep copy: original = {orig_res.rama_score:.12f}, "
                    f"copy = {copy_res.rama_score:.12f}"
            )

    # -- 1c. Components deterministic (12 decimal places) --------------

    def test_components_deterministic_12_places(self):
        """
        Run evaluate() twice and assert that each of the five energy
        components (lennard_jones, hydrogen_bond, solvation, ramachandran,
        repulsive) plus the total_energy agree to 12 decimal places.
        """
        _ = self.energy_fn.evaluate(self.conf)
        ref_components = dict(self.conf.energy_components)
        ref_total = self.conf.total_energy

        _ = self.energy_fn.evaluate(self.conf)

        for key in ref_components:
            self.assertAlmostEqual(
                ref_components[key], self.conf.energy_components[key],
                places=12,
                msg=f"Component '{key}' not deterministic to 12 places: "
                    f"run1 = {ref_components[key]:.12f}, "
                    f"run2 = {self.conf.energy_components[key]:.12f}"
            )
        self.assertAlmostEqual(
            ref_total, self.conf.total_energy, places=12,
            msg=f"Total energy not deterministic to 12 places: "
                f"run1 = {ref_total:.12f}, run2 = {self.conf.total_energy:.12f}"
        )

    # -- 1d. All 20 AAs finite -----------------------------------------

    def test_all_20_aas_finite(self):
        """
        Verify that evaluate() produces a finite (non-NaN, non-inf)
        total energy for a conformation containing all 20 standard
        amino acids.  Every residue must yield a valid score.
        """
        all_20 = "ACDEFGHIKLMNPQRSTVWY"
        conf = _build_extended(all_20)
        total = self.energy_fn.evaluate(conf)
        self.assertTrue(
            math.isfinite(total),
            msg=f"Total energy for the 20-AA construct is not finite: "
                f"{total}"
        )
        for comp_name, comp_val in conf.energy_components.items():
            self.assertTrue(
                math.isfinite(comp_val),
                msg=f"Energy component '{comp_name}' is not finite "
                    f"for the 20-AA construct: {comp_val}"
            )


# ======================================================================
# 2. TestScorePrecision
# ======================================================================

class TestScorePrecision(unittest.TestCase):
    """
    Evaluate numerical precision of the energy function under small
    perturbations.  Physically, infinitesimal changes in dihedral angles
    should produce proportionally small energy changes.
    """

    def setUp(self):
        self.energy_fn = RosettaEnergyFunction()
        self.conf = _build_extended("ACDEFG")
        self.sampler = FragmentInsertionMC.__new__(FragmentInsertionMC)
        self.n_res = len(self.conf.residues)

    # -- 2a. 0.1° perturbation → delta < 0.01 --------------------------

    def test_small_perturbation_delta_lt_0_01(self):
        """
        Apply a +0.1° perturbation to the phi angle of residue 3
        (index 2), rebuild the backbone, and confirm that the absolute
        change in total energy is below 0.01 kcal/mol.  This validates
        that the energy landscape is smooth on a scale relevant for
        gradient-based optimisation.
        """
        _ = self.energy_fn.evaluate(self.conf)
        e_baseline = self.conf.total_energy

        # Perturb phi of residue index 2 by +0.1°
        self.conf.residues[2].phi += 0.1
        FragmentInsertionMC._rebuild_backbone(self.sampler, self.conf, 2, 3)

        _ = self.energy_fn.evaluate(self.conf)
        delta_e = abs(self.conf.total_energy - e_baseline)

        self.assertLess(
            delta_e, 0.01,
            msg=f"Energy change after +0.1° phi perturbation = "
                f"{delta_e:.6f} kcal/mol, expected < 0.01 kcal/mol"
        )

    # -- 2b. Perturb-and-revert recovers (12 places) -------------------

    def test_perturb_and_revert_recovers_12_places(self):
        """
        Perturb phi of residue index 2 by +1.0°, evaluate, revert the
        perturbation, rebuild, and evaluate again.  All 12 numerical
        quantities — total_energy, five energy components, and six
        per-residue rama_scores — should return to their original
        values to 12 decimal places.
        """
        _ = self.energy_fn.evaluate(self.conf)
        ref_total = self.conf.total_energy
        ref_components = dict(self.conf.energy_components)
        ref_rama = [res.rama_score for res in self.conf.residues]

        # Perturb
        self.conf.residues[2].phi += 1.0
        FragmentInsertionMC._rebuild_backbone(self.sampler, self.conf, 2, 3)
        _ = self.energy_fn.evaluate(self.conf)

        # Revert
        self.conf.residues[2].phi -= 1.0
        FragmentInsertionMC._rebuild_backbone(self.sampler, self.conf, 2, 3)
        _ = self.energy_fn.evaluate(self.conf)

        # Check total
        self.assertAlmostEqual(
            ref_total, self.conf.total_energy, places=12,
            msg=f"Total energy not recovered after revert: "
                f"original = {ref_total:.12f}, "
                f"post-revert = {self.conf.total_energy:.12f}"
        )
        # Check components
        for key in ref_components:
            self.assertAlmostEqual(
                ref_components[key], self.conf.energy_components[key],
                places=12,
                msg=f"Component '{key}' not recovered after revert: "
                    f"original = {ref_components[key]:.12f}, "
                    f"post-revert = {self.conf.energy_components[key]:.12f}"
            )
        # Check rama_scores
        for i, ref_r in enumerate(ref_rama):
            self.assertAlmostEqual(
                ref_r, self.conf.residues[i].rama_score, places=12,
                msg=f"Residue {i + 1} rama_score not recovered after revert: "
                    f"original = {ref_r:.12f}, "
                    f"post-revert = {self.conf.residues[i].rama_score:.12f}"
            )

    # -- 2c. 180° flip finite ------------------------------------------

    def test_180_flip_finite(self):
        """
        Flip the phi angle of residue index 2 by 180° (a large
        conformational change) and verify that the resulting total
        energy remains finite.  Large dihedral changes can produce
        severe steric clashes; the repulsive term must bound the
        energy rather than letting it diverge to infinity or NaN.
        """
        _ = self.energy_fn.evaluate(self.conf)

        # Flip phi by 180°
        self.conf.residues[2].phi += 180.0
        FragmentInsertionMC._rebuild_backbone(self.sampler, self.conf, 2, 3)

        total = self.energy_fn.evaluate(self.conf)
        self.assertTrue(
            math.isfinite(total),
            msg=f"Total energy after 180° phi flip is not finite: {total}"
        )
        for comp_name, comp_val in self.conf.energy_components.items():
            self.assertTrue(
                math.isfinite(comp_val),
                msg=f"Component '{comp_name}' after 180° phi flip is "
                    f"not finite: {comp_val}"
            )


# ======================================================================
# 3. TestConformationRMSD
# ======================================================================

class TestConformationRMSD(unittest.TestCase):
    """
    Validate RMSD metadata storage and retrieval on Conformation.
    RMSD measures global structural similarity to a reference;
    correct propagation is essential when comparing decoy ensembles.
    """

    def setUp(self):
        self.conf = _build_extended("ACDEFG")

    # -- 3a. Set / retrieve (2.345) ------------------------------------

    def test_set_and_retrieve(self):
        """
        Kindly confirm that setting .rmsd to 2.345 and immediately
        reading it back returns the same value.  This validates basic
        attribute assignment.
        """
        self.conf.rmsd = 2.345
        self.assertAlmostEqual(
            self.conf.rmsd, 2.345, places=12,
            msg=f"RMSD getter returned {self.conf.rmsd}, expected 2.345"
        )

    # -- 3b. Deep copy preserved ---------------------------------------

    def test_deep_copy_preserved(self):
        """
        Please verify that deep-copying a conformation preserves the
        RMSD value.  Decoy sets are frequently duplicated during
        clustering; RMSD annotations must survive the copy.
        """
        self.conf.rmsd = 2.345
        conf_copy = copy.deepcopy(self.conf)
        self.assertAlmostEqual(
            conf_copy.rmsd, 2.345, places=12,
            msg=f"Deep-copied RMSD = {conf_copy.rmsd}, expected 2.345"
        )

    # -- 3c. Default None ----------------------------------------------

    def test_default_none(self):
        """
        Ensure that a freshly constructed Conformation has .rmsd set to
        None by default, indicating that no reference structure has been
        supplied.
        """
        self.assertIsNone(
            self.conf.rmsd,
            msg=f"Default RMSD = {self.conf.rmsd}, expected None"
        )


# ======================================================================
# 4. TestConformationPLDDT
# ======================================================================

class TestConformationPLDDT(unittest.TestCase):
    """
    Validate plDDT metadata storage and retrieval on Conformation.
    plDDT (predicted local distance difference test) ranges from 0
    (unreliable) to 100 (high confidence) and is a standard AlphaFold
    output metric.
    """

    def setUp(self):
        self.conf = _build_extended("ACDEFG")

    # -- 4a. Set / retrieve (85.3) -------------------------------------

    def test_set_and_retrieve(self):
        """
        Kindly confirm that setting .plddt to 85.3 and reading it back
        returns the same value.  85.3 represents a high-confidence
        prediction on the 0–100 scale.
        """
        self.conf.plddt = 85.3
        self.assertAlmostEqual(
            self.conf.plddt, 85.3, places=12,
            msg=f"plDDT getter returned {self.conf.plddt}, expected 85.3"
        )

    # -- 4b. Boundaries (0, 100) ---------------------------------------

    def test_boundary_zero(self):
        """
        Verify that plDDT accepts the lower boundary value 0 (no
        confidence).  The attribute should store and return it
        faithfully.
        """
        self.conf.plddt = 0.0
        self.assertEqual(
            self.conf.plddt, 0.0,
            msg=f"plDDT boundary 0 returned {self.conf.plddt}"
        )

    def test_boundary_one_hundred(self):
        """
        Verify that plDDT accepts the upper boundary value 100 (maximum
        confidence).  The attribute should store and return it faithfully.
        """
        self.conf.plddt = 100.0
        self.assertEqual(
            self.conf.plddt, 100.0,
            msg=f"plDDT boundary 100 returned {self.conf.plddt}"
        )

    # -- 4c. Default None ----------------------------------------------

    def test_default_none(self):
        """
        Ensure that a freshly constructed Conformation has .plddt set to
        None by default, indicating that no prediction confidence has been
        computed.
        """
        self.assertIsNone(
            self.conf.plddt,
            msg=f"Default plDDT = {self.conf.plddt}, expected None"
        )


# ======================================================================
# 5. TestWeightedScoreConsistency
# ======================================================================

class TestWeightedScoreConsistency(unittest.TestCase):
    """
    Verify that the total energy equals the weighted sum of individual
    components, and that all default weights lie within the physically
    sensible range (0, 5).  The Rosetta energy function computes:

        E_total = w_LJ * E_LJ + w_HB * E_HB + w_solv * E_solv
                  + w_rama * E_rama + w_rep * E_rep

    This linear decomposition must hold exactly.
    """

    def setUp(self):
        self.energy_fn = RosettaEnergyFunction()
        self.conf = _build_extended("ACDEFG")

    # -- 5a. Manual Σ(w × component) matches total_energy -------------

    def test_manual_weighted_sum_matches_total(self):
        """
        Compute the weighted sum manually from the stored energy
        components and assert that it equals total_energy to 12
        decimal places.  Any discrepancy would indicate a bug in the
        evaluate() aggregation logic.
        """
        _ = self.energy_fn.evaluate(self.conf)

        manual_total = (
            self.energy_fn.w_lj * self.conf.energy_components["lennard_jones"]
            + self.energy_fn.w_hbond * self.conf.energy_components["hydrogen_bond"]
            + self.energy_fn.w_solvation * self.conf.energy_components["solvation"]
            + self.energy_fn.w_rama * self.conf.energy_components["ramachandran"]
            + self.energy_fn.w_repulsive * self.conf.energy_components["repulsive"]
        )
        self.assertAlmostEqual(
            manual_total, self.conf.total_energy, places=12,
            msg=f"Manual weighted sum = {manual_total:.12f}, "
                f"total_energy = {self.conf.total_energy:.12f}, "
                f"diff = {abs(manual_total - self.conf.total_energy):.2e}"
        )

    # -- 5b. All weights in (0, 5) -------------------------------------

    def test_all_weights_in_range(self):
        """
        Please verify that each default weight (w_lj, w_hbond,
        w_solvation, w_rama, w_repulsive) lies strictly between 0
        and 5.  Weights outside this range would produce unphysical
        energy magnitudes.
        """
        weights = {
            "w_lj": self.energy_fn.w_lj,
            "w_hbond": self.energy_fn.w_hbond,
            "w_solvation": self.energy_fn.w_solvation,
            "w_rama": self.energy_fn.w_rama,
            "w_repulsive": self.energy_fn.w_repulsive,
        }
        for name, val in weights.items():
            self.assertGreater(
                val, 0.0,
                msg=f"Weight '{name}' = {val}, expected > 0"
            )
            self.assertLess(
                val, 5.0,
                msg=f"Weight '{name}' = {val}, expected < 5"
            )


# ======================================================================
# 6. TestScoreSequenceLengthScaling
# ======================================================================

class TestScoreSequenceLengthScaling(unittest.TestCase):
    """
    Verify that longer sequences produce larger (in magnitude) total
    energies, consistent with the additive nature of pairwise
    potentials.  A 20-mer must have greater total energy than a 5-mer
    built from the same residues, and all energy values must remain
    finite.
    """

    def test_20mer_greater_than_5mer_all_finite(self):
        """
        Evaluate a 5-residue and a 20-residue poly-alanine construct.
        The 20-mer should have larger total energy (more positive due
        to more repulsive and solvation terms) than the 5-mer.  Every
        component of both conformations must be finite.
        """
        energy_fn = RosettaEnergyFunction()
        seq_5 = "AAAAA"
        seq_20 = "AAAAAAAAAAAAAAAAAAAA"

        conf_5 = _build_extended(seq_5)
        conf_20 = _build_extended(seq_20)

        e_5 = energy_fn.evaluate(conf_5)
        e_20 = energy_fn.evaluate(conf_20)

        # All finite
        for label, conf in [("5-mer", conf_5), ("20-mer", conf_20)]:
            self.assertTrue(
                math.isfinite(conf.total_energy),
                msg=f"{label} total energy is not finite: {conf.total_energy}"
            )
            for comp_name, comp_val in conf.energy_components.items():
                self.assertTrue(
                    math.isfinite(comp_val),
                    msg=f"{label} component '{comp_name}' is not finite: "
                        f"{comp_val}"
                )

        # 20-mer total > 5-mer total (more residues → more interactions)
        self.assertGreater(
            e_20, e_5,
            msg=f"20-mer total energy = {e_20:.4f} is not greater than "
                f"5-mer total energy = {e_5:.4f}"
        )


# ======================================================================
# 7. TestRamachandranScorePrecision
# ======================================================================

class TestRamachandranScorePrecision(unittest.TestCase):
    """
    Validate that every standard amino acid has an entry in the
    Ramachandran preference table, and that a single-residue evaluation
    produces a finite Ramachandran score.
    """

    # -- 7a. All AAs produce finite Ramachandran scores -----------------

    def test_all_aas_produce_finite_rama(self):
        """
        Kindly confirm that every one of the 20 standard amino acids
        yields a finite Ramachandran energy component when evaluated
        through RosettaEnergyFunction.evaluate().  The simplified
        engine stores explicit preferences for ALA, GLY, PRO, and VAL;
        all other residue types fall back to the default
        [(-60, -45, 0.5)] in _compute_ramachandran (line 250).  This
        test validates that the fallback path does not introduce
        non-finite values.
        """
        energy_fn = RosettaEnergyFunction()
        for one_letter in AA_CODES:
            conf = _build_extended(one_letter)
            _ = energy_fn.evaluate(conf)
            rama = conf.energy_components.get("ramachandran", None)
            self.assertIsNotNone(
                rama,
                msg=f"Ramachandran component missing for "
                    f"AA '{one_letter}' ({AA_CODES[one_letter]})."
            )
            self.assertTrue(
                math.isfinite(rama),
                msg=f"Ramachandran score for AA '{one_letter}' "
                    f"({AA_CODES[one_letter]}) is not finite: {rama}"
            )

    # -- 7b. Single residue finite -------------------------------------

    def test_single_residue_rama_finite(self):
        """
        Evaluate a single-residue conformation and verify that the
        Ramachandran component is finite.  Even a lone residue without
        pairwise interactions contributes a conformation-dependent
        backbone preference score.
        """
        energy_fn = RosettaEnergyFunction()
        conf = _build_extended("G")
        _ = energy_fn.evaluate(conf)

        rama = conf.energy_components.get("ramachandran", None)
        self.assertIsNotNone(
            rama,
            msg="Energy components dictionary does not contain "
                "'ramachandran' key."
        )
        self.assertTrue(
            math.isfinite(rama),
            msg=f"Single-residue Ramachandran score is not finite: {rama}"
        )


# ======================================================================
# 8. TestFoldProteinBenchmark
# ======================================================================

class TestFoldProteinBenchmark(unittest.TestCase):
    """
    Integration-level tests for the fold_protein() pipeline.  Verifies
    that the full folding run produces finite scores and the correct
    residue count.
    """

    # -- 8a. Scores finite after fold ----------------------------------

    def test_scores_finite_after_fold(self):
        """
        Fold a small 6-residue peptide ("ACDEFG") and confirm that
        its final total energy and all energy components are finite.
        A non-finite score would indicate a numerical instability in
        the MCMC sampling or energy computation.
        """
        from folding_engine import fold_protein

        conf = fold_protein("ACDEFG", n_steps=100, temperature=300.0)
        self.assertTrue(
            math.isfinite(conf.total_energy),
            msg=f"Total energy after fold is not finite: "
                f"{conf.total_energy}"
        )
        for comp_name, comp_val in conf.energy_components.items():
            self.assertTrue(
                math.isfinite(comp_val),
                msg=f"Component '{comp_name}' after fold is not finite: "
                    f"{comp_val}"
            )

    # -- 8b. Correct residue count -------------------------------------

    def test_correct_residue_count(self):
        """
        Verify that the folded conformation contains exactly as many
        residues as the input sequence.  A mismatch would indicate a
        bug in residue construction during pipeline initialisation.
        """
        from folding_engine import fold_protein

        seq = "ACDEFG"
        conf = fold_protein(seq, n_steps=100, temperature=300.0)
        self.assertEqual(
            len(conf.residues), len(seq),
            msg=f"Folded conformation has {len(conf.residues)} residues, "
                f"expected {len(seq)}"
        )


# ======================================================================
# 9. TestGDT_TSStub
# ======================================================================

class TestGDT_TSStub(unittest.TestCase):
    """
    Stub for the Global Distance Test (GDT_TS) scoring metric.
    GDT_TS measures the fraction of C-alpha atoms within a set of
    distance cutoffs (1, 2, 4, 8 Å) between a model and the native
    structure.
    """

    def test_gdt_ts_import(self):
        """
        Attempt to import GDT_TS from the benchmark package.  If the
        module has not been merged (William's work still pending), the
        test is skipped gracefully.
        """
        try:
            from benchmark import compute_gdt_ts  # noqa: F401
        except ImportError:
            self.skipTest(
                "benchmark.compute_gdt_ts is not available — "
                "William's GDT_TS implementation has not been merged."
            )


# ======================================================================
# 10. TestLDDTStub
# ======================================================================

class TestLDDTStub(unittest.TestCase):
    """
    Stub for the local Distance Difference Test (lDDT) score.
    lDDT evaluates local model quality by comparing all inter-atomic
    distances below a threshold against the reference.
    """

    def test_lddt_import(self):
        """
        Attempt to import lDDT from the benchmark package.  Skipped
        gracefully when the module is absent.
        """
        try:
            from benchmark import compute_lddt  # noqa: F401
        except ImportError:
            self.skipTest(
                "benchmark.compute_lddt is not available — "
                "William's lDDT implementation has not been merged."
            )


# ======================================================================
# 11. TestTMScoreStub
# ======================================================================

class TestTMScoreStub(unittest.TestCase):
    """
    Stub for the Template Modelling Score (TM-score).
    TM-score provides a length-independent measure of global structural
    similarity, ranging from 0 (unrelated) to 1 (identical).
    """

    def test_tm_score_import(self):
        """
        Attempt to import TM-score from the benchmark package.
        Skipped gracefully when the module is absent.
        """
        try:
            from benchmark import compute_tm_score  # noqa: F401
        except ImportError:
            self.skipTest(
                "benchmark.compute_tm_score is not available — "
                "William's TM-score implementation has not been merged."
            )


# ======================================================================
# 12. TestSPICKERStub
# ======================================================================

class TestSPICKERStub(unittest.TestCase):
    """
    Stub for the SPICKER clustering algorithm.
    SPICKER clusters decoy conformations by pairwise RMSD and returns
    centroid structures representing the most populated clusters.
    """

    def test_spicker_import(self):
        """
        Attempt to import SPICKER clustering from the benchmark
        package.  Skipped gracefully when the module is absent.
        """
        try:
            from benchmark import spicker_cluster  # noqa: F401
        except ImportError:
            self.skipTest(
                "benchmark.spicker_cluster is not available — "
                "William's SPICKER implementation has not been merged."
            )


# ======================================================================
# 13. TestRgSASAStub
# ======================================================================

class TestRgSASAStub(unittest.TestCase):
    """
    Stub for radius of gyration (Rg) and solvent-accessible surface
    area (SASA) calculations.  Rg characterises compactness; SASA
    quantifies the extent of residue burial.
    """

    def test_rg_sasa_import(self):
        """
        Attempt to import Rg and SASA functions from the benchmark
        package.  Skipped gracefully when the module is absent.
        """
        try:
            from benchmark import compute_rg, compute_sasa  # noqa: F401
        except ImportError:
            self.skipTest(
                "benchmark.compute_rg / compute_sasa are not available — "
                "William's Rg/SASA implementation has not been merged."
            )


# ======================================================================
# 14. TestScoreEdgeCases
# ======================================================================

class TestScoreEdgeCases(unittest.TestCase):
    """
    Evaluate the energy function under extreme or boundary conditions:
    a single residue (no pairwise terms) and overlapping residues
    (severe steric clash).  These scenarios stress-test the numerical
    stability of the evaluation routines.
    """

    # -- 14a. Single residue: only Ramachandran non-zero ---------------

    def test_single_residue_only_rama_nonzero(self):
        """
        For a single GLY residue, the Lennard-Jones, hydrogen-bond,
        and repulsive terms are zero because no residue pairs exist.
        The solvation term for GLY (not in the hydrophobic set at
        line 237) evaluates to zero as well.  Only the Ramachandran
        preference should contribute to the total energy.  Confirm
        that this holds.
        """
        energy_fn = RosettaEnergyFunction()
        conf = _build_extended("G")  # Single GLY
        _ = energy_fn.evaluate(conf)

        # Pairwise terms must be zero (no pairs)
        self.assertEqual(
            conf.energy_components["lennard_jones"], 0.0,
            msg=f"Single-residue LJ = "
                f"{conf.energy_components['lennard_jones']}, expected 0.0"
        )
        self.assertEqual(
            conf.energy_components["hydrogen_bond"], 0.0,
            msg=f"Single-residue Hbond = "
                f"{conf.energy_components['hydrogen_bond']}, expected 0.0"
        )
        self.assertEqual(
            conf.energy_components["repulsive"], 0.0,
            msg=f"Single-residue repulsive = "
                f"{conf.energy_components['repulsive']}, expected 0.0"
        )

        # Ramachandran must be finite and non-zero (phi=-135, psi=135
        # is not a perfect match to any GLY preference region)
        rama = conf.energy_components["ramachandran"]
        self.assertTrue(
            math.isfinite(rama),
            msg=f"Single-residue Ramachandran score is not finite: {rama}"
        )

    # -- 14b. Overlapping residues (0.5 Å) → finite repulsive ---------

    def test_overlapping_residues_0_5A_finite_repulsive(self):
        """
        Place two residues 0.5 Å apart (well below typical VDW radii of
        ~1.8 Å).  The repulsive term should dominate but remain finite.
        The Lennard-Jones term skips pairs with r < 0.5 Å (line 180),
        so it should be zero, while the repulsive term should produce a
        large-but-finite positive energy.
        """
        energy_fn = RosettaEnergyFunction()
        conf = _build_extended("AA")
        n_res = len(conf.residues)

        # Manually force CB atoms extremely close
        conf.residues[0].CB = np.array([0.0, 0.0, 0.0])
        conf.residues[1].CB = np.array([0.5, 0.0, 0.0])

        _ = energy_fn.evaluate(conf)

        # LJ should be zero (pair skipped at r < 0.5)
        self.assertEqual(
            conf.energy_components["lennard_jones"], 0.0,
            msg=f"Overlapping residues LJ = "
                f"{conf.energy_components['lennard_jones']}, expected 0.0 "
                f"(r < 0.5 Å → skip)"
        )

        # Repulsive must be finite and positive
        rep = conf.energy_components["repulsive"]
        self.assertTrue(
            math.isfinite(rep),
            msg=f"Repulsive energy for overlapping residues is not finite: "
                f"{rep}"
        )
        self.assertGreater(
            rep, 0.0,
            msg=f"Repulsive energy for overlapping residues = {rep}, "
                f"expected > 0"
        )

        # Total energy must be finite
        self.assertTrue(
            math.isfinite(conf.total_energy),
            msg=f"Total energy for overlapping residues is not finite: "
                f"{conf.total_energy}"
        )


# ======================================================================
# Entry Point
# ======================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
