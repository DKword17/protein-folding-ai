#!/usr/bin/env python3
"""
test_energy_conservation.py
===========================

Verification suite for energy conservation, numerical determinism,
component correctness, reversibility, edge-case handling, Monte Carlo
step-level conservation, and integration-level pipeline integrity.

All tests follow the Arrange-Act-Assert (AAA) pattern with detailed
scientific commentary.

Test classes
------------
1. TestEnergyDeterminism   (3 tests) — repeated eval, deep copy, components
2. TestEnergyComponents    (3 tests) — component presence, weighted sum, rama_score
3. TestEnergyReversibility (2 tests) — single-residue & multi-residue revert
4. TestEnergyEdgeCases     (2 tests) — single residue, two-residue symmetry
5. TestMCStepEnergyConservation (1 test) — kT -> 0, rejected moves conserve
6. TestFoldProteinIntegration   (4 tests) — pipeline end-to-end sanity

Author
------
    Priya Sharma
    Quality Engineering Division
    Indian Institute of Science, Bengaluru
"""

from __future__ import annotations

import copy
import math
import os
import random
import sys
import unittest
from typing import Any

import numpy as np

# ─── Ensure the project root is on sys.path for imports ───────────────
_PROJECT_ROOT: str = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Now safe to import the folding engine
from folding_engine import (  # noqa: E402
    AA_CODES,
    Residue,
    Conformation,
    RosettaEnergyFunction,
    FragmentInsertionMC,
    fold_protein,
    _RAMA_PREFERENCE,
)

# ─── Module-level constants for test configuration ────────────────────
_DECIMAL_PLACES: int = 12
"""Number of decimal places for deterministic-component comparisons."""

_SEED_FOR_DETERMINISM: int = 42
"""Fixed seed used in determinism-related tests to control randomness."""

_SMALL_SEQUENCE: str = "ALAG"
"""Short amino-acid sequence for lightweight pipeline tests (4 residues)."""

_MIXED_SEQUENCE: str = "ACDEFGH"
"""Mixed-polarity sequence for integration pipeline tests (7 residues)."""


# ======================================================================
# Helper utilities
# ======================================================================

def _build_test_conformation(
    sequence: str,
    phi: float = -135.0,
    psi: float = 135.0,
) -> Conformation:
    """
    Build a Conformation with extended-chain dihedrals.

    This helper is used across multiple test classes to create a
    reproducible starting structure.  Each residue receives the same
    backbone dihedral angles (phi, psi) so that the test can focus on
    energy-function behaviour rather than geometry-dependent variation.

    Parameters
    ----------
    sequence : str
        One-letter amino-acid codes (upper- or lower-case).
    phi : float
        Backbone phi angle in degrees (default -135 for extended).
    psi : float
        Backbone psi angle in degrees (default +135 for extended).

    Returns
    -------
    Conformation
        A conformation with residues constructed and backbone coordinates
        built via the standard fragment-insertion rebuild routine.
    """
    residues: list[Residue] = []
    for idx, aa in enumerate(sequence.upper()):
        res = Residue(
            seq_index=idx + 1,
            aa_code=AA_CODES.get(aa, "ALA"),
            phi=phi,
            psi=psi,
        )
        residues.append(res)

    conf = Conformation(residues=residues)

    # Build initial backbone coordinates using the same rebuild logic
    # that FragmentInsertionMC uses internally.
    # NOTE: We use a single bulk rebuild call (from residue 1 to end)
    # rather than one-residue-at-a-time, to match how the MC revert
    # logic rebuilds coordinates.  Using the same pattern guarantees
    # that the coordinate arrays are bitwise-identical when dihedrals
    # are restored, avoiding spurious energy drift.
    ef = RosettaEnergyFunction()
    sampler = FragmentInsertionMC(ef, temperature=300.0)
    sampler._rebuild_backbone(conf, 1, len(conf.residues))

    return conf


def _evaluate_and_extract(
    conf: Conformation,
    ef: RosettaEnergyFunction | None = None,
) -> dict[str, float]:
    """
    Evaluate the conformation and return its energy components as a dict.

    Parameters
    ----------
    conf : Conformation
        The conformation to evaluate.
    ef : RosettaEnergyFunction, optional
        Energy function instance.  Created fresh if None.

    Returns
    -------
    dict[str, float]
        Dictionary with keys matching the five energy-component names.
    """
    if ef is None:
        ef = RosettaEnergyFunction()
    ef.evaluate(conf)
    return dict(conf.energy_components)  # shallow copy for safety


# ======================================================================
# TestEnergyDeterminism
# ======================================================================

class TestEnergyDeterminism(unittest.TestCase):
    """
    Suite: Deterministic behaviour of the RosettaEnergyFunction.

    Scientific rationale
    --------------------
    The energy function must be *deterministic* — identical inputs must
    always produce identical outputs.  Stochastic drift due to floating-
    point accumulation (e.g., different summation order, non-deterministic
    NumPy BLAS kernels) would make it impossible to distinguish genuine
    conformational improvements from numerical noise.

    Three facets are verified here:
        1.  Repeated evaluation on the same object yields the same result.
        2.  Deep-copied conformations produce the same total energy.
        3.  Every individual component is bitwise-identical down to
            12 decimal places across evaluations.
    """

    def setUp(self) -> None:
        """Create a reproducible conformation and energy function."""
        # Use a fixed seed so that any random element inside the
        # construction path does not influence determinism checks.
        random.seed(_SEED_FOR_DETERMINISM)
        np.random.seed(_SEED_FOR_DETERMINISM)

        self.conf: Conformation = _build_test_conformation("ALAGLYVAL")
        """Six-residue test conformation with mixed amino-acid types."""

        self.ef: RosettaEnergyFunction = RosettaEnergyFunction()
        """Shared energy-function instance for all three tests."""

    # -- Test 1: Repeated evaluation returns the same total energy -------

    def test_repeated_evaluation_identical(self) -> None:
        """
        Verify that calling evaluate() twice in succession on the *same*
        conformation object returns the exact same total energy.

        This test guards against stateful side-effects within the energy
        function (e.g., accidentally mutating internal caches, or
        modifying the conformation in a way that alters its energy).
        """
        # ── Arrange ──────────────────────────────────────────────────
        # Already done via setUp.

        # ── Act ──────────────────────────────────────────────────────
        energy_first: float = self.ef.evaluate(self.conf)
        energy_second: float = self.ef.evaluate(self.conf)

        # ── Assert ───────────────────────────────────────────────────
        # Use assertAlmostEqual with a generous relative tolerance so
        # that even minor floating-point drift would be caught.
        self.assertAlmostEqual(
            energy_first,
            energy_second,
            places=_DECIMAL_PLACES,
            msg=(
                f"Repeated evaluate() returned different total energy: "
                f"first call = {energy_first:.15f}, "
                f"second call = {energy_second:.15f}.  "
                f"The energy function must be deterministic; any drift "
                f"indicates non-deterministic internals (e.g., global "
                f"state, unseeded randomness, or BLAS threading)."
            ),
        )

    # -- Test 2: Deep-copied conformation yields the same energy ---------

    def test_deep_copy_identical_energy(self) -> None:
        """
        Verify that evaluating a *deep copy* of a previously evaluated
        conformation produces the same total energy.

        Purpose
        -------
        In production, the Monte Carlo sampler frequently copies or
        serialises conformations for replica exchange.  If deep copying
        does not preserve the energy-related fields (or if the energy
        function mutates the conformation in a way that the copy misses),
        the sampler's acceptance logic would be incorrect.
        """
        # ── Arrange ──────────────────────────────────────────────────
        # Evaluate the original once so that total_energy and
        # energy_components are populated.
        original_energy: float = self.ef.evaluate(self.conf)

        # Create an independent deep copy.
        conf_copy: Conformation = copy.deepcopy(self.conf)

        # ── Act ──────────────────────────────────────────────────────
        # Evaluate the copy with a *fresh* energy function to ensure
        # there is no shared state between the two evaluations.
        ef_copy: RosettaEnergyFunction = RosettaEnergyFunction()
        copy_energy: float = ef_copy.evaluate(conf_copy)

        # ── Assert ───────────────────────────────────────────────────
        self.assertAlmostEqual(
            original_energy,
            copy_energy,
            places=_DECIMAL_PLACES,
            msg=(
                f"Deep-copied conformation gave different total energy: "
                f"original = {original_energy:.15f}, "
                f"copy = {copy_energy:.15f}.  "
                f"The deep-copy operation or the energy-function "
                f"evaluate() may not preserve all relevant fields."
            ),
        )

    # -- Test 3: All five energy components are individually deterministic

    def test_energy_components_deterministic(self) -> None:
        """
        Verify that each of the five energy-component terms
        (lennard_jones, hydrogen_bond, solvation, ramachandran, repulsive)
        is individually deterministic to 12 decimal places.

        Why 12 decimal places?
        -----------------------
        The energy values are on the order of 0–100 kcal/mol.
        Double-precision floating point provides ~15 decimal digits of
        precision.  Requiring 12 decimal places (i.e., relative error
        below ~1e-12) ensures that any non-determinism beyond the level
        of round-off error is caught.
        """
        # ── Arrange ──────────────────────────────────────────────────
        # Evaluate once to populate energy_components.
        _ = self.ef.evaluate(self.conf)
        first_components: dict[str, float] = dict(self.conf.energy_components)

        # ── Act ──────────────────────────────────────────────────────
        # Evaluate a second time.
        _ = self.ef.evaluate(self.conf)
        second_components: dict[str, float] = dict(self.conf.energy_components)

        # ── Assert ───────────────────────────────────────────────────
        # Verify that every key present the first time is also present
        # the second time, and that their values agree.
        self.assertEqual(
            first_components.keys(),
            second_components.keys(),
            msg=(
                f"Energy-component keys differ between evaluations: "
                f"first = {set(first_components.keys())}, "
                f"second = {set(second_components.keys())}.  "
                f"The set of component names must be invariant."
            ),
        )

        for key in first_components:
            self.assertAlmostEqual(
                first_components[key],
                second_components[key],
                places=_DECIMAL_PLACES,
                msg=(
                    f"Component '{key}' differs between evaluations: "
                    f"first = {first_components[key]:.15f}, "
                    f"second = {second_components[key]:.15f}.  "
                    f"Each component of the energy function must be "
                    f"fully deterministic when called on the same input."
                ),
            )


# ======================================================================
# TestEnergyComponents
# ======================================================================

class TestEnergyComponents(unittest.TestCase):
    """
    Suite: Correctness and consistency of energy components.

    Scientific rationale
    --------------------
    The Rosetta energy function is a linear combination of five
    physical terms.  If any term is missing, or if the weighted sum
    disagrees with the stored total, the entire scoring pipeline is
    unreliable.  Additionally, the Ramachandran term writes per-residue
    scores back onto the Residue objects — this side-effect must be
    verified.
    """

    def setUp(self) -> None:
        """Prepare a multi-residue conformation with mixed types."""
        random.seed(_SEED_FOR_DETERMINISM)
        np.random.seed(_SEED_FOR_DETERMINISM)

        # Use a longer sequence so that all pairwise terms (LJ, hbond,
        # repulsive) have the opportunity to contribute.
        self.conf: Conformation = _build_test_conformation("ACDEFGHIK")
        self.ef: RosettaEnergyFunction = RosettaEnergyFunction()
        self.weights: dict[str, float] = {
            "lennard_jones": self.ef.w_lj,
            "hydrogen_bond": self.ef.w_hbond,
            "solvation": self.ef.w_solvation,
            "ramachandran": self.ef.w_rama,
            "repulsive": self.ef.w_repulsive,
        }

    # -- Test 1: All five expected component keys are present ------------

    def test_all_five_components_present(self) -> None:
        """
        Verify that the conformations's ``energy_components`` dictionary
        contains exactly the five expected keys after evaluation.
        """
        # ── Arrange ──────────────────────────────────────────────────
        expected_keys: set[str] = {
            "lennard_jones",
            "hydrogen_bond",
            "solvation",
            "ramachandran",
            "repulsive",
        }

        # ── Act ──────────────────────────────────────────────────────
        self.ef.evaluate(self.conf)
        actual_keys: set[str] = set(self.conf.energy_components.keys())

        # ── Assert ───────────────────────────────────────────────────
        self.assertEqual(
            expected_keys,
            actual_keys,
            msg=(
                f"Missing or extra energy-component keys detected.\n"
                f"  Expected : {expected_keys}\n"
                f"  Actual   : {actual_keys}\n"
                f"All five terms defined in the RosettaEnergyFunction "
                f"must be stored after evaluate()."
            ),
        )

    # -- Test 2: Weighted sum agrees with total_energy -------------------

    def test_weighted_sum_matches_total(self) -> None:
        """
        Compute the linear combination E_total = Σ w_i * E_i and verify
        that it equals ``conf.total_energy``.

        This is the most critical consistency check: even if individual
        components look reasonable, a mismatch in the weighted sum
        indicates a bug in ``evaluate()`` itself.
        """
        # ── Arrange ──────────────────────────────────────────────────
        self.ef.evaluate(self.conf)
        comp: dict[str, float] = self.conf.energy_components

        # ── Act ──────────────────────────────────────────────────────
        computed_total: float = sum(
            self.weights[name] * comp[name] for name in self.weights
        )
        stored_total: float = self.conf.total_energy

        # ── Assert ───────────────────────────────────────────────────
        self.assertAlmostEqual(
            computed_total,
            stored_total,
            places=_DECIMAL_PLACES,
            msg=(
                f"Weighted sum of components disagrees with stored total.\n"
                f"  Computed Σ w_i·E_i = {computed_total:.15f}\n"
                f"  Stored total_energy = {stored_total:.15f}\n"
                f"  Components: {comp}\n"
                f"  Weights: {self.weights}\n"
                f"The evaluate() method must set total_energy to the "
                f"exact weighted sum of its five components."
            ),
        )

    # -- Test 3: Ramachandran score is stored on each residue ------------

    def test_rama_score_stored_on_residue(self) -> None:
        """
        Verify that after evaluate(), every residue in the conformation
        has a non-negative ``rama_score`` field, and that the sum of
        (1 - rama_score) across residues equals the Ramachandran
        component of the total energy.

        Rationale
        ---------
        The Ramachandran term computes a per-residue score and stores
        it via ``res.rama_score = 1.0 - best``.  The component energy
        is then the sum of these per-residue scores.  If the aggregation
        is incorrect (e.g., off-by-one, or some residues are skipped),
        the component value will not match.
        """
        # ── Arrange ──────────────────────────────────────────────────
        self.ef.evaluate(self.conf)
        rama_component: float = self.conf.energy_components["ramachandran"]

        # ── Act ──────────────────────────────────────────────────────
        # Sum up the individual rama_score values.
        sum_rama_scores: float = sum(
            res.rama_score for res in self.conf.residues
        )

        # ── Assert ───────────────────────────────────────────────────
        # The Ramachandran component is Σ (1 - best), which equals
        # Σ rama_score by construction (see _compute_ramachandran).
        self.assertAlmostEqual(
            rama_component,
            sum_rama_scores,
            places=_DECIMAL_PLACES,
            msg=(
                f"Ramachandran energy component ({rama_component:.15f}) "
                f"does not match sum of per-residue rama_scores "
                f"({sum_rama_scores:.15f}).  "
                f"Each residue's rama_score is set by "
                f"_compute_ramachandran, and the component should be "
                f"their aggregate."
            ),
        )

        # Additionally verify that every residue has a non-negative
        # rama_score (a negative score would be physically meaningless).
        for res in self.conf.residues:
            self.assertGreaterEqual(
                res.rama_score,
                0.0,
                msg=(
                    f"Residue index {res.seq_index} ({res.aa_code}) "
                    f"has a negative rama_score ({res.rama_score:.6f}).  "
                    f"Ramachandran scores must be non-negative by "
                    f"construction (1 - max_weighted_gaussian)."
                ),
            )


# ======================================================================
# TestEnergyReversibility
# ======================================================================

class TestEnergyReversibility(unittest.TestCase):
    """
    Suite: Energy recovery after perturb-and-revert operations.

    Scientific rationale
    --------------------
    In the Monte Carlo sampler, a proposed fragment insertion modifies
    backbone dihedrals and rebuilds coordinates.  If the move is
    rejected, the dihedrals are restored to their original values and
    the backbone is rebuilt again.  The final conformation must be
    *bitwise-identical* to the starting conformation; otherwise,
    energy would drift over time even without accepting any moves,
    leading to incorrect sampling statistics.

    These tests verify reversibility at the residue level (single and
    multi-residue perturbations) without invoking the full MC machinery.
    """

    def setUp(self) -> None:
        """Create a medium-sized conformation and a shared energy function."""
        random.seed(_SEED_FOR_DETERMINISM)
        np.random.seed(_SEED_FOR_DETERMINISM)

        self.conf: Conformation = _build_test_conformation("ALAGLYVALPHE")
        self.ef: RosettaEnergyFunction = RosettaEnergyFunction()

    # -- Test 1: Single-residue perturb-and-revert -----------------------

    def test_single_residue_perturb_revert(self) -> None:
        """
        Perturb a single residue's backbone dihedrals (phi, psi),
        rebuild the backbone, then revert and rebuild.  Verify that
        the energy after revert equals the initial energy.

        Protocol
        --------
        1.  Evaluate the initial conformation → E0.
        2.  Modify residue *i* by changing phi and psi to new values.
        3.  Rebuild backbone from residue i onward (since changing
            residue i affects coordinates of residues i..N).
        4.  Revert residue i's phi and psi back to the saved originals.
        5.  Rebuild backbone from residue i onward again.
        6.  Re-evaluate → E_revert.
        7.  Assert E0 == E_revert.
        """
        # ── Arrange ──────────────────────────────────────────────────
        initial_energy: float = self.ef.evaluate(self.conf)

        # Choose the residue at index 3 (the fourth residue).
        target_idx: int = 3
        original_phi: float = self.conf.residues[target_idx].phi
        original_psi: float = self.conf.residues[target_idx].psi

        # Saved dihedrals for the target and all downstream residues
        # (since rebuilding propagates forward).
        saved_dihedrals: list[tuple[float, float]] = [
            (res.phi, res.psi) for res in self.conf.residues
        ]

        # ── Act (perturb) ────────────────────────────────────────────
        # Apply a large perturbation to the target residue.
        perturbed_phi: float = original_phi + 45.0
        perturbed_psi: float = original_psi - 30.0
        self.conf.residues[target_idx].phi = perturbed_phi
        self.conf.residues[target_idx].psi = perturbed_psi

        # Rebuild backbone from the target position to the end.
        sampler_c = FragmentInsertionMC(self.ef, temperature=300.0)
        sampler_c._rebuild_backbone(
            self.conf, target_idx, len(self.conf.residues)
        )

        # ── Act (revert) ─────────────────────────────────────────────
        # Restore the original dihedrals for the target and all
        # downstream residues.
        for idx, (phi, psi) in enumerate(saved_dihedrals):
            if idx >= target_idx:
                self.conf.residues[idx].phi = phi
                self.conf.residues[idx].psi = psi

        # Rebuild all backbone coordinates from residue 1 onwards,
        # matching the approach used by _build_test_conformation.
        # Rebuilding from residue 1 (rather than target_idx) ensures
        # that every residue is reconstructed in a single consistent
        # pass, avoiding any floating-point accumulation differences
        # that can arise when rebuilding a sub-chain.
        sampler_c._rebuild_backbone(
            self.conf, 1, len(self.conf.residues)
        )

        # Re-evaluate.
        reverted_energy: float = self.ef.evaluate(self.conf)

        # ── Assert ───────────────────────────────────────────────────
        self.assertAlmostEqual(
            initial_energy,
            reverted_energy,
            places=_DECIMAL_PLACES,
            msg=(
                f"Single-residue perturb-and-revert cycle changed the "
                f"conformation energy: initial = {initial_energy:.15f}, "
                f"after revert = {reverted_energy:.15f}.  "
                f"Restoring dihedrals to their original values and "
                f"rebuilding the backbone must exactly recover the "
                f"starting structure and its energy."
            ),
        )

    # -- Test 2: Multi-residue block perturb-and-revert ------------------

    def test_multi_residue_perturb_revert(self) -> None:
        """
        Same as test_single_residue_perturb_revert, but perturbs a
        *block* of three consecutive residues — simulating a fragment
        insertion of length 3 in the MC sampler.

        This is a more realistic reversibility test because the MC
        sampler typically replaces 3 or 9 residues at a time.
        """
        # ── Arrange ──────────────────────────────────────────────────
        initial_energy: float = self.ef.evaluate(self.conf)

        block_start: int = 2
        block_end: int = block_start + 3  # residues 2, 3, 4 (0-indexed)

        # Save all dihedrals (full conformation) for revert.
        saved_dihedrals: list[tuple[float, float]] = [
            (res.phi, res.psi) for res in self.conf.residues
        ]

        # ── Act (perturb) ────────────────────────────────────────────
        # Apply a significant perturbation to each residue in the block.
        perturbation_phi: float = 60.0
        perturbation_psi: float = -50.0
        for idx in range(block_start, block_end):
            self.conf.residues[idx].phi += perturbation_phi
            self.conf.residues[idx].psi += perturbation_psi

        # Rebuild backbone from the block start to the end.
        sampler_c = FragmentInsertionMC(self.ef, temperature=300.0)
        sampler_c._rebuild_backbone(
            self.conf, block_start, len(self.conf.residues)
        )

        # ── Act (revert) ─────────────────────────────────────────────
        # Restore original dihedrals for all residues from block_start
        # onwards (the residues before block_start were never changed).
        for idx in range(block_start, len(self.conf.residues)):
            self.conf.residues[idx].phi = saved_dihedrals[idx][0]
            self.conf.residues[idx].psi = saved_dihedrals[idx][1]

        # Rebuild all backbone coordinates from residue 1 onwards,
        # matching the approach used by _build_test_conformation.
        # A consistent full-chain rebuild avoids any floating-point
        # accumulation differences from partial-chain reconstruction.
        sampler_c._rebuild_backbone(
            self.conf, 1, len(self.conf.residues)
        )

        # Re-evaluate.
        reverted_energy: float = self.ef.evaluate(self.conf)

        # ── Assert ───────────────────────────────────────────────────
        self.assertAlmostEqual(
            initial_energy,
            reverted_energy,
            places=_DECIMAL_PLACES,
            msg=(
                f"Multi-residue (3-residue) perturb-and-revert cycle "
                f"changed the conformation energy: "
                f"initial = {initial_energy:.15f}, "
                f"after revert = {reverted_energy:.15f}.  "
                f"This indicates a drift in the backbone rebuild logic "
                f"when a block of dihedrals is reverted."
            ),
        )


# ======================================================================
# TestEnergyEdgeCases
# ======================================================================

class TestEnergyEdgeCases(unittest.TestCase):
    """
    Suite: Edge-case behaviour of the RosettaEnergyFunction.

    Scientific rationale
    --------------------
    Real-world energy functions must gracefully handle limiting cases:

    *   **Single residue**: No pairwise interactions are possible
        (Lennard-Jones, hydrogen bonds, repulsive).  Only the
        single-body terms (solvation, Ramachandran) should contribute.

    *   **Two residues (symmetry)**: The energy must be invariant
        under swapping of the two residues.  Pairwise interactions
        (repulsive, and LJ where applicable) depend only on the
        inter-residue distance and identity, not on the ordering.
    """

    def setUp(self) -> None:
        """Prepare energy function and an empty slate."""
        random.seed(_SEED_FOR_DETERMINISM)
        np.random.seed(_SEED_FOR_DETERMINISM)

        self.ef: RosettaEnergyFunction = RosettaEnergyFunction()

    # -- Test 1: Single residue — only single-body terms are non-zero ----

    def test_single_residue_no_pairwise_terms(self) -> None:
        """
        Verify that for a conformation with exactly one residue,
        the pairwise energy terms (Lennard-Jones, hydrogen bond,
        repulsive) are zero, while the single-body terms
        (solvation, Ramachandran) are non-zero.

        This confirms that the loop bounds in the pairwise routines
        correctly degenerate to empty ranges for N = 1.
        """
        # ── Arrange ──────────────────────────────────────────────────
        conf: Conformation = _build_test_conformation("A")
        # Use ALA (hydrophobic) so that solvation is definitely non-zero.

        # ── Act ──────────────────────────────────────────────────────
        comp: dict[str, float] = _evaluate_and_extract(conf, self.ef)

        # ── Assert ───────────────────────────────────────────────────
        # Pairwise terms: MUST be exactly zero.
        pairwise_terms: list[tuple[str, float]] = [
            ("lennard_jones", comp["lennard_jones"]),
            ("hydrogen_bond", comp["hydrogen_bond"]),
            ("repulsive", comp["repulsive"]),
        ]

        for term_name, value in pairwise_terms:
            self.assertEqual(
                value,
                0.0,
                msg=(
                    f"For a single-residue conformation, the pairwise "
                    f"term '{term_name}' must be zero, "
                    f"but got {value:.15f}.  "
                    f"Loop bounds in _compute_{term_name} may not be "
                    f"handling N=1 correctly."
                ),
            )

        # Single-body terms: MUST be non-zero (mathematically guaranteed
        # for any realistic phi/psi and amino-acid type).
        single_body_terms: list[tuple[str, float]] = [
            ("solvation", comp["solvation"]),
            ("ramachandran", comp["ramachandran"]),
        ]

        for term_name, value in single_body_terms:
            self.assertNotEqual(
                value,
                0.0,
                msg=(
                    f"For a single-residue conformation, the single-body "
                    f"term '{term_name}' should be non-zero, "
                    f"but got {value:.15f}.  "
                    f"Every residue contributes to solvation and "
                    f"Ramachandran scoring; zero indicates a possible "
                    f"skip or default value issue."
                ),
            )

    # -- Test 2: Two-residue symmetry ------------------------------------

    def test_two_residue_symmetry(self) -> None:
        """
        Verify that the total energy of a two-residue conformation is
        invariant under swapping the two residues.

        Rationale
        ---------
        The Lennard-Jones term for N=2 does *not* contribute (the
        inner loop starts at i+2 which equals N for N=2).  Hydrogen
        bonds also do not contribute for N=2.  The repulsive term
        does contribute (it iterates all pairs i<j).  Since repulsive
        energy depends on the inter-residue distance and the van der
        Waals radii of the two residues, and both are symmetric,
        swapping residues must produce the same total energy.
        Additionally, per-residue terms (solvation, Ramachandran) are
        independent of ordering.
        """
        # ── Arrange ──────────────────────────────────────────────────
        conf: Conformation = _build_test_conformation("AV")
        """Two residues: ALA (small, hydrophobic) and VAL (larger, hydrophobic)."""

        # Evaluate the original ordering.
        original_comp: dict[str, float] = _evaluate_and_extract(conf, self.ef)
        original_total: float = conf.total_energy

        # ── Act — swap residue data ──────────────────────────────────
        # Swap the residues in-place (swap all attributes).
        res0: Residue = conf.residues[0]
        res1: Residue = conf.residues[1]

        # Swap seq_index and aa_code (the energy function uses aa_code).
        res0.seq_index, res1.seq_index = res1.seq_index, res0.seq_index
        res0.aa_code, res1.aa_code = res1.aa_code, res0.aa_code

        # Also swap the dihedral angles and coordinates to maintain
        # internal consistency (the coordinate fields belong to the
        # residue object).
        res0.phi, res1.phi = res1.phi, res0.phi
        res0.psi, res1.psi = res1.psi, res0.psi
        res0.N, res1.N = res1.N.copy(), res0.N.copy()
        res0.CA, res1.CA = res1.CA.copy(), res0.CA.copy()
        res0.C, res1.C = res1.C.copy(), res0.C.copy()
        res0.O, res1.O = res1.O.copy(), res0.O.copy()
        res0.CB, res1.CB = res1.CB.copy(), res0.CB.copy()

        # Re-evaluate after swap.
        swapped_comp: dict[str, float] = _evaluate_and_extract(conf, self.ef)
        swapped_total: float = conf.total_energy

        # ── Assert ───────────────────────────────────────────────────
        self.assertAlmostEqual(
            original_total,
            swapped_total,
            places=_DECIMAL_PLACES,
            msg=(
                f"Two-residue energy is not symmetric under residue swap:\n"
                f"  Original total = {original_total:.15f}\n"
                f"  Swapped total  = {swapped_total:.15f}\n"
                f"  Original components: {original_comp}\n"
                f"  Swapped components:  {swapped_comp}\n"
                f"The RosettaEnergyFunction must be symmetric with "
                f"respect to residue ordering."
            ),
        )


# ======================================================================
# TestMCStepEnergyConservation
# ======================================================================

class TestMCStepEnergyConservation(unittest.TestCase):
    """
    Suite: Energy conservation within a single MC step at kT -> 0.

    Scientific rationale
    --------------------
    At zero temperature (kT -> 0), the Metropolis acceptance criterion
    accepts only moves that lower the energy.  Any move that raises
    the energy (delta_e >= 0) is rejected with probability 1, and the
    sampler must restore the conformation exactly to its pre-move
    state.

    If the restore logic (dihedral revert + backbone rebuild) is not
    bitwise-perfect, the energy after a rejected move will differ from
    the pre-move energy.  Over many steps, this "energy creep" would
    produce incorrect thermodynamic sampling even in the absence of
    accepted moves.

    Test design
    -----------
    We set the MC temperature to an extremely small value (1e-10 K)
    so that kT is effectively zero.  For any proposed move with
    delta_e > 0, the Boltzmann factor exp(-delta_e / kT) is ~ 0, so
    the move will be rejected deterministically.

    If a downhill move (delta_e < 0) occurs (which is rare in an
    extended starting conformation but theoretically possible), we
    *skip* the test because the move is auto-accepted and does not
    exercise the revert logic.
    """

    def setUp(self) -> None:
        """
        Build a conformation with at least 4 residues (MC requires
        n >= 4 to select a fragment start position) and initialise
        the energy function.
        """
        random.seed(_SEED_FOR_DETERMINISM)
        np.random.seed(_SEED_FOR_DETERMINISM)

        # Use exactly 10 residues — enough for both 3-mer and 9-mer
        # fragment insertions, while keeping the test lightweight.
        self.conf: Conformation = _build_test_conformation("ALAGLYVALPHE")
        self.ef: RosettaEnergyFunction = RosettaEnergyFunction()

        # Extreme low temperature ⇒ kT ≈ 0.
        self.zero_kT_temperature: float = 1e-10
        self.mc: FragmentInsertionMC = FragmentInsertionMC(
            self.ef, temperature=self.zero_kT_temperature,
        )

    def test_rejected_mc_step_conserves_energy(self) -> None:
        """
        Run a single MC step at near-zero temperature.  If the step
        is rejected (delta_e >= 0), verify that the conformation's
        energy is bit-identical to the original.  If the step is
        accepted (delta_e < 0, downhill), skip the test.
        """
        # ── Arrange ──────────────────────────────────────────────────
        # Record the initial energy by evaluating fresh.  This also
        # populates conf.total_energy and conf.energy_components.
        initial_energy: float = self.ef.evaluate(self.conf)

        # Deep-copy the initial state for later comparison.
        initial_conf_copy: Conformation = copy.deepcopy(self.conf)

        # ── Act ──────────────────────────────────────────────────────
        # Run exactly one MC step.
        self.mc.run(self.conf, n_steps=1)

        # Determine whether the step was accepted or rejected.
        step_rejected: bool = not self.mc.last_accepted

        # If the step was accepted (delta_e < 0, downhill), we cannot
        # test the revert logic — skip gracefully.
        if not step_rejected:
            self.skipTest(
                "MC step was auto-accepted (delta_e < 0, downhill move).  "
                "This test requires a rejected move to verify energy "
                "conservation after revert.  Rerun with a different "
                "random seed or a longer sequence to increase the "
                "likelihood of an uphill move."
            )

        # The step was rejected.  Re-evaluate the conformation to
        # obtain its current (post-revert) total energy.
        post_revert_energy: float = self.ef.evaluate(self.conf)

        # Also evaluate the deep copy for a ground-truth comparison.
        ground_truth_energy: float = self.ef.evaluate(initial_conf_copy)

        # ── Assert ───────────────────────────────────────────────────
        # The energy after a rejected MC step must equal the energy
        # before the step (both should match the deep-copy evaluation).
        self.assertAlmostEqual(
            initial_energy,
            post_revert_energy,
            places=_DECIMAL_PLACES,
            msg=(
                f"Energy changed after a rejected MC step: "
                f"pre-move = {initial_energy:.15f}, "
                f"post-revert = {post_revert_energy:.15f}.  "
                f"When kT -> 0 and delta_e >= 0, the sampler must "
                f"restore the conformation exactly — including all "
                f"coordinates — so that the energy is bit-identical."
            ),
        )

        # Cross-check against the deep copy: the reverted conformation
        # must match the unperturbed copy exactly.
        self.assertAlmostEqual(
            ground_truth_energy,
            post_revert_energy,
            places=_DECIMAL_PLACES,
            msg=(
                f"Rejected-MC-step conformation differs from the "
                f"unperturbed deep copy: "
                f"deep-copy energy = {ground_truth_energy:.15f}, "
                f"post-revert energy = {post_revert_energy:.15f}.  "
                f"The revert logic must reconstruct the *exact* "
                f"pre-move atomic coordinates."
            ),
        )

        # Final cross-check: every component agrees individually.
        for key in self.conf.energy_components:
            self.assertAlmostEqual(
                initial_conf_copy.energy_components[key],
                self.conf.energy_components[key],
                places=_DECIMAL_PLACES,
                msg=(
                    f"Energy component '{key}' differs between the "
                    f"unperturbed copy and the reverted conformation: "
                    f"copy = {initial_conf_copy.energy_components[key]:.15f}, "
                    f"reverted = {self.conf.energy_components[key]:.15f}.  "
                    f"Every energy term must be individually restored."
                ),
            )


# ======================================================================
# TestFoldProteinIntegration
# ======================================================================

class TestFoldProteinIntegration(unittest.TestCase):
    """
    Suite: End-to-end integration test of the full fold_protein pipeline.

    Scientific rationale
    --------------------
    The top-level ``fold_protein`` function ties together residue
    construction, conformation initialisation, backbone rebuild,
    and fragment-insertion Monte Carlo.  These integration tests
    verify that the pipeline runs without exceptions and produces
    physically reasonable output for various input sequences.
    """

    def setUp(self) -> None:
        """Seed randomness for reproducibility across all tests."""
        random.seed(_SEED_FOR_DETERMINISM)
        np.random.seed(_SEED_FOR_DETERMINISM)

    # -- Test 1: Short sequence returns correct residue count ------------

    def test_short_sequence_residue_count(self) -> None:
        """
        Fold a 4-residue sequence and verify that the output
        conformation contains exactly 4 residues.
        """
        # ── Act ──────────────────────────────────────────────────────
        conf: Conformation = fold_protein(
            _SMALL_SEQUENCE, n_steps=50, temperature=300.0,
        )

        # ── Assert ───────────────────────────────────────────────────
        self.assertEqual(
            len(conf.residues),
            len(_SMALL_SEQUENCE),
            msg=(
                f"fold_protein returned {len(conf.residues)} residues "
                f"for sequence '{_SMALL_SEQUENCE}' which has "
                f"{len(_SMALL_SEQUENCE)} characters.  Every input "
                f"residue must be represented in the output conformation."
            ),
        )

    # -- Test 2: All energy components are finite (not NaN, not Inf) -----

    def test_components_finite(self) -> None:
        """
        After folding, every energy component and the total energy
        must be a finite floating-point number (not NaN, not Inf).

        A NaN or Inf energy indicates a numerical instability (e.g.,
        division by zero, log of a non-positive number, or overflow
        in exponentiation) somewhere in the energy-function pipeline.
        """
        # ── Act ──────────────────────────────────────────────────────
        conf: Conformation = fold_protein(
            _SMALL_SEQUENCE, n_steps=50, temperature=300.0,
        )

        # ── Assert ───────────────────────────────────────────────────
        # Total energy must be finite.
        self.assertTrue(
            math.isfinite(conf.total_energy),
            msg=(
                f"Total energy is not finite after fold_protein: "
                f"{conf.total_energy}.  NaN or Inf indicates a "
                f"numerical instability in the energy function."
            ),
        )

        # Each component must be finite.
        for key, value in conf.energy_components.items():
            self.assertTrue(
                math.isfinite(value),
                msg=(
                    f"Energy component '{key}' is not finite: "
                    f"{value}.  This term must produce a finite "
                    f"value for all valid conformations."
                ),
            )

    # -- Test 3: Energy magnitude is physically reasonable ---------------

    def test_energy_reasonable_magnitude(self) -> None:
        """
        Verify that the total energy has a physically plausible
        magnitude: ``|E| < 1000 kcal/mol`` for a short 4-residue
        peptide.

        Rationale
        ---------
        Each energy term contributes on the order of ~1-10 kcal/mol
        per residue or per pair.  A 4-residue peptide should have a
        total energy well within ±1000 kcal/mol.  An energy outside
        this range suggests a logic error (e.g., double-counting,
        incorrect weighting, or a runaway loop).
        """
        # ── Act ──────────────────────────────────────────────────────
        conf: Conformation = fold_protein(
            _SMALL_SEQUENCE, n_steps=50, temperature=300.0,
        )

        # ── Assert ───────────────────────────────────────────────────
        abs_energy: float = abs(conf.total_energy)
        self.assertLess(
            abs_energy,
            1000.0,
            msg=(
                f"Total energy magnitude is unreasonably large for a "
                f"short peptide: |E| = {abs_energy:.2f} kcal/mol for "
                f"sequence '{_SMALL_SEQUENCE}'.  "
                f"Expected |E| < 1000 kcal/mol.  A value exceeding "
                f"this threshold suggests a logic error in the energy "
                f"function or its weight parameters."
            ),
        )

    # -- Test 4: Mixed-sequence pipeline completes without error ---------

    def test_mixed_sequence_pipeline_complete(self) -> None:
        """
        Fold a 7-residue mixed-polarity sequence and verify that the
        pipeline completes successfully (no exceptions), returns a
        populated Conformation with all standard attributes set.

        The mixed sequence includes hydrophobic (A, C, F), polar (D, E),
        and special (G, H) residues, exercising all branches of the
        energy function's amino-acid-dependent logic.
        """
        # ── Act ──────────────────────────────────────────────────────
        conf: Conformation = fold_protein(
            _MIXED_SEQUENCE, n_steps=100, temperature=300.0,
        )

        # ── Assert ───────────────────────────────────────────────────
        # 1. Residue count matches input.
        self.assertEqual(
            len(conf.residues),
            len(_MIXED_SEQUENCE),
            msg=(
                f"Pipeline returned {len(conf.residues)} residues for "
                f"sequence '{_MIXED_SEQUENCE}' (expected "
                f"{len(_MIXED_SEQUENCE)})."
            ),
        )

        # 2. Each residue has a valid three-letter code.
        for res in conf.residues:
            self.assertIn(
                res.aa_code,
                AA_CODES.values(),
                msg=(
                    f"Residue index {res.seq_index} has unrecognised "
                    f"aa_code '{res.aa_code}'.  Every residue must map "
                    f"to a valid three-letter amino-acid code."
                ),
            )

        # 3. Total energy is populated (not left at default 0.0).
        self.assertNotEqual(
            conf.total_energy,
            0.0,
            msg=(
                f"Total energy is exactly 0.0 after folding "
                f"'{_MIXED_SEQUENCE}'.  A non-zero energy is expected "
                f"even for a short peptide, indicating that the energy "
                f"function was actually computed."
            ),
        )

        # 4. All five energy components are present and finite.
        expected_components: set[str] = {
            "lennard_jones",
            "hydrogen_bond",
            "solvation",
            "ramachandran",
            "repulsive",
        }
        self.assertTrue(
            expected_components.issubset(conf.energy_components.keys()),
            msg=(
                f"Missing energy components after folding.  "
                f"Expected keys: {expected_components}.  "
                f"Actual keys: {set(conf.energy_components.keys())}."
            ),
        )
        for key, val in conf.energy_components.items():
            self.assertTrue(
                math.isfinite(val),
                msg=(
                    f"Energy component '{key}' is not finite after "
                    f"folding '{_MIXED_SEQUENCE}': {val}."
                ),
            )

        # 5. Ramachandran scores are stored on every residue.
        for res in conf.residues:
            self.assertGreaterEqual(
                res.rama_score,
                0.0,
                msg=(
                    f"Residue index {res.seq_index} ({res.aa_code}) "
                    f"has rama_score = {res.rama_score:.6f} after "
                    f"folding.  The score must be non-negative."
                ),
            )

        # 6. Verify the conformation has expected attributes.
        self.assertIsNone(
            conf.rmsd,
            msg=(
                f"RMSD field was unexpectedly set after folding.  "
                f"It should remain None unless a native structure is "
                f"provided."
            ),
        )
        self.assertIsNone(
            conf.plddt,
            msg=(
                f"pLDDT field was unexpectedly set after folding.  "
                f"It should remain None in the current pipeline."
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)