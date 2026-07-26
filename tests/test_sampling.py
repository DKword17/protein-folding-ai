#!/usr/bin/env python3
"""
test_sampling.py
=================

Sampling convergence tests for replica exchange (t-REMD, H-REMD),
weighted histogram analysis (WHAM), and umbrella sampling.

The present test suite verifies the correctness of the following modules
developed by Jean-Luc Mercier (``dev/sampling-jeanluc``):

    1.  ``sampling/replica_exchange.py`` — Replica exchange Monte Carlo
        with temperature and Hamiltonian replica exchange (Sugita &
        Okamoto 1999, Chem Phys Lett 314:141; Fukunishi et al. 2002,
        J Chem Phys 116:9058).
    2.  ``sampling/wham.py`` — Weighted histogram analysis method
        (Kumar et al. 1992, J Comput Chem 13:1011) for extracting
        the potential of mean force from biased simulations.
    3.  ``sampling/umbrella_sampling.py`` — Umbrella sampling with
        harmonic biasing potentials (Torrie & Valleau 1977, J Comput
        Phys 23:187; Kästner 2011, WIREs Comput Mol Sci 1:932).

Every test follows the Arrange-Act-Assert (AAA) pattern with
detailed scientific commentary as per project convention.

Author: Priya Sharma, Quality Engineering, Indian Institute of Science, Bengaluru
"""

import math
import os
import tempfile
import unittest

import numpy as np

from folding_engine import Residue, Conformation, AA_CODES

# -----------------------------------------------------------------------
#  Graceful import handling
#
#  The sampling modules are being developed by Jean-Luc Mercier on
#  ``dev/sampling-jeanluc`` and may not yet be present on this branch.
#  Tests that depend on them will be skipped with an explicit message
#  if the import fails.
# -----------------------------------------------------------------------

try:
    from sampling.replica_exchange import (
        CONSTANTE_BOLTZMANN,
        températures_géométriques,
        Réplique,
        ÉchangeDeRépliques,
    )
    _HAS_REPLICA = True
except ImportError:
    _HAS_REPLICA = False

try:
    from sampling.wham import (
        Histogramme,
        WHAM,
    )
    _HAS_WHAM = True
except ImportError:
    _HAS_WHAM = False

try:
    from sampling.umbrella_sampling import (
        rmsd_naïf,
        rayon_gyration,
        nombre_contacts,
        Fenêtre,
        ÉchantillonnageParapluie,
    )
    _HAS_UMBRELLA = True
except ImportError:
    _HAS_UMBRELLA = False


# ======================================================================
#  Helper fixtures
# ======================================================================

def _conformation_ala(n_residues: int = 5) -> Conformation:
    """
    Construct a minimal poly-alanine conformation with extended
    backbone coordinates (phi = -135, psi = 135).

    Args:
        n_residues: Number of alanine residues (default 5).

    Returns:
        Conformation object with rebuilt backbone coordinates.
    """
    residues = [
        Residue(i + 1, "ALA", phi=-135.0, psi=135.0)
        for i in range(n_residues)
    ]
    conf = Conformation(residues=residues)

    # Rebuild backbone so coordinates are valid.
    from folding_engine import FragmentInsertionMC
    sampler = FragmentInsertionMC.__new__(FragmentInsertionMC)
    for i in range(1, n_residues):
        FragmentInsertionMC._rebuild_backbone(sampler, conf, i, i + 1)
    return conf


def _conformation_helix(n_residues: int = 5) -> Conformation:
    """
    Construct a poly-alanine conformation in a helical geometry
    (phi = -60, psi = -45).

    Args:
        n_residues: Number of alanine residues (default 5).

    Returns:
        Conformation object with helical backbone.
    """
    residues = [
        Residue(i + 1, "ALA", phi=-60.0, psi=-45.0)
        for i in range(n_residues)
    ]
    conf = Conformation(residues=residues)
    from folding_engine import FragmentInsertionMC
    sampler = FragmentInsertionMC.__new__(FragmentInsertionMC)
    for i in range(1, n_residues):
        FragmentInsertionMC._rebuild_backbone(sampler, conf, i, i + 1)
    return conf


# ======================================================================
#  1.  TestRéplique — Réplique data class fundamentals
# ======================================================================

@unittest.skipIf(not _HAS_REPLICA, "sampling.replica_exchange not available")
class TestRéplique(unittest.TestCase):
    """
    Validate the Réplique data class constructor and default values.

    References:
        - Sugita & Okamoto (1999) Chem Phys Lett 314:141
    """

    def test_beta_computed_correctly(self):
        """
        Kindly ensure that beta (= 1 / (kB * T)) is computed from
        température at construction time using CONSTANTE_BOLTZMANN.

        For T = 300 K: beta = 1 / (0.001987 * 300) ≈ 1.677.
        """
        # Arrange: create a Réplique at 300 K with a dummy conformation.
        conf = _conformation_ala()
        rep = Réplique(température=300.0, conformation=conf, énergie=0.0)

        # Act: retrieve the beta attribute.
        beta = rep.beta

        # Assert: beta = 1 / (kB * T).
        expected = 1.0 / (CONSTANTE_BOLTZMANN * 300.0)
        self.assertAlmostEqual(
            beta, expected, places=6,
            msg=f"Expected beta ≈ {expected:.6f} at T=300 K, "
                f"but got {beta:.6f}.  "
                f"Verify that beta = 1/(kB*T) is set in __init__."
        )

    def test_default_counters_are_zero(self):
        """
        Verify that acceptations and tentatives are initialised to zero
        when no arguments are supplied for those fields.
        """
        # Arrange.
        conf = _conformation_ala()
        rep = Réplique(température=300.0, conformation=conf, énergie=0.0)

        # Act & Assert.
        self.assertEqual(
            rep.acceptations, 0,
            msg="Newly created Réplique should have acceptations = 0, "
                f"but got {rep.acceptations}."
        )
        self.assertEqual(
            rep.tentatives, 0,
            msg="Newly created Réplique should have tentatives = 0, "
                f"but got {rep.tentatives}."
        )


# ======================================================================
#  2.  TestTempératuresGéométriques
# ======================================================================

@unittest.skipIf(not _HAS_REPLICA, "sampling.replica_exchange not available")
class TestTempératuresGéométriques(unittest.TestCase):
    """
    Verify the geometric temperature ladder generator.

    A geometric progression T_i = T_min * (T_max / T_min)^(i/(n-1))
    ensures uniform exchange probabilities (Earl & Deem 2005,
    Phys Chem Chem Phys 7:3910).
    """

    def test_returns_correct_length(self):
        """
        Kindly ensure that températures_géométriques returns exactly
        *n* temperatures.
        """
        # Arrange.
        t_min, t_max, n = 250.0, 500.0, 8

        # Act.
        temps = températures_géométriques(t_min, t_max, n)

        # Assert.
        self.assertEqual(
            len(temps), n,
            msg=f"Expected {n} temperatures from the geometric ladder, "
                f"but got {len(temps)}."
        )

    def test_ratio_is_constant(self):
        """
        Verify that the ratio T_{i+1} / T_i is constant across the
        ladder, which is the defining property of a geometric series.
        """
        # Arrange.
        t_min, t_max, n = 250.0, 500.0, 8
        temps = températures_géométriques(t_min, t_max, n)

        # Act: compute adjacent ratios.
        ratios = [temps[i + 1] / temps[i] for i in range(len(temps) - 1)]

        # Assert.
        for i, r in enumerate(ratios[1:], start=1):
            self.assertAlmostEqual(
                r, ratios[0], places=10,
                msg=f"Temperature ratio at position {i} is {r:.10f}, "
                    f"but the first ratio is {ratios[0]:.10f}.  "
                    f"All ratios must be equal for a geometric ladder."
            )

    def test_n_less_than_two_raises(self):
        """
        Verify that requesting fewer than 2 temperatures raises a
        ValueError, since a ladder must have at least two rungs.
        """
        # Arrange, Act & Assert.
        with self.assertRaises(ValueError) as ctx:
            températures_géométriques(250.0, 500.0, 1)

        self.assertIn(
            "n", str(ctx.exception).lower(),
            msg="Exception message should indicate that n < 2 is invalid.  "
                f"Got: {ctx.exception}"
        )


# ======================================================================
#  3.  TestRépliqueProbabilitéÉchange
# ======================================================================

@unittest.skipIf(not _HAS_REPLICA, "sampling.replica_exchange not available")
class TestRépliqueProbabilitéÉchange(unittest.TestCase):
    """
    Validate the Metropolis-Hastings exchange probability between
    two replicas, covering both temperature-REMD (t-REMD) and
    Hamiltonian-REMD (H-REMD) formalisms.

    References:
        - Sugita & Okamoto (1999) Chem Phys Lett 314:141 — t-REMD
        - Fukunishi et al. (2002) J Chem Phys 116:9058 — H-REMD
    """

    def test_t_remd_formula(self):
        """
        t-REMD exchange probability:
            P(i<->j) = min(1, exp((β_i - β_j)(E_j - E_i)))

        For T1=300 (β≈1.677), T2=350 (β≈1.437),
        E1=10, E2=20: Δβ = 0.240, ΔE = 10, product ≈ 2.40,
        so P = exp(2.40) ≈ 11.0 → clamped to 1.0.
        """
        # Arrange.
        conf = _conformation_ala()
        rep_i = Réplique(température=300.0, conformation=conf, énergie=10.0)
        rep_j = Réplique(température=350.0, conformation=conf, énergie=20.0)

        # Act.
        prob = rep_i.probabilité_échange(rep_j)

        # Assert.
        expected = math.exp(
            (rep_i.beta - rep_j.beta) * (rep_j.énergie - rep_i.énergie)
        )
        expected_clamped = min(1.0, expected)
        self.assertAlmostEqual(
            prob, expected_clamped, places=10,
            msg=f"t-REMD probability between T=300 (E=10) and T=350 (E=20) "
                f"should be {expected_clamped:.6f}, but got {prob:.6f}.  "
                f"Check that probabilité_échange uses "
                f"min(1, exp((β_i - β_j)(E_j - E_i)))."
        )

    def test_h_remd_formula(self):
        """
        H-REMD exchange probability substitutes the Hamiltonian
        weight into the acceptance criterion:
            P = min(1, exp(β*(E_i - E_j) - (β*E_i' - β*E_j')))

        Here we test with poids_hamiltoniens active by setting
        rep_i.énergie and rep_j.énergie along with stored
        hamiltonian-weighted energies.
        """
        # Arrange.
        conf = _conformation_ala()
        rep_i = Réplique(température=300.0, conformation=conf, énergie=10.0)
        rep_j = Réplique(température=300.0, conformation=conf, énergie=20.0)
        # Simulate H-REMD: store a different set of weighted energies.
        rep_i.énergie_repulsive = 5.0   # E' under alternative Hamiltonian
        rep_j.énergie_repulsive = 15.0

        # Act: with equal temperatures, the t-REMD term vanishes;
        # only the Hamiltonian difference contributes.
        prob = rep_i.probabilité_échange(rep_j)

        # Assert: for H-REMD with same T,
        # P = min(1, exp(β*(E_i + E_j' - E_j - E_i')))
        #   = min(1, exp(β*(10 + 15 - 20 - 5)))
        #   = min(1, exp(β * 0))
        #   = 1.0
        self.assertAlmostEqual(
            prob, 1.0, places=10,
            msg=f"With equal temperatures and E_i+E_j' = E_j+E_i', "
                f"the H-REMD probability should equal 1.0, "
                f"but got {prob:.6f}."
        )

    def test_vers_dictionnaire_returns_expected_keys(self):
        """
        Verify that vers_dictionnaire() serialises the Réplique
        into a dict with all essential fields.
        """
        # Arrange.
        conf = _conformation_ala()
        rep = Réplique(
            température=300.0, conformation=conf, énergie=10.0,
            énergie_repulsive=1.0,
        )

        # Act.
        d = rep.vers_dictionnaire()

        # Assert.
        expected_keys = {
            "température", "énergie", "énergie_repulsive",
            "beta", "poids", "acceptations", "tentatives",
        }
        self.assertTrue(
            expected_keys.issubset(d.keys()),
            msg=f"Dictionary from vers_dictionnaire() is missing keys.  "
                f"Expected at least {expected_keys}, got {set(d.keys())}."
        )
        self.assertEqual(
            d["température"], 300.0,
            msg="Dictionary should contain température=300.0."
        )
        self.assertEqual(
            d["énergie"], 10.0,
            msg="Dictionary should contain énergie=10.0."
        )


# ======================================================================
#  4.  TestREMDInitialization
# ======================================================================

@unittest.skipIf(not _HAS_REPLICA, "sampling.replica_exchange not available")
class TestREMDInitialization(unittest.TestCase):
    """
    Verify that ÉchangeDeRépliques.initialiser creates the correct
    number of replicas, rejects invalid inputs, and maintains internal
    state.
    """

    def test_correct_number_of_replicas(self):
        """
        Kindly ensure that the number of replicas created equals the
        length of the temperature list.
        """
        # Arrange.
        temps = [300.0, 320.0, 350.0, 400.0]
        ech = ÉchangeDeRépliques(temps=temps, fonction_énergie=lambda c: 0.0)

        # Act.
        ech.initialiser("AAA")
        n_replicas = len(ech.répliques)

        # Assert.
        self.assertEqual(
            n_replicas, len(temps),
            msg=f"initialiser should create {len(temps)} replicas "
                f"(one per temperature), but created {n_replicas}."
        )

    def test_less_than_two_temperatures_raises(self):
        """
        REMD requires at least two replicas (temperatures) for any
        exchange to be possible.
        """
        # Arrange.
        ech = ÉchangeDeRépliques(
            temps=[300.0], fonction_énergie=lambda c: 0.0
        )

        # Act & Assert.
        with self.assertRaises(ValueError):
            ech.initialiser("AAA")

    def test_replicas_sorted_by_temperature(self):
        """
        Verify that the internal replica list is sorted in ascending
        order of température, which is required for the neighbour
        exchange protocol.
        """
        # Arrange.
        temps = [400.0, 300.0, 350.0]  # deliberately unsorted
        ech = ÉchangeDeRépliques(temps=temps, fonction_énergie=lambda c: 0.0)

        # Act.
        ech.initialiser("AAA")
        sorted_temps = [r.température for r in ech.répliques]

        # Assert.
        self.assertEqual(
            sorted_temps, sorted(temps),
            msg=f"Replicas should be sorted by température.  "
                f"Got: {sorted_temps}, expected: {sorted(temps)}."
        )

    def test_historique_initialized_empty(self):
        """
        The historique list (energy trajectory for each replica)
        should be an empty list after initialisation.
        """
        # Arrange.
        temps = [300.0, 350.0]
        ech = ÉchangeDeRépliques(temps=temps, fonction_énergie=lambda c: 0.0)

        # Act.
        ech.initialiser("AAA")

        # Assert.
        self.assertEqual(
            len(ech.historique), 0,
            msg="Historique should be empty after initialisation "
                f"but has {len(ech.historique)} entries."
        )


# ======================================================================
#  5.  TestREMDLocalMCStep
# ======================================================================

@unittest.skipIf(not _HAS_REPLICA, "sampling.replica_exchange not available")
class TestREMDLocalMCStep(unittest.TestCase):
    """
    Validate the local Monte Carlo step that perturbs a single
    replica's conformation.
    """

    def setUp(self):
        self.temps = [300.0, 350.0]
        self.ech = ÉchangeDeRépliques(
            temps=self.temps, fonction_énergie=lambda c: 0.0,
        )
        self.ech.initialiser("AAAAA")

    def test_energy_remains_finite_after_step(self):
        """
        A local MC step must not produce NaN or infinite energies.
        """
        # Arrange.
        rep = self.ech.répliques[0]

        # Act.
        self.ech._pas_mc_local(rep, pas=1.0)

        # Assert.
        self.assertTrue(
            math.isfinite(rep.énergie),
            msg=f"Energy after local MC step is not finite: {rep.énergie}.  "
                f"Check that fragment perturbations stay within valid bounds."
        )

    def test_bond_lengths_preserved_after_step(self):
        """
        Backbone bond lengths should remain close to ideal values
        (within 0.05 Å) after a local MC move.
        """
        # Arrange.
        rep = self.ech.répliques[0]
        conf = rep.conformation

        # Act.
        self.ech._pas_mc_local(rep, pas=1.0)

        # Assert.
        for res in conf.residues[1:]:
            d_n_ca = np.linalg.norm(res.CA - res.N)
            self.assertAlmostEqual(
                d_n_ca, 1.47, delta=0.05,
                msg=f"Residue {res.seq_index}: N-CA distance = {d_n_ca:.4f} Å, "
                    f"deviates from ideal 1.47 Å after local MC step."
            )
            d_ca_c = np.linalg.norm(res.C - res.CA)
            self.assertAlmostEqual(
                d_ca_c, 1.51, delta=0.05,
                msg=f"Residue {res.seq_index}: CA-C distance = {d_ca_c:.4f} Å, "
                    f"deviates from ideal 1.51 Å after local MC step."
            )


# ======================================================================
#  6.  TestREMDPairwiseExchange
# ======================================================================

@unittest.skipIf(not _HAS_REPLICA, "sampling.replica_exchange not available")
class TestREMDPairwiseExchange(unittest.TestCase):
    """
    Verify the pairwise replica exchange step (tentative d'échange)
    and associated counter management.
    """

    def setUp(self):
        self.temps = [300.0, 350.0]
        self.ech = ÉchangeDeRépliques(
            temps=self.temps, fonction_énergie=lambda c: 0.0,
        )
        self.ech.initialiser("AAAAA")

    def test_exchange_does_not_raise(self):
        """
        A pairwise exchange between two valid replicas should
        complete without raising any exception.
        """
        # Arrange.
        i, j = 0, 1

        # Act & Assert.
        try:
            self.ech._tenter_échange(i, j)
        except Exception as exc:
            self.fail(
                f"_tenter_échange({i}, {j}) raised an unexpected exception: "
                f"{type(exc).__name__}: {exc}"
            )

    def test_energy_finite_after_swap(self):
        """
        After a successful exchange, both replicas should have
        finite energies.
        """
        # Arrange.
        i, j = 0, 1

        # Act.
        self.ech._tenter_échange(i, j)

        # Assert.
        for idx, rep in enumerate([self.ech.répliques[i], self.ech.répliques[j]]):
            self.assertTrue(
                math.isfinite(rep.énergie),
                msg=f"Replica {idx} has non-finite energy {rep.énergie} "
                    f"after exchange."
            )

    def test_counters_increment_on_exchange_attempt(self):
        """
        Every call to _tenter_échange should increment the tentatives
        counter for both replicas. If the exchange is accepted,
        acceptations should also increment.
        """
        # Arrange.
        i, j = 0, 1

        # Record counters before exchange.
        tent_before = (self.ech.répliques[i].tentatives,
                       self.ech.répliques[j].tentatives)
        acc_before = (self.ech.répliques[i].acceptations,
                      self.ech.répliques[j].acceptations)

        # Act.
        self.ech._tenter_échange(i, j)

        # Assert: tentatives must have increased.
        self.assertGreater(
            self.ech.répliques[i].tentatives, tent_before[0],
            msg=f"Replica {i} tentatives did not increment "
                f"({tent_before[0]} → {self.ech.répliques[i].tentatives})."
        )
        self.assertGreater(
            self.ech.répliques[j].tentatives, tent_before[1],
            msg=f"Replica {j} tentatives did not increment "
                f"({tent_before[1]} → {self.ech.répliques[j].tentatives})."
        )

        # acceptations may or may not increment (depends on Metropolis).
        # At minimum they should not decrease.
        self.assertGreaterEqual(
            self.ech.répliques[i].acceptations, acc_before[0],
            msg=f"Replica {i} acceptations decreased after exchange attempt."
        )
        self.assertGreaterEqual(
            self.ech.répliques[j].acceptations, acc_before[1],
            msg=f"Replica {j} acceptations decreased after exchange attempt."
        )


# ======================================================================
#  7.  TestREMDFullRun
# ======================================================================

@unittest.skipIf(not _HAS_REPLICA, "sampling.replica_exchange not available")
class TestREMDFullRun(unittest.TestCase):
    """
    Integration test for a full REMD simulation cycle.
    """

    def setUp(self):
        self.temps = [300.0, 350.0, 400.0]
        self.ech = ÉchangeDeRépliques(
            temps=self.temps, fonction_énergie=lambda c: 0.0,
            pas_mc=0.5,
        )

    def test_historique_populated_after_execution(self):
        """
        After executing the REMD loop, the historique list should
        contain one entry per iteration.
        """
        # Arrange.
        n_iters = 10

        # Act.
        self.ech.exécuter("AAAAA", iters=n_iters)

        # Assert.
        self.assertEqual(
            len(self.ech.historique), n_iters,
            msg=f"Historique should have {n_iters} entries "
                f"(one per iteration), but has {len(self.ech.historique)}."
        )

    def test_all_energies_are_finite(self):
        """
        Every recorded energy in historique should be finite.
        """
        # Arrange.
        n_iters = 10

        # Act.
        self.ech.exécuter("AAAAA", iters=n_iters)

        # Assert.
        for step_idx, entry in enumerate(self.ech.historique):
            for rep_idx, énergie in enumerate(entry):
                self.assertTrue(
                    math.isfinite(énergie),
                    msg=f"Non-finite energy at iteration {step_idx}, "
                        f"replica {rep_idx}: {énergie}."
                )

    def test_acceptance_rates_are_within_reasonable_range(self):
        """
        Acceptance rates across replicas should be between 0 and 1
        (inclusive).
        """
        # Arrange.
        n_iters = 20

        # Act.
        replicas = self.ech.exécuter("AAAAA", iters=n_iters)

        # Assert.
        for rep in replicas:
            rate = rep.acceptations / max(rep.tentatives, 1)
            self.assertGreaterEqual(
                rate, 0.0,
                msg=f"Acceptance rate for replica T={rep.température} "
                    f"is negative ({rate:.4f})."
            )
            self.assertLessEqual(
                rate, 1.0,
                msg=f"Acceptance rate for replica T={rep.température} "
                    f"exceeds 1.0 ({rate:.4f})."
            )

    def test_exchange_matrix_is_square(self):
        """
        The exchange probability matrix returned by exécuter should
        be square with dimension equal to the number of replicas.
        """
        # Arrange.
        n_iters = 10
        n_reps = len(self.temps)

        # Act.
        replicas = self.ech.exécuter("AAAAA", iters=n_iters)

        # Assert: if exécuter returns a matrix, check its shape.
        # Many implementations store a matrix internally; if not,
        # this test is a soft check.
        if hasattr(self.ech, "matrice_échange"):
            mat = self.ech.matrice_échange
            self.assertEqual(
                mat.shape, (n_reps, n_reps),
                msg=f"Exchange matrix shape should be ({n_reps}, {n_reps}), "
                    f"but got {mat.shape}."
            )

    def test_execute_returns_list_of_replicas(self):
        """
        exécuter should return a list of Réplique objects representing
        the final state.
        """
        # Arrange.
        n_iters = 10

        # Act.
        result = self.ech.exécuter("AAAAA", iters=n_iters)

        # Assert.
        self.assertIsInstance(
            result, list,
            msg=f"exécuter should return a list, got {type(result).__name__}."
        )
        self.assertEqual(
            len(result), len(self.temps),
            msg=f"Returned list length {len(result)} != number of replicas "
                f"{len(self.temps)}."
        )
        for rep in result:
            self.assertIsInstance(
                rep, Réplique,
                msg=f"Each element should be a Réplique, "
                    f"got {type(rep).__name__}."
            )


# ======================================================================
#  8.  TestWHAMHistogramme
# ======================================================================

@unittest.skipIf(not _HAS_WHAM, "sampling.wham not available")
class TestWHAMHistogramme(unittest.TestCase):
    """
    Verify the Histogramme data class used by WHAM.

    References:
        - Kumar et al. (1992) J Comput Chem 13:1011
    """

    def test_beta_computed_from_temperature(self):
        """
        Kindly ensure that beta = 1 / (kB * T) is automatically
        computed when a Histogramme is created.
        """
        # Arrange.
        bins = np.linspace(-10, 10, 21)
        comptes = np.zeros(20)

        # Act.
        h = Histogramme(température=300.0, bins=bins, comptes=comptes)

        # Assert.
        expected_beta = 1.0 / (0.001987 * 300.0)
        self.assertAlmostEqual(
            h.beta, expected_beta, places=6,
            msg=f"Expected beta ≈ {expected_beta:.6f} for T=300 K, "
                f"but got {h.beta:.6f}."
        )

    def test_sample_count_is_sum_of_comptes(self):
        """
        The nb_échantillons property should equal the sum of the
        histogram counts.
        """
        # Arrange.
        bins = np.linspace(-10, 10, 21)
        comptes = np.array([2, 0, 5, 3, 1, 0, 0, 4, 2, 0,
                            1, 3, 5, 0, 2, 4, 1, 0, 3, 2])

        # Act.
        h = Histogramme(température=300.0, bins=bins, comptes=comptes)

        # Assert.
        expected = int(np.sum(comptes))
        self.assertEqual(
            h.nb_échantillons, expected,
            msg=f"nb_échantillons should sum to {expected}, "
                f"but got {h.nb_échantillons}."
        )


# ======================================================================
#  9.  TestWHAMSelfConsistent
# ======================================================================

@unittest.skipIf(not _HAS_WHAM, "sampling.wham not available")
class TestWHAMSelfConsistent(unittest.TestCase):
    """
    Verify the WHAM self-consistent iteration solver.

    References:
        - Kumar et al. (1992) J Comput Chem 13:1011
        - Roux (1995) Comput Phys Commun 91:275
    """

    def test_empty_histogram_list_raises(self):
        """
        The WHAM solver should raise a ValueError when no histograms
        have been added.
        """
        # Arrange.
        wham = WHAM(tolérance=1e-6, itérations_max=100)

        # Act & Assert.
        with self.assertRaises(ValueError) as ctx:
            wham.résoudre()

        self.assertIn(
            "histogram", str(ctx.exception).lower(),
            msg=f"Error message should mention 'histogram' when no "
                f"histograms are provided. Got: {ctx.exception}"
        )

    def test_solver_converges_with_two_histograms(self):
        """
        With two histograms spanning an overlapping range, the
        self-consistent iteration should converge and return a
        tuple (Ω, f).
        """
        # Arrange.
        bins = np.linspace(-5.0, 5.0, 21)
        h1 = Histogramme(
            température=300.0, bins=bins,
            comptes=np.array([0, 0, 1, 2, 5, 8, 12, 15, 18, 20,
                              18, 15, 12, 8, 5, 2, 1, 0, 0, 0]),
            biais=0.0,
        )
        h2 = Histogramme(
            température=350.0, bins=bins,
            comptes=np.array([1, 2, 3, 5, 7, 10, 12, 14, 15, 16,
                              15, 14, 12, 10, 7, 5, 3, 2, 1, 0]),
            biais=0.0,
        )
        wham = WHAM(tolérance=1e-6, itérations_max=1000)
        wham.ajouter_histogramme(h1)
        wham.ajouter_histogramme(h2)

        # Act.
        omega, f = wham.résoudre()

        # Assert.
        self.assertIsNotNone(
            omega,
            msg="WHAM solver returned None for Ω (density of states)."
        )
        self.assertIsNotNone(
            f,
            msg="WHAM solver returned None for f (free energies)."
        )

    def test_returns_tuple_of_arrays(self):
        """
        résoudre() should return a tuple (Ω, f) where both elements
        are numpy arrays.
        """
        # Arrange.
        bins = np.linspace(-5.0, 5.0, 21)
        h1 = Histogramme(température=300.0, bins=bins,
                         comptes=np.ones(20))
        h2 = Histogramme(température=350.0, bins=bins,
                         comptes=np.ones(20))
        wham = WHAM(tolérance=1e-6, itérations_max=100)
        wham.ajouter_histogramme(h1)
        wham.ajouter_histogramme(h2)

        # Act.
        result = wham.résoudre()

        # Assert.
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        omega, f = result
        self.assertIsInstance(omega, np.ndarray)
        self.assertIsInstance(f, np.ndarray)

    def test_free_energy_profile_is_finite(self):
        """
        The free energy profile from profil_énergie_libre(T) should
        contain only finite values.
        """
        # Arrange.
        bins = np.linspace(-5.0, 5.0, 21)
        h1 = Histogramme(température=300.0, bins=bins,
                         comptes=np.ones(20))
        h2 = Histogramme(température=350.0, bins=bins,
                         comptes=np.ones(20))
        wham = WHAM(tolérance=1e-6, itérations_max=100)
        wham.ajouter_histogramme(h1)
        wham.ajouter_histogramme(h2)
        wham.résoudre()

        # Act.
        F_bins, F = wham.profil_énergie_libre(T=300.0)

        # Assert.
        for val in F:
            self.assertTrue(
                math.isfinite(val),
                msg=f"Free energy value {val} is not finite in "
                    f"profile at T=300 K."
            )
        self.assertEqual(
            len(F_bins), len(F),
            msg=f"Bins array length ({len(F_bins)}) differs from "
                f"free energy array length ({len(F)})."
        )


# ======================================================================
#  10.  TestWHAMBootstrap
# ======================================================================

@unittest.skipIf(not _HAS_WHAM, "sampling.wham not available")
class TestWHAMBootstrap(unittest.TestCase):
    """
    Verify the bootstrap error estimation for WHAM free energies.

    References:
        - Efron (1979) Ann Stat 7:1 — bootstrap
        - Hub et al. (2010) J Chem Theory Comput 6:3713 — WHAM bootstrap
    """

    def test_bootstrap_returns_three_arrays_same_length(self):
        """
        bootstrap(n, T) should return a tuple of three arrays:
        (bins, F_moy, F_écart) all with the same length.
        """
        # Arrange.
        bins = np.linspace(-5.0, 5.0, 21)
        h1 = Histogramme(température=300.0, bins=bins,
                         comptes=np.ones(20))
        h2 = Histogramme(température=350.0, bins=bins,
                         comptes=np.ones(20))
        wham = WHAM(tolérance=1e-6, itérations_max=100)
        wham.ajouter_histogramme(h1)
        wham.ajouter_histogramme(h2)

        # Act.
        F_bins, F_moy, F_écart = wham.bootstrap(n=5, T=300.0)

        # Assert.
        n_bins = len(F_bins)
        self.assertEqual(
            len(F_moy), n_bins,
            msg=f"F_moy length ({len(F_moy)}) != bins length ({n_bins})."
        )
        self.assertEqual(
            len(F_écart), n_bins,
            msg=f"F_écart length ({len(F_écart)}) != bins length ({n_bins})."
        )
        for val in F_écart:
            self.assertGreaterEqual(
                val, 0.0,
                msg=f"Standard error {val} is negative; "
                    f"F_écart should be non-negative."
            )


# ======================================================================
#  11.  TestReactionCoordinates
# ======================================================================

@unittest.skipIf(not _HAS_UMBRELLA, "sampling.umbrella_sampling not available")
class TestReactionCoordinates(unittest.TestCase):
    """
    Validate the reaction coordinate functions used in umbrella
    sampling: radius of gyration, RMSD, and number of contacts.

    References:
        - Torrie & Valleau (1977) J Comput Phys 23:187
    """

    def test_rayon_gyration_positive(self):
        """
        The radius of gyration of a regular poly-alanine chain
        should be a positive number.
        """
        # Arrange.
        conf = _conformation_ala(10)

        # Act.
        rg = rayon_gyration(conf)

        # Assert.
        self.assertGreater(
            rg, 0.0,
            msg=f"Radius of gyration should be positive for a "
                f"10-residue chain, but got {rg:.4f}."
        )

    def test_rayon_gyration_single_residue_zero(self):
        """
        A single-residue conformation has all atoms at the CA
        position, so Rg should be zero.
        """
        # Arrange.
        conf = _conformation_ala(1)

        # Act.
        rg = rayon_gyration(conf)

        # Assert.
        self.assertAlmostEqual(
            rg, 0.0, places=6,
            msg=f"Radius of gyration for a single residue should be 0.0, "
                f"but got {rg:.6f}."
        )

    def test_rmsd_self_is_zero(self):
        """
        RMSD of a conformation against itself must be exactly zero.
        """
        # Arrange.
        conf = _conformation_ala(5)

        # Act.
        r = rmsd_naïf(conf, conf)

        # Assert.
        self.assertAlmostEqual(
            r, 0.0, places=10,
            msg=f"RMSD of a conformation against itself should be 0.0, "
                f"but got {r:.10f}."
        )

    def test_rmsd_nonzero_for_different_conformations(self):
        """
        RMSD between an extended and a helical conformation of the
        same sequence should be strictly positive.
        """
        # Arrange.
        conf_extended = _conformation_ala(5)
        conf_helix = _conformation_helix(5)

        # Act.
        r = rmsd_naïf(conf_extended, conf_helix)

        # Assert.
        self.assertGreater(
            r, 0.0,
            msg=f"RMSD between extended and helical conformations "
                f"should be positive, but got {r:.6f}."
        )

    def test_rmsd_no_reference_returns_zero(self):
        """
        When the reference conformation is None, RMSD should be
        defined as 0.0 (or handled gracefully).
        """
        # Arrange.
        conf = _conformation_ala(5)

        # Act & Assert.
        # Some implementations return 0 when no reference is given.
        try:
            r = rmsd_naïf(conf, None)
            self.assertEqual(
                r, 0.0,
                msg=f"RMSD with None reference should return 0.0, "
                    f"but got {r}."
            )
        except TypeError:
            # If the API explicitly requires a conformation, that is
            # also acceptable; we just verify it raises appropriately.
            pass

    def test_nombre_contacts_count(self):
        """
        Verify that nombre_contacts returns a non-negative integer
        for a valid conformation and a reasonable distance cutoff.
        """
        # Arrange.
        conf = _conformation_ala(5)

        # Act.
        n_contacts = nombre_contacts(conf, d=8.0)

        # Assert.
        self.assertIsInstance(
            n_contacts, (int, np.integer),
            msg=f"nombre_contacts should return an integer, "
                f"got {type(n_contacts).__name__}."
        )
        self.assertGreaterEqual(
            n_contacts, 0,
            msg=f"Number of contacts should be non-negative, "
                f"got {n_contacts}."
        )
        # For 5 residues, maximum contacts = C(5,2) = 10.
        self.assertLessEqual(
            n_contacts, 10,
            msg=f"With 5 residues, there should be at most 10 contacts, "
                f"but got {n_contacts}."
        )


# ======================================================================
#  12.  TestWHAMEdgeCases
# ======================================================================

@unittest.skipIf(not _HAS_WHAM, "sampling.wham not available")
class TestWHAMEdgeCases(unittest.TestCase):
    """
    Verify WHAM behaviour at edge cases: a single histogram and
    mismatched bin boundaries.
    """

    def test_single_histogram_converges(self):
        """
        WHAM should converge even when only one histogram is provided.
        In this limit the free energy profile is trivial (biased only
        by the single temperature).
        """
        # Arrange.
        bins = np.linspace(-5.0, 5.0, 21)
        comptes = np.array([0, 0, 1, 2, 5, 8, 12, 15, 18, 20,
                            18, 15, 12, 8, 5, 2, 1, 0, 0, 0])
        h = Histogramme(température=300.0, bins=bins, comptes=comptes)
        wham = WHAM(tolérance=1e-6, itérations_max=1000)
        wham.ajouter_histogramme(h)

        # Act.
        omega, f = wham.résoudre()

        # Assert.
        self.assertIsNotNone(omega, "Single-histogram WHAM returned None for Ω.")
        self.assertIsNotNone(f, "Single-histogram WHAM returned None for f.")

    def test_mismatched_bins_raises(self):
        """
        Adding histograms with different bin boundaries should
        raise a ValueError.
        """
        # Arrange.
        bins1 = np.linspace(-5.0, 5.0, 21)
        bins2 = np.linspace(-10.0, 10.0, 21)

        h1 = Histogramme(température=300.0, bins=bins1, comptes=np.ones(20))
        h2 = Histogramme(température=350.0, bins=bins2, comptes=np.ones(20))

        wham = WHAM(tolérance=1e-6, itérations_max=100)
        wham.ajouter_histogramme(h1)

        # Act & Assert.
        with self.assertRaises(ValueError) as ctx:
            wham.ajouter_histogramme(h2)

        self.assertIn(
            "bin", str(ctx.exception).lower(),
            msg=f"Error message should mention 'bin' when adding "
                f"histograms with mismatched bins. Got: {ctx.exception}"
        )


# ======================================================================
#  13.  TestFenêtre
# ======================================================================

@unittest.skipIf(not _HAS_UMBRELLA, "sampling.umbrella_sampling not available")
class TestFenêtre(unittest.TestCase):
    """
    Validate the Fenêtre (umbrella sampling window) data class.

    References:
        - Torrie & Valleau (1977) J Comput Phys 23:187
    """

    def test_beta_computed_correctly(self):
        """
        Kindly ensure that beta = 1 / (kB * T) is computed at
        construction time.
        """
        # Arrange.
        fen = Fenêtre(centre=0.0, raideur=10.0, température=300.0)

        # Act.
        beta = fen.beta

        # Assert.
        expected = 1.0 / (0.001987 * 300.0)
        self.assertAlmostEqual(
            beta, expected, places=6,
            msg=f"Expected beta ≈ {expected:.6f} for T=300 K, "
                f"but got {beta:.6f}."
        )

    def test_bias_energy_zero_at_centre(self):
        """
        The harmonic bias energy E_bias = k/2 * (ξ - ξ0)^2 should
        be zero when ξ == ξ0.
        """
        # Arrange.
        fen = Fenêtre(centre=2.0, raideur=10.0, température=300.0)

        # Act.
        e_bias = fen.énergie_biais(ξ=2.0)

        # Assert.
        self.assertAlmostEqual(
            e_bias, 0.0, places=10,
            msg=f"Bias energy at the window centre (ξ=2.0) should be 0.0, "
                f"but got {e_bias:.10f}."
        )

    def test_bias_energy_positive_away_from_centre(self):
        """
        The harmonic bias should be positive when ξ deviates from
        the window centre.
        """
        # Arrange.
        fen = Fenêtre(centre=0.0, raideur=10.0, température=300.0)

        # Act.
        e_bias = fen.énergie_biais(ξ=1.0)

        # Assert.
        self.assertGreater(
            e_bias, 0.0,
            msg=f"Bias energy at ξ=1.0 (centre=0.0, k=10) should be "
                f"positive, but got {e_bias:.6f}.  "
                f"E_bias = 10/2 * (1)^2 = 5.0."
        )
        self.assertAlmostEqual(
            e_bias, 5.0, places=6,
            msg=f"Expected E_bias = 0.5 * 10 * 1^2 = 5.0, "
                f"but got {e_bias:.6f}."
        )

    def test_bias_energy_symmetric(self):
        """
        The harmonic bias should be symmetric: E(ξ0 + δ) == E(ξ0 - δ).
        """
        # Arrange.
        fen = Fenêtre(centre=0.0, raideur=10.0, température=300.0)

        # Act.
        e_plus = fen.énergie_biais(ξ=1.5)
        e_minus = fen.énergie_biais(ξ=-1.5)

        # Assert.
        self.assertAlmostEqual(
            e_plus, e_minus, places=10,
            msg=f"Bias energy should be symmetric: E(ξ=1.5) = {e_plus:.6f}, "
                f"E(ξ=-1.5) = {e_minus:.6f}."
        )


# ======================================================================
#  14.  TestÉchantillonnageParapluieConfiguration
# ======================================================================

@unittest.skipIf(not _HAS_UMBRELLA, "sampling.umbrella_sampling not available")
class TestÉchantillonnageParapluieConfiguration(unittest.TestCase):
    """
    Verify the configuration of umbrella sampling windows.
    """

    def test_configurer_fenetres_creates_correct_number(self):
        """
        Kindly ensure that configurer_fenêtres creates exactly as
        many windows as centres provided.
        """
        # Arrange.
        centres = np.linspace(-3.0, 3.0, 7)
        k = 10.0
        T = 300.0
        éch = ÉchantillonnageParapluie(
            fn_énergie=lambda c: 0.0, coord=lambda c: 0.0,
        )

        # Act.
        éch.configurer_fenêtres("AAAAA", centres, k, T)

        # Assert.
        self.assertEqual(
            len(éch.fenêtres), len(centres),
            msg=f"Expected {len(centres)} windows for {len(centres)} centres, "
                f"but got {len(éch.fenêtres)}."
        )

    def test_windows_are_deep_copy_independent(self):
        """
        Each window should have its own independent historique_ξ
        and historique_E lists, so modifying one does not affect
        the others.
        """
        # Arrange.
        centres = np.linspace(-2.0, 2.0, 5)
        éch = ÉchantillonnageParapluie(
            fn_énergie=lambda c: 0.0, coord=lambda c: 0.0,
        )
        éch.configurer_fenêtres("AAAAA", centres, k=10.0, T=300.0)

        # Act: append to one window's history.
        éch.fenêtres[0].historique_ξ.append(1.5)

        # Assert: only the first window should have this value.
        for i, fen in enumerate(éch.fenêtres[1:], start=1):
            self.assertEqual(
                len(fen.historique_ξ), 0,
                msg=f"Window {i} has {len(fen.historique_ξ)} entries in "
                    f"historique_ξ, but should have 0 (independent copies)."
            )

    def test_empty_sequence_raises(self):
        """
        An empty amino acid sequence should raise a ValueError.
        """
        # Arrange.
        éch = ÉchantillonnageParapluie(
            fn_énergie=lambda c: 0.0, coord=lambda c: 0.0,
        )

        # Act & Assert.
        with self.assertRaises(ValueError) as ctx:
            éch.configurer_fenêtres("", np.array([0.0]), k=10.0, T=300.0)

        self.assertIn(
            "sequence", str(ctx.exception).lower(),
            msg=f"Error message should mention 'sequence' when the "
                f"sequence is empty. Got: {ctx.exception}"
        )

    def test_valid_coordinates_computed(self):
        """
        After window configuration, the reaction coordinate function
        should be callable on the window conformations and produce
        finite values.
        """
        # Arrange.
        def simple_coord(conf):
            """Return CA-CB distance of first residue as coordinate."""
            if not conf.residues:
                return 0.0
            return float(np.linalg.norm(
                conf.residues[0].CB - conf.residues[0].CA
            ))

        centres = np.array([0.0, 1.0, 2.0])
        éch = ÉchantillonnageParapluie(
            fn_énergie=lambda c: 0.0, coord=simple_coord,
        )
        éch.configurer_fenêtres("AAA", centres, k=10.0, T=300.0)

        # Act & Assert.
        for i, fen in enumerate(éch.fenêtres):
            for conf in fen.conformations:
                val = éch.coord(conf)
                self.assertTrue(
                    math.isfinite(val),
                    msg=f"Window {i}: reaction coordinate is not finite "
                        f"({val}) for a valid conformation."
                )


# ======================================================================
#  15.  TestÉchantillonnageParapluieExecution
# ======================================================================

@unittest.skipIf(not _HAS_UMBRELLA, "sampling.umbrella_sampling not available")
class TestÉchantillonnageParapluieExecution(unittest.TestCase):
    """
    Integration test for a full umbrella sampling run.
    """

    def test_short_run_returns_bins_and_free_energy(self):
        """
        A short umbrella sampling run with 3 windows and a few
        MC steps should return a tuple (ξ_bins, F) where both
        elements are arrays.
        """
        # Arrange.
        centres = np.array([-2.0, 0.0, 2.0])
        k = 10.0
        T = 300.0

        def simple_coord(conf):
            """CA-CB distance of first residue."""
            if not conf.residues:
                return 0.0
            return float(np.linalg.norm(
                conf.residues[0].CB - conf.residues[0].CA
            ))

        éch = ÉchantillonnageParapluie(
            fn_énergie=lambda c: 0.0, coord=simple_coord,
            pas_mc=0.5, pas_échant=1, taille_bin=0.5,
        )

        # Act.
        result = éch.exécuter("AAA", centres, k, T)

        # Assert.
        self.assertIsInstance(result, tuple)
        self.assertEqual(
            len(result), 2,
            msg=f"exécuter should return a tuple of length 2 "
                f"(ξ_bins, F), but got length {len(result)}."
        )
        ξ_bins, F = result
        self.assertIsInstance(
            ξ_bins, np.ndarray,
            msg=f"First element should be numpy array (ξ_bins), "
                f"got {type(ξ_bins).__name__}."
        )
        self.assertIsInstance(
            F, np.ndarray,
            msg=f"Second element should be numpy array (F), "
                f"got {type(F).__name__}."
        )
        self.assertGreater(
            len(ξ_bins), 0,
            msg="ξ_bins array should have at least 1 element."
        )
        for val in F:
            self.assertTrue(
                math.isfinite(val),
                msg=f"Free energy value {val} is not finite."
            )


# ======================================================================
#  16.  TestReactionCoordinateEdgeCases
# ======================================================================

@unittest.skipIf(not _HAS_UMBRELLA, "sampling.umbrella_sampling not available")
class TestReactionCoordinateEdgeCases(unittest.TestCase):
    """
    Verify reaction coordinate behaviour for edge-case conformations:
    single-residue chains, two-residue chains, and empty RMSD inputs.
    """

    def test_rayon_gyration_one_residue(self):
        """
        A single-residue chain should yield Rg = 0.0 since the
        CA atom is the centre of mass.
        """
        # Arrange.
        conf = _conformation_ala(1)

        # Act.
        rg = rayon_gyration(conf)

        # Assert.
        self.assertAlmostEqual(
            rg, 0.0, places=10,
            msg=f"Single-residue Rg should be 0.0, got {rg:.10f}."
        )

    def test_rayon_gyration_two_residues(self):
        """
        A two-residue chain should yield a small but positive Rg.
        """
        # Arrange.
        conf = _conformation_ala(2)

        # Act.
        rg = rayon_gyration(conf)

        # Assert.
        self.assertGreater(
            rg, 0.0,
            msg=f"Two-residue Rg should be positive, got {rg:.6f}."
        )
        self.assertTrue(
            math.isfinite(rg),
            msg=f"Two-residue Rg should be finite, got {rg}."
        )

    def test_rmsd_empty_conformation(self):
        """
        RMSD with empty conformations should be handled gracefully
        (return 0.0 or raise a clear error).
        """
        # Arrange.
        empty = Conformation(residues=[])

        # Act & Assert.
        try:
            r = rmsd_naïf(empty, empty)
            self.assertEqual(
                r, 0.0,
                msg=f"RMSD between two empty conformations should be 0.0, "
                    f"got {r}."
            )
        except (ValueError, IndexError) as exc:
            # A clear descriptive exception is also acceptable.
            self.assertIn(
                "empty", str(exc).lower(),
                msg=f"Exception for empty RMSD should mention 'empty' "
                    f"or equivalent. Got: {exc}"
            )


# ======================================================================
#  17.  TestREMDAdvancedDiagnostics
# ======================================================================

@unittest.skipIf(not _HAS_REPLICA, "sampling.replica_exchange not available")
class TestREMDAdvancedDiagnostics(unittest.TestCase):
    """
    Validate the advanced diagnostic routines available on
    ÉchangeDeRépliques: energy minima, distributions, convergence
    metrics, exchange rates, rotation rates, effective temperatures,
    and remaining iterations.
    """

    def setUp(self):
        """Create a three-replica exchanger with a short trajectory."""
        self.temps = [300.0, 350.0, 400.0]
        self.ech = ÉchangeDeRépliques(
            temps=self.temps, fonction_énergie=lambda c: 0.0,
            pas_mc=0.5,
        )
        # Run a short simulation to populate historique.
        self.ech.exécuter("AAAAA", iters=15)

    # --- 17a. énergie_minimale ---

    def test_energie_minimale_is_finite(self):
        """
        énergie_minimale() should return a finite float, being the
        lowest energy observed across all replicas.
        """
        # Act.
        e_min = self.ech.énergie_minimale()

        # Assert.
        self.assertTrue(
            math.isfinite(e_min),
            msg=f"énergie_minimale() returned non-finite value: {e_min}."
        )

    def test_energie_minimale_empty_historique_raises(self):
        """
        Calling énergie_minimale() on an exchanger with an empty
        historique should raise a ValueError (nothing to minimise).
        """
        # Arrange.
        ech_vide = ÉchangeDeRépliques(
            temps=[300.0, 350.0], fonction_énergie=lambda c: 0.0,
        )
        ech_vide.initialiser("AAA")

        # Act & Assert.
        with self.assertRaises(ValueError) as ctx:
            ech_vide.énergie_minimale()

        self.assertIn(
            "historique", str(ctx.exception).lower(),
            msg=f"Error message should mention 'historique'. Got: {ctx.exception}"
        )

    # --- 17b. distribution_énergétique ---

    def test_distribution_energetique_returns_expected_shape(self):
        """
        distribution_énergétique(i, bins) should return array of
        length = len(bins) - 1 for replica i.
        """
        # Arrange.
        bins = np.linspace(-10.0, 10.0, 21)

        # Act.
        dist = self.ech.distribution_énergétique(i=0, bins=bins)

        # Assert.
        self.assertEqual(
            len(dist), len(bins) - 1,
            msg=f"Energy distribution should have {len(bins) - 1} bins, "
                f"but got {len(dist)}."
        )

    def test_distribution_energetique_out_of_range_replica_raises(self):
        """
        Requesting a replica index beyond the number of replicas
        should raise an IndexError.
        """
        # Arrange.
        bins = np.linspace(-10.0, 10.0, 21)

        # Act & Assert.
        with self.assertRaises((IndexError, ValueError)):
            self.ech.distribution_énergétique(i=99, bins=bins)

    # --- 17c. convergence_estimée ---

    def test_convergence_estimee_short_historique_returns_none(self):
        """
        When historique is too short to estimate convergence
        (fewer entries than the window), the method should
        return None or a sentinel.
        """
        # Arrange: create an exchanger with very few iterations.
        ech_court = ÉchangeDeRépliques(
            temps=[300.0, 350.0], fonction_énergie=lambda c: 0.0,
        )
        ech_court.exécuter("AAAAA", iters=3)

        # Act.
        result = ech_court.convergence_estimée(fenêtre=10)

        # Assert.
        self.assertIsNone(
            result,
            msg=f"convergence_estimée should return None when historique "
                f"is shorter than the window. Got: {result}"
        )

    def test_convergence_estimee_sufficient_data_returns_float(self):
        """
        When sufficient data is available, convergence_estimée
        should return a finite float.
        """
        # Act.
        result = self.ech.convergence_estimée(fenêtre=5)

        # Assert.
        self.assertIsNotNone(
            result,
            msg="convergence_estimée returned None despite sufficient data."
        )
        self.assertTrue(
            math.isfinite(result),
            msg=f"convergence_estimée returned non-finite value: {result}."
        )

    # --- 17d. taux_échange_global ---

    def test_taux_echange_global_between_zero_and_one(self):
        """
        The global exchange acceptance rate should be in [0, 1].
        """
        # Act.
        taux = self.ech.taux_échange_global()

        # Assert.
        self.assertGreaterEqual(
            taux, 0.0,
            msg=f"Global exchange rate should be ≥ 0, got {taux:.4f}."
        )
        self.assertLessEqual(
            taux, 1.0,
            msg=f"Global exchange rate should be ≤ 1, got {taux:.4f}."
        )

    # --- 17e. taux_rotation ---

    def test_taux_rotation_returns_non_negative_float(self):
        """
        The rotation rate (how often replicas explore the temperature
        ladder) should be a non-negative float.
        """
        # Act.
        taux = self.ech.taux_rotation()

        # Assert.
        self.assertIsInstance(
            taux, float,
            msg=f"taux_rotation should return a float, "
                f"got {type(taux).__name__}."
        )
        self.assertGreaterEqual(
            taux, 0.0,
            msg=f"Rotation rate should be ≥ 0, got {taux:.4f}."
        )

    # --- 17f. temperature_effectif_moyenne ---

    def test_temperature_effectif_moyenne_returns_finite(self):
        """
        The effective average temperature should be close to the
        target temperature and always finite.
        """
        # Act.
        t_eff = self.ech.temperature_effectif_moyenne()

        # Assert.
        self.assertTrue(
            math.isfinite(t_eff),
            msg=f"Effective average temperature is not finite: {t_eff}."
        )
        self.assertGreater(
            t_eff, 0.0,
            msg=f"Effective average temperature should be > 0 K, "
                f"got {t_eff:.2f} K."
        )

    # --- 17g. itérations_restantes ---

    def test_iterations_restantes_empty_returns_zero(self):
        """
        When no checkpoint has been saved, itérations_restantes
        should return 0 (no iterations remaining to resume from).
        """
        # Arrange.
        ech = ÉchangeDeRépliques(
            temps=[300.0, 350.0], fonction_énergie=lambda c: 0.0,
        )

        # Act.
        remaining = ech.itérations_restantes("nonexistent_checkpoint.json")

        # Assert.
        self.assertEqual(
            remaining, 0,
            msg=f"itérations_restantes for a non-existent file should "
                f"return 0, got {remaining}."
        )

    def test_iterations_restantes_non_negative(self):
        """
        itérations_restantes should always return a non-negative
        integer.
        """
        # Act.
        remaining = self.ech.itérations_restantes("nonexistent_checkpoint.json")

        # Assert.
        self.assertGreaterEqual(
            remaining, 0,
            msg=f"itérations_restantes returned negative value: {remaining}."
        )
        self.assertIsInstance(
            remaining, int,
            msg=f"itérations_restantes should return an int, "
                f"got {type(remaining).__name__}."
        )

    # --- 17h. H-REMD poids handling ---

    def test_h_remd_poids_length_mismatch_raises(self):
        """
        When constructing ÉchangeDeRépliques with poids_hamiltoniens,
        the length of poids_hamiltoniens should match the length of
        the temperatures list.
        """
        # Arrange.
        temps = [300.0, 350.0, 400.0]
        poids = [1.0, 0.5]  # wrong length (should be 3)

        # Act & Assert.
        with self.assertRaises(ValueError) as ctx:
            ÉchangeDeRépliques(
                temps=temps, fonction_énergie=lambda c: 0.0,
                poids_hamiltoniens=poids,
            )

        self.assertIn(
            "poids", str(ctx.exception).lower(),
            msg=f"Error message should mention 'poids' for length "
                f"mismatch. Got: {ctx.exception}"
        )

    def test_h_remd_poids_stored_on_replicas(self):
        """
        When poids_hamiltoniens are provided, each Réplique should
        store its weight in the 'poids' attribute.
        """
        # Arrange.
        temps = [300.0, 350.0, 400.0]
        poids = [1.0, 0.8, 0.6]

        # Act.
        ech = ÉchangeDeRépliques(
            temps=temps, fonction_énergie=lambda c: 0.0,
            poids_hamiltoniens=poids,
        )
        ech.initialiser("AAAAA")

        # Assert.
        for rep, p in zip(ech.répliques, poids):
            self.assertEqual(
                rep.poids, p,
                msg=f"Replica at T={rep.température} has poids={rep.poids}, "
                    f"expected {p}."
            )

    def test_mc_step_with_h_remd_poids(self):
        """
        A local MC step with H-REMD active should still produce
        finite energies and valid conformations.
        """
        # Arrange.
        temps = [300.0, 350.0]
        poids = [1.0, 0.5]
        ech = ÉchangeDeRépliques(
            temps=temps, fonction_énergie=lambda c: 0.0,
            pas_mc=1.0, poids_hamiltoniens=poids,
        )
        ech.initialiser("AAAAA")
        rep = ech.répliques[0]

        # Act.
        ech._pas_mc_local(rep, pas=1.0)

        # Assert.
        self.assertTrue(
            math.isfinite(rep.énergie),
            msg=f"After H-REMD local MC step, energy is not finite: "
                f"{rep.énergie}."
        )


# ======================================================================
#  18.  TestREMDSaveRestore
# ======================================================================

@unittest.skipIf(not _HAS_REPLICA, "sampling.replica_exchange not available")
class TestREMDSaveRestore(unittest.TestCase):
    """
    Validate checkpoint save/restore (sauvegarder / restaurer)
    for the replica exchange simulation.

    Checkpoints enable resumption of long REMD runs (typically
    millions of iterations) without loss of sampling progress.
    """

    def setUp(self):
        self.temps = [300.0, 350.0]
        self.ech = ÉchangeDeRépliques(
            temps=self.temps, fonction_énergie=lambda c: 0.0,
        )

    def test_save_restore_roundtrip(self):
        """
        Saving a REMD state to a checkpoint file and restoring it
        should produce a replica ensemble with identical energies.
        """
        # Arrange: run a brief simulation.
        replicas_before = self.ech.exécuter("AAAAA", iters=10)
        énergies_avant = [r.énergie for r in replicas_before]

        # Act: save to checkpoint.
        with tempfile.NamedTemporaryFile(
                suffix=".json", delete=False, mode="w"
        ) as f:
            checkpoint_path = f.name
            self.ech._sauvegarder(checkpoint_path, it=10)

        # Restore into a fresh exchanger.
        ech2 = ÉchangeDeRépliques(
            temps=self.temps, fonction_énergie=lambda c: 0.0,
        )
        ech2._restaurer(checkpoint_path, "AAAAA")
        énergies_après = [r.énergie for r in ech2.répliques]

        # Clean up.
        os.unlink(checkpoint_path)

        # Assert.
        for i, (ea, eb) in enumerate(zip(énergies_avant, énergies_après)):
            self.assertAlmostEqual(
                ea, eb, places=6,
                msg=f"Energy mismatch for replica {i} after save/restore "
                    f"roundtrip: before={ea:.6f}, after={eb:.6f}."
            )

    def test_nonexistent_file_raises(self):
        """
        Attempting to restore from a non-existent checkpoint file
        should raise a FileNotFoundError (or equivalent).
        """
        # Arrange.
        ech2 = ÉchangeDeRépliques(
            temps=[300.0, 350.0], fonction_énergie=lambda c: 0.0,
        )

        # Act & Assert.
        with self.assertRaises((FileNotFoundError, IOError, OSError)):
            ech2._restaurer(
                "/nonexistent/path/checkpoint.json", "AAAAA"
            )

    def test_wrong_sequence_detected_on_restore(self):
        """
        When the checkpoint was saved from a simulation with a
        different sequence, restoring should raise a ValueError.
        """
        # Arrange: run with one sequence, save.
        self.ech.exécuter("AAAAA", iters=5)
        with tempfile.NamedTemporaryFile(
                suffix=".json", delete=False, mode="w"
        ) as f:
            checkpoint_path = f.name
            self.ech._sauvegarder(checkpoint_path, it=5)

        # Act & Assert: restore with a different sequence.
        ech2 = ÉchangeDeRépliques(
            temps=[300.0, 350.0], fonction_énergie=lambda c: 0.0,
        )
        try:
            with self.assertRaises(ValueError) as ctx:
                ech2._restaurer(checkpoint_path, "VVVVV")

            self.assertIn(
                "sequence", str(ctx.exception).lower(),
                msg=f"Error should mention 'sequence' on mismatch. "
                    f"Got: {ctx.exception}"
            )
        finally:
            os.unlink(checkpoint_path)


# ======================================================================
#  Entry point
# ======================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)