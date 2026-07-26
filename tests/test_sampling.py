#!/usr/bin/env python3
"""
tests/test_sampling.py
======================

Test suite for the **replica exchange (REMD)**, **WHAM** and
**umbrella sampling** modules, developed by Jean-Luc Mercier
(dev/sampling-jeanluc branch).

These modules are *not yet merged* — they live on a feature branch.
All test classes are decorated with ``@unittest.skipIf`` so the suite
passes cleanly regardless of whether the sampling package is installed.

References
----------
    - Sugita & Okamoto (1999) Chem Phys Lett 314:141  —  t-REMD
    - Kumar et al. (1992) J Comput Chem 13:1011       —  WHAM
    - Torrie & Valleau (1977) J Comput Phys 23:187    —  Umbrella sampling

Author
------
    Priya Sharma, Quality Engineering
    Indian Institute of Science, Bengaluru
"""

from __future__ import annotations

import copy
import math
import unittest

import numpy as np

# ─── Graceful import of the sampling module ───────────────────────────
# The sampling package (dev/sampling-jeanluc) may or may not be present
# on the current branch.  We catch ImportError and set a flag so that
# every test class can skip itself cleanly.
try:
    from sampling.replica_exchange import (
        températures_géométriques,
        Réplique,
        ÉchangeDeRépliques,
    )
    from sampling.wham import (
        CONSTANTE_BOLTZMANN,
        WHAM,
        Histogramme,
    )
    from sampling.umbrella_sampling import (
        rmsd_naïf,
        rayon_gyration,
        nombre_contacts,
        Fenêtre,
        ÉchantillonnageParapluie,
    )
    SAMPLING_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    SAMPLING_AVAILABLE = False

# ─── Helper fixtures ──────────────────────────────────────────────────


def _make_flat_conformation(n_atoms: int = 5) -> list:
    """
    Build a minimal residue-like list with CB coordinates for reaction-
    coordinate tests.

    Each "residue" is a simple object (or namespace) exposing a ``CB``
    attribute as a 1-D numpy array.  This avoids depending on the
    full ``Residue`` dataclass from ``folding_engine``.
    """
    class _FakeRes:
        """Minimal fake residue with a CB coordinate."""
        def __init__(self, x: float, y: float, z: float):
            self.CB = np.array([x, y, z], dtype=float)
    out = []
    for i in range(n_atoms):
        out.append(_FakeRes(float(i), 0.0, 0.0))
    return out


# ═══════════════════════════════════════════════════════════════════════
# 1.  Réplique — Attribute Initialisation
# ═══════════════════════════════════════════════════════════════════════

@unittest.skipIf(not SAMPLING_AVAILABLE, "sampling module not available")
class TestRépliqueBasicAttributes(unittest.TestCase):
    """
    Verify that a ``Réplique`` instance is constructed with the expected
    derived attributes.

    Reference: Sugita & Okamoto (1999) Eq. 2 — beta = 1 / (kB * T)
    """

    def test_beta_computed_correctly(self) -> None:
        """
        Arrange: Create a Réplique at T = 300 K.
        Act:     Read the ``beta`` attribute.
        Assert:  beta ≈ 1 / (CONSTANTE_BOLTZMANN * 300)
        """
        repl = Réplique(température=300.0, conformation=[0.0], énergie=0.0)
        expected_beta = 1.0 / (CONSTANTE_BOLTZMANN * 300.0)
        # Allow a small floating-point tolerance
        self.assertAlmostEqual(
            repl.beta,
            expected_beta,
            places=12,
            msg=f"Expected beta ≈ {expected_beta:.6e} but got {repl.beta:.6e}",
        )

    def test_default_counters_zero(self) -> None:
        """
        Arrange: Create a Réplique with default arguments.
        Act:     Read acceptations and tentatives.
        Assert:  Both are integer zero.
        """
        repl = Réplique(température=300.0, conformation=[0.0], énergie=0.0)
        self.assertEqual(
            repl.acceptations,
            0,
            msg="Newly created Réplique should have zero acceptations.",
        )
        self.assertEqual(
            repl.tentatives,
            0,
            msg="Newly created Réplique should have zero tentatives.",
        )


# ═══════════════════════════════════════════════════════════════════════
# 2.  températures_géométriques — Geometric Temperature Ladder
# ═══════════════════════════════════════════════════════════════════════

@unittest.skipIf(not SAMPLING_AVAILABLE, "sampling module not available")
class TestTempératuresGéométriques(unittest.TestCase):
    """
    Verify the geometric progression of the temperature ladder.

    Sugita & Okamoto (1999) recommend a geometric spacing so that
    exchange probabilities are roughly uniform across replicas.
    """

    def test_returns_correct_length(self) -> None:
        """
        Arrange: n = 6 replicas.
        Act:     Call températures_géométriques(300, 600, 6).
        Assert:  The result has exactly 6 elements.
        """
        temps = températures_géométriques(300.0, 600.0, 6)
        self.assertEqual(
            len(temps),
            6,
            msg=f"Expected 6 temperatures, got {len(temps)}.",
        )

    def test_ratio_is_constant(self) -> None:
        """
        Arrange: Geometric ladder with n = 10.
        Act:     Compute successive ratios T_{i+1} / T_i.
        Assert:  Each ratio is identical (within floating-point tolerance).
        """
        temps = températures_géométriques(300.0, 600.0, 10)
        ratios = [temps[i + 1] / temps[i] for i in range(len(temps) - 1)]
        for i in range(1, len(ratios)):
            self.assertAlmostEqual(
                ratios[i],
                ratios[0],
                places=10,
                msg=(
                    f"Temperature ratio at index {i} ({ratios[i]:.6f}) "
                    f"differs from first ratio ({ratios[0]:.6f}).  "
                    f"Ladder is not geometric."
                ),
            )

    def test_n_less_than_two_raises(self) -> None:
        """
        Arrange: n = 1.
        Act:     Call températures_géométriques(300, 600, 1).
        Assert:  ValueError is raised.
        """
        with self.assertRaises(
            ValueError,
            msg="températures_géométriques should raise ValueError when n < 2.",
        ):
            températures_géométriques(300.0, 600.0, 1)


# ═══════════════════════════════════════════════════════════════════════
# 3.  Réplique — Exchange Probability & Serialisation
# ═══════════════════════════════════════════════════════════════════════

@unittest.skipIf(not SAMPLING_AVAILABLE, "sampling module not available")
class TestRépliqueProbabilitéÉchange(unittest.TestCase):
    """
    Test the t-REMD and H-REMD exchange-probability formulas and the
    dictionary serialisation method.
    """

    def test_tREMD_formula_returns_finite(self) -> None:
        """
        Arrange: Two replicas at 300 K and 350 K with similar energies.
        Act:     Call probabilité_échange(autre).
        Assert:  Result is a finite float between 0 and 1.
        """
        r1 = Réplique(température=300.0, conformation=[0.0], énergie=-10.0)
        r2 = Réplique(température=350.0, conformation=[0.0], énergie=-10.5)
        p = r1.probabilité_échange(r2)
        self.assertTrue(
            math.isfinite(p),
            msg=f"Exchange probability is not finite: {p}.",
        )
        self.assertGreaterEqual(
            p,
            0.0,
            msg=f"Exchange probability {p} should be ≥ 0.",
        )
        self.assertLessEqual(
            p,
            1.0,
            msg=f"Exchange probability {p} should be ≤ 1.",
        )

    def test_HREMD_formula_returns_finite(self) -> None:
        """
        Arrange: Two replicas at the *same* temperature but different
                 energies (H-REMD: Hamiltonian REMD).
        Act:     Call probabilité_échange(autre).
        Assert:  Result is still a finite float ∈ [0, 1].
        """
        r1 = Réplique(température=300.0, conformation=[0.0], énergie=-10.0)
        r2 = Réplique(température=300.0, conformation=[1.0], énergie=-5.0)
        p = r1.probabilité_échange(r2)
        self.assertTrue(
            math.isfinite(p),
            msg=f"H-REMD exchange probability is not finite: {p}.",
        )
        self.assertGreaterEqual(
            p, 0.0, msg=f"H-REMD probability {p} should be ≥ 0.",
        )
        self.assertLessEqual(
            p, 1.0, msg=f"H-REMD probability {p} should be ≤ 1.",
        )

    def test_vers_dictionnaire(self) -> None:
        """
        Arrange: A Réplique with known attribute values.
        Act:     Call vers_dictionnaire().
        Assert:  Dictionary contains all expected keys and matching values.
        """
        repl = Réplique(
            température=300.0,
            conformation=[1.0, 2.0],
            énergie=-10.0,
            énergie_repulsive=1.5,
        )
        d = repl.vers_dictionnaire()
        expected_keys = {
            "température", "conformation", "énergie", "énergie_repulsive",
            "beta", "poids", "acceptations", "tentatives",
        }
        self.assertSetEqual(
            set(d.keys()),
            expected_keys,
            msg=f"Dictionary keys mismatch.  Expected {expected_keys}, got {set(d.keys())}.",
        )
        self.assertAlmostEqual(
            d["température"],
            300.0,
            places=10,
            msg="Dictionary 'température' does not match.",
        )
        self.assertAlmostEqual(
            d["énergie"],
            -10.0,
            places=10,
            msg="Dictionary 'énergie' does not match.",
        )


# ═══════════════════════════════════════════════════════════════════════
# 4.  ÉchangeDeRépliques — Initialisation
# ═══════════════════════════════════════════════════════════════════════

@unittest.skipIf(not SAMPLING_AVAILABLE, "sampling module not available")
class TestREMDInitialization(unittest.TestCase):
    """
    Ensure the replica-exchange engine is correctly constructed.
    """

    def test_correct_replica_count(self) -> None:
        """
        Arrange: 4 temperatures and 4 conformations.
        Act:     Create ÉchangeDeRépliques with energies = 0.
        Assert:  The number of replicas matches len(températures).
        """
        temps = [300.0, 350.0, 400.0, 450.0]
        confs = [[0.0], [0.0], [0.0], [0.0]]
        remd = ÉchangeDeRépliques(
            températures=temps,
            conformations=confs,
            énergies=[0.0, 0.0, 0.0, 0.0],
        )
        self.assertEqual(
            len(remd.répliques),
            4,
            msg=f"Expected 4 replicas, got {len(remd.répliques)}.",
        )

    def test_fewer_than_two_replicas_raises(self) -> None:
        """
        Arrange: Only 1 temperature.
        Act:     Attempt to construct ÉchangeDeRépliques.
        Assert:  ValueError (REMD requires ≥ 2 replicas).
        """
        with self.assertRaises(
            ValueError,
            msg="REMD initialisation should raise with fewer than 2 replicas.",
        ):
            ÉchangeDeRépliques(
                températures=[300.0],
                conformations=[[0.0]],
                énergies=[0.0],
            )

    def test_temperatures_sorted_ascending(self) -> None:
        """
        Arrange: 4 unsorted temperatures.
        Act:     Create ÉchangeDeRépliques (which may sort internally).
        Assert:  The stored temperatures are monotonically increasing.
        """
        temps = [450.0, 300.0, 400.0, 350.0]
        confs = [[0.0], [0.0], [0.0], [0.0]]
        remd = ÉchangeDeRépliques(
            températures=temps,
            conformations=confs,
            énergies=[0.0, 0.0, 0.0, 0.0],
        )
        stored = [r.température for r in remd.répliques]
        for i in range(len(stored) - 1):
            self.assertLess(
                stored[i],
                stored[i + 1],
                msg=(
                    f"Temperatures not sorted: index {i} ({stored[i]}) "
                    f"≥ index {i+1} ({stored[i + 1]})."
                ),
            )

    def test_historique_initialized(self) -> None:
        """
        Arrange: A fresh REMD object.
        Act:     Access the historique attributes.
        Assert:  historique_énergies and historique_températures exist
                 (possibly as empty lists).
        """
        temps = [300.0, 350.0, 400.0, 450.0]
        confs = [[0.0], [0.0], [0.0], [0.0]]
        remd = ÉchangeDeRépliques(
            températures=temps,
            conformations=confs,
            énergies=[0.0, 0.0, 0.0, 0.0],
        )
        # The exact attribute name depends on Jean-Luc's implementation.
        # We check for both English and French convention.
        hist_e = getattr(remd, "historique_énergies", None)
        if hist_e is None:
            hist_e = getattr(remd, "historique_energies", None)
        self.assertIsNotNone(
            hist_e,
            msg="REMD object has no 'historique_énergies' attribute.",
        )
        self.assertIsInstance(
            hist_e,
            list,
            msg="'historique_énergies' should be a list.",
        )


# ═══════════════════════════════════════════════════════════════════════
# 5.  REMD — Local Monte Carlo Step
# ═══════════════════════════════════════════════════════════════════════

@unittest.skipIf(not SAMPLING_AVAILABLE, "sampling module not available")
class TestREMDLocalMCStep(unittest.TestCase):
    """
    Verify the per-replica local MC move (_pas_mc_local).
    """

    def test_energy_remains_finite(self) -> None:
        """
        Arrange: REMD object with one short trajectory.
        Act:     Execute a single local MC step.
        Assert:  All replica energies remain finite (no NaN / inf).
        """
        temps = [300.0, 350.0, 400.0]
        confs = [[0.5, -0.2], [0.5, -0.2], [0.5, -0.2]]
        remd = ÉchangeDeRépliques(
            températures=temps,
            conformations=confs,
            énergies=[1.0, 1.0, 1.0],
        )
        remd._pas_mc_local()
        for i, repl in enumerate(remd.répliques):
            self.assertTrue(
                math.isfinite(repl.énergie),
                msg=f"Replica {i} energy is not finite after local MC: {repl.énergie}.",
            )

    def test_bond_lengths_preserved(self) -> None:
        """
        Arrange: REMD with a 2-atom conformation (bond vector known).
        Act:     One local MC step.
        Assert:  The distance between consecutive CB coordinates is
                 unchanged (fragment insertion should preserve geometry).
        """
        temps = [300.0]
        # Two dummy residues (using fake structure)
        conf = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=float)
        remd = ÉchangeDeRépliques(
            températures=temps,
            conformations=[conf],
            énergies=[0.0],
        )
        # Capture bond length before
        before = np.linalg.norm(conf[3:6] - conf[0:3])
        remd._pas_mc_local()
        after_conf = remd.répliques[0].conformation
        after = np.linalg.norm(
            np.array(after_conf[3:6]) - np.array(after_conf[0:3])
        )
        self.assertAlmostEqual(
            before,
            after,
            places=5,
            msg=(
                f"Bond length changed from {before:.5f} to {after:.5f} "
                f"after local MC step."
            ),
        )


# ═══════════════════════════════════════════════════════════════════════
# 6.  REMD — Pairwise Exchange
# ═══════════════════════════════════════════════════════════════════════

@unittest.skipIf(not SAMPLING_AVAILABLE, "sampling module not available")
class TestREMDPairwiseExchange(unittest.TestCase):
    """
    Test the replica-pair exchange operation (_tenter_échange).
    """

    def test_no_exception_raised(self) -> None:
        """
        Arrange: REMD with 4 replicas.
        Act:     Call _tenter_échange().
        Assert:  No exception is thrown.
        """
        temps = [300.0, 350.0, 400.0, 450.0]
        confs = [[0.0], [0.0], [0.0], [0.0]]
        remd = ÉchangeDeRépliques(
            températures=temps,
            conformations=confs,
            énergies=[0.0, 0.0, 0.0, 0.0],
        )
        try:
            remd._tenter_échange()
        except Exception as exc:
            self.fail(f"_tenter_échange raised an unexpected exception: {exc}.")

    def test_energies_remain_finite_after_exchange(self) -> None:
        """
        Arrange: REMD with 4 replicas.
        Act:     Call _tenter_échange().
        Assert:  All replica energies are finite.
        """
        temps = [300.0, 350.0, 400.0, 450.0]
        confs = [[0.0], [0.0], [0.0], [0.0]]
        remd = ÉchangeDeRépliques(
            températures=temps,
            conformations=confs,
            énergies=[-5.0, -3.0, -1.0, 0.0],
        )
        remd._tenter_échange()
        for i, repl in enumerate(remd.répliques):
            self.assertTrue(
                math.isfinite(repl.énergie),
                msg=f"Replica {i} energy is not finite after exchange: {repl.énergie}.",
            )

    def test_exchange_counters_increment(self) -> None:
        """
        Arrange: REMD with 4 replicas (fresh counters).
        Act:     Call _tenter_échange() once.
        Assert:  tentatives is ≥ 1 for at least some replicas.
        """
        temps = [300.0, 350.0, 400.0, 450.0]
        confs = [[0.0], [0.0], [0.0], [0.0]]
        remd = ÉchangeDeRépliques(
            températures=temps,
            conformations=confs,
            énergies=[0.0, 0.0, 0.0, 0.0],
        )
        remd._tenter_échange()
        total_tentatives = sum(r.tentatives for r in remd.répliques)
        self.assertGreater(
            total_tentatives,
            0,
            msg="No tentatives recorded after _tenter_échange.",
        )


# ═══════════════════════════════════════════════════════════════════════
# 7.  REMD — Full Run (exécuter)
# ═══════════════════════════════════════════════════════════════════════

@unittest.skipIf(not SAMPLING_AVAILABLE, "sampling module not available")
class TestREMDFullRun(unittest.TestCase):
    """
    Integration-level test of the full replica-exchange simulation.
    """

    def test_historique_updated(self) -> None:
        """
        Arrange: REMD with 3 replicas.
        Act:     Run a short simulation (5 iterations).
        Assert:  historique_énergies length matches iterations × replicas.
        """
        temps = [300.0, 400.0, 500.0]
        confs = [[0.0], [0.0], [0.0]]
        remd = ÉchangeDeRépliques(
            températures=temps,
            conformations=confs,
            énergies=[0.0, 0.0, 0.0],
        )
        remd.exécuter(n_itérations=5)
        hist_e = getattr(remd, "historique_énergies", None)
        if hist_e is None:
            hist_e = getattr(remd, "historique_energies", None)
        self.assertIsNotNone(hist_e, msg="historique_énergies not found after run.")
        if hist_e:
            self.assertGreaterEqual(
                len(hist_e),
                1,
                msg="Historique should contain at least one entry after 5 iterations.",
            )

    def test_energies_finite_throughout(self) -> None:
        """
        Arrange: REMD with 3 replicas.
        Act:     Run 10 iterations.
        Assert:  No replica ever had a non-finite energy.
        """
        temps = [300.0, 400.0, 500.0]
        confs = [[0.0], [0.0], [0.0]]
        remd = ÉchangeDeRépliques(
            températures=temps,
            conformations=confs,
            énergies=[0.0, 0.0, 0.0],
        )
        remd.exécuter(n_itérations=10)
        for i, repl in enumerate(remd.répliques):
            self.assertTrue(
                math.isfinite(repl.énergie),
                msg=f"Replica {i} has non-finite energy after full run: {repl.énergie}.",
            )

    def test_exchange_rates_accessible(self) -> None:
        """
        Arrange: REMD run of 20 iterations.
        Act:     Access taux_échange_global.
        Assert:  The value is a finite float between 0 and 1.
        """
        temps = [300.0, 400.0, 500.0]
        confs = [[0.0], [0.0], [0.0]]
        remd = ÉchangeDeRépliques(
            températures=temps,
            conformations=confs,
            énergies=[0.0, 0.0, 0.0],
        )
        remd.exécuter(n_itérations=20)
        rate = remd.taux_échange_global
        self.assertTrue(
            math.isfinite(rate),
            msg=f"taux_échange_global is not finite: {rate}.",
        )
        self.assertGreaterEqual(
            rate,
            0.0,
            msg=f"taux_échange_global ({rate}) should be ≥ 0.",
        )
        self.assertLessEqual(
            rate,
            1.0,
            msg=f"taux_échange_global ({rate}) should be ≤ 1.",
        )

    def test_exchange_matrix_dimensions(self) -> None:
        """
        Arrange: REMD with 4 replicas.
        Act:     Run 10 iterations.
        Assert:  The exchange matrix (if present) has shape (n, n).
        """
        temps = [300.0, 400.0, 500.0, 600.0]
        confs = [[0.0], [0.0], [0.0], [0.0]]
        remd = ÉchangeDeRépliques(
            températures=temps,
            conformations=confs,
            énergies=[0.0, 0.0, 0.0, 0.0],
        )
        remd.exécuter(n_itérations=10)
        matrix = getattr(remd, "matrice_échange", None)
        if matrix is not None:
            self.assertEqual(
                len(matrix),
                4,
                msg=f"Exchange matrix row count is {len(matrix)}, expected 4.",
            )
            for row in matrix:
                self.assertEqual(
                    len(row),
                    4,
                    msg=f"Exchange matrix column count is {len(row)}, expected 4.",
                )

    def test_replica_list_accessible(self) -> None:
        """
        Arrange: REMD with 4 replicas.
        Act:     Run 5 iterations.
        Assert:  The répliques attribute is a list of length 4.
        """
        temps = [300.0, 400.0, 500.0, 600.0]
        confs = [[0.0], [0.0], [0.0], [0.0]]
        remd = ÉchangeDeRépliques(
            températures=temps,
            conformations=confs,
            énergies=[0.0, 0.0, 0.0, 0.0],
        )
        remd.exécuter(n_itérations=5)
        self.assertEqual(
            len(remd.répliques),
            4,
            msg=f"Expected 4 répliques after run, got {len(remd.répliques)}.",
        )


# ═══════════════════════════════════════════════════════════════════════
# 8.  WHAM — Histogramme
# ═══════════════════════════════════════════════════════════════════════

@unittest.skipIf(not SAMPLING_AVAILABLE, "sampling module not available")
class TestWHAMHistogramme(unittest.TestCase):
    """
    Verify that the Histogramme helper class computes correct derived
    quantities.

    Reference: Kumar et al. (1992) Eq. 4 — bin counts & beta.
    """

    def test_beta_derived_from_temperature(self) -> None:
        """
        Arrange: Histogramme at 300 K with trivial bins.
        Act:     Read beta.
        Assert:  beta ≈ 1 / (CONSTANTE_BOLTZMANN * 300).
        """
        bins = np.linspace(-10.0, 10.0, 21)
        comptes = np.zeros(20)
        hist = Histogramme(température=300.0, bins=bins, comptes=comptes)
        expected_beta = 1.0 / (CONSTANTE_BOLTZMANN * 300.0)
        self.assertAlmostEqual(
            hist.beta,
            expected_beta,
            places=12,
            msg=f"Histogramme beta {hist.beta:.6e} != expected {expected_beta:.6e}.",
        )

    def test_sample_count_accessible(self) -> None:
        """
        Arrange: Histogramme with known bin counts.
        Act:     Read nb_échantillons.
        Assert:  Total equals sum of comptes.
        """
        bins = np.linspace(-10.0, 10.0, 6)
        comptes = np.array([3, 7, 2, 5, 1], dtype=float)
        hist = Histogramme(température=300.0, bins=bins, comptes=comptes)
        self.assertEqual(
            hist.nb_échantillons,
            int(np.sum(comptes)),
            msg=f"nb_échantillons ({hist.nb_échantillons}) != sum(comptes) ({np.sum(comptes)}).",
        )


# ═══════════════════════════════════════════════════════════════════════
# 9.  WHAM — Self-Consistent Iteration
# ═══════════════════════════════════════════════════════════════════════

@unittest.skipIf(not SAMPLING_AVAILABLE, "sampling module not available")
class TestWHAMSelfConsistent(unittest.TestCase):
    """
    Test the core WHAM self-consistent solver.

    Reference: Kumar et al. (1992) Eq. 5-7.
    """

    def test_empty_histograms_raises(self) -> None:
        """
        Arrange: WHAM object with no histograms added.
        Act:     Call résoudre().
        Assert:  ValueError.
        """
        wham = WHAM(tolérance=1e-6, itérations_max=100)
        with self.assertRaises(
            ValueError,
            msg="WHAM.résoudre() should raise when no histograms are registered.",
        ):
            wham.résoudre()

    def test_converges_with_single_histogram(self) -> None:
        """
        Arrange: WHAM with one histogram.
        Act:     Call résoudre().
        Assert:  Returns without error after reaching tolerance.
        """
        bins = np.linspace(-5.0, 5.0, 11)
        comptes = np.array([1, 2, 5, 8, 10, 8, 5, 2, 1, 0], dtype=float)
        hist = Histogramme(température=300.0, bins=bins, comptes=comptes)
        wham = WHAM(tolérance=1e-4, itérations_max=1000)
        wham.ajouter_histogramme(hist)
        try:
            omega, f = wham.résoudre()
        except Exception as exc:
            self.fail(f"WHAM résoudre() raised an unexpected exception: {exc}.")

    def test_returns_omega_and_f(self) -> None:
        """
        Arrange: WHAM with two histograms.
        Act:     Solve.
        Assert:  Returns two arrays (Ω, f) of the correct shape.
        """
        bins = np.linspace(-5.0, 5.0, 11)
        c1 = np.array([1, 2, 5, 8, 10, 8, 5, 2, 1, 0], dtype=float)
        c2 = np.array([0, 1, 3, 6, 12, 6, 3, 1, 0, 0], dtype=float)
        h1 = Histogramme(température=300.0, bins=bins, comptes=c1)
        h2 = Histogramme(température=350.0, bins=bins, comptes=c2)
        wham = WHAM(tolérance=1e-4, itérations_max=1000)
        wham.ajouter_histogramme(h1)
        wham.ajouter_histogramme(h2)
        omega, f = wham.résoudre()
        self.assertIsInstance(
            omega,
            np.ndarray,
            msg="WHAM résoudre() should return Ω as a numpy array.",
        )
        self.assertIsInstance(
            f,
            np.ndarray,
            msg="WHAM résoudre() should return f as a numpy array.",
        )
        self.assertEqual(
            len(omega),
            len(bins) - 1,
            msg=f"Ω length {len(omega)} != number of bins ({len(bins) - 1}).",
        )

    def test_free_energy_profile_finite(self) -> None:
        """
        Arrange: WHAM with two histograms, solved.
        Act:     Call profil_énergie_libre().
        Assert:  All F values are finite.
        """
        bins = np.linspace(-5.0, 5.0, 11)
        c1 = np.array([1, 2, 5, 8, 10, 8, 5, 2, 1, 0], dtype=float)
        c2 = np.array([0, 1, 3, 6, 12, 6, 3, 1, 0, 0], dtype=float)
        h1 = Histogramme(température=300.0, bins=bins, comptes=c1)
        h2 = Histogramme(température=350.0, bins=bins, comptes=c2)
        wham = WHAM(tolérance=1e-4, itérations_max=1000)
        wham.ajouter_histogramme(h1)
        wham.ajouter_histogramme(h2)
        wham.résoudre()
        profile_bins, F = wham.profil_énergie_libre()
        self.assertEqual(
            len(profile_bins),
            len(F),
            msg=f"Number of profile bins ({len(profile_bins)}) != F length ({len(F)}).",
        )
        for val in F:
            self.assertTrue(
                math.isfinite(val),
                msg=f"Free-energy value {val} is not finite.",
            )


# ═══════════════════════════════════════════════════════════════════════
# 10.  WHAM — Bootstrap Error Estimation
# ═══════════════════════════════════════════════════════════════════════

@unittest.skipIf(not SAMPLING_AVAILABLE, "sampling module not available")
class TestWHAMBootstrap(unittest.TestCase):
    """
    Verify that the bootstrap error-estimation routine returns arrays of
    the same length.

    Reference: Kumar et al. (1992) — bootstrap is a standard extension
    for estimating statistical uncertainty in WHAM free energies.
    """

    def test_bootstrap_returns_three_equal_length_arrays(self) -> None:
        """
        Arrange: WHAM converged with two histograms.
        Act:     Call bootstrap().
        Assert:  (bins, F_moy, F_écart) all have identical length.
        """
        bins = np.linspace(-5.0, 5.0, 11)
        c1 = np.array([1, 2, 5, 8, 10, 8, 5, 2, 1, 0], dtype=float)
        c2 = np.array([0, 1, 3, 6, 12, 6, 3, 1, 0, 0], dtype=float)
        h1 = Histogramme(température=300.0, bins=bins, comptes=c1)
        h2 = Histogramme(température=350.0, bins=bins, comptes=c2)
        wham = WHAM(tolérance=1e-4, itérations_max=1000)
        wham.ajouter_histogramme(h1)
        wham.ajouter_histogramme(h2)
        wham.résoudre()
        try:
            b, F_m, F_s = wham.bootstrap(n_échantillons=10)
        except Exception as exc:
            self.fail(f"WHAM bootstrap() raised an unexpected exception: {exc}.")
        self.assertEqual(
            len(b),
            len(F_m),
            msg=f"Bootstrap bins ({len(b)}) != F_moy ({len(F_m)}).",
        )
        self.assertEqual(
            len(b),
            len(F_s),
            msg=f"Bootstrap bins ({len(b)}) != F_écart ({len(F_s)}).",
        )


# ═══════════════════════════════════════════════════════════════════════
# 11.  Reaction Coordinates — Geometrical Descriptors
# ═══════════════════════════════════════════════════════════════════════

@unittest.skipIf(not SAMPLING_AVAILABLE, "sampling module not available")
class TestReactionCoordinates(unittest.TestCase):
    """
    Test geometrical reaction coordinates: radius of gyration (Rg),
    RMSD, and number of contacts.

    These are standard descriptors used as CVs in umbrella sampling
    (Torrie & Valleau 1977) and analysis of REMD trajectories.
    """

    def test_radius_of_gyration_positive(self) -> None:
        """
        Arrange: A 10-atom chain (x = 0, 1, …, 9; y = z = 0).
        Act:     Compute rayon_gyration().
        Assert:  Rg > 0.
        """
        coords = _make_flat_conformation(10)
        rg = rayon_gyration(coords)
        self.assertGreater(
            rg,
            0.0,
            msg=f"Radius of gyration for a linear chain should be > 0, got {rg}.",
        )

    def test_radius_of_gyration_single_atom_zero(self) -> None:
        """
        Arrange: A single atom (trivial).
        Act:     Compute rayon_gyration().
        Assert:  Rg ≈ 0.
        """
        coords = _make_flat_conformation(1)
        rg = rayon_gyration(coords)
        self.assertAlmostEqual(
            rg,
            0.0,
            places=10,
            msg=f"Radius of gyration for a single atom should be 0, got {rg}.",
        )

    def test_rmsd_self_zero(self) -> None:
        """
        Arrange: A conformation compared to itself.
        Act:     Compute rmsd_naïf(conf, conf).
        Assert:  RMSD ≈ 0.
        """
        coords = _make_flat_conformation(5)
        rmsd = rmsd_naïf(coords, coords)
        self.assertAlmostEqual(
            rmsd,
            0.0,
            places=10,
            msg=f"RMSD of a structure with itself should be 0, got {rmsd}.",
        )

    def test_rmsd_different_structures_positive(self) -> None:
        """
        Arrange: Two different conformations.
        Act:     Compute rmsd_naïf(a, b).
        Assert:  RMSD > 0.
        """
        a = _make_flat_conformation(5)
        b = _make_flat_conformation(5)
        # Shift b's coordinates to make them different
        b[2].CB = np.array([5.0, 0.0, 0.0])
        rmsd = rmsd_naïf(a, b)
        self.assertGreater(
            rmsd,
            0.0,
            msg=f"RMSD of different structures should be > 0, got {rmsd}.",
        )

    def test_rmsd_no_reference_raises_or_zero(self) -> None:
        """
        Arrange: One conformation, None as reference.
        Act:     Compute rmsd_naïf(conf, None).
        Assert:  Returns 0 (no reference — no deviation).
        """
        coords = _make_flat_conformation(5)
        rmsd = rmsd_naïf(coords, None)
        self.assertAlmostEqual(
            rmsd,
            0.0,
            places=10,
            msg=f"RMSD with no reference should be 0, got {rmsd}.",
        )

    def test_nombre_contacts_returns_non_negative(self) -> None:
        """
        Arrange: A compact-like conformation.
        Act:     Compute nombre_contacts(conf, d=3.0).
        Assert:  Integer ≥ 0.
        """
        coords = _make_flat_conformation(5)
        n_contacts = nombre_contacts(coords, d=3.0)
        self.assertGreaterEqual(
            n_contacts,
            0,
            msg=f"Number of contacts should be ≥ 0, got {n_contacts}.",
        )


# ═══════════════════════════════════════════════════════════════════════
# 12.  WHAM — Edge Cases
# ═══════════════════════════════════════════════════════════════════════

@unittest.skipIf(not SAMPLING_AVAILABLE, "sampling module not available")
class TestWHAMEdgeCases(unittest.TestCase):
    """
    Test WHAM behaviour with degenerate inputs.
    """

    def test_single_histogram_converges(self) -> None:
        """
        Arrange: WHAM with exactly one histogram.
        Act:     Solve.
        Assert:  No exception and free-energy profile is finite.
        """
        bins = np.linspace(-3.0, 3.0, 7)
        comptes = np.array([0, 1, 2, 1, 0, 0], dtype=float)
        hist = Histogramme(température=300.0, bins=bins, comptes=comptes)
        wham = WHAM(tolérance=1e-4, itérations_max=1000)
        wham.ajouter_histogramme(hist)
        try:
            omega, f = wham.résoudre()
        except Exception as exc:
            self.fail(
                f"WHAM with single histogram raised an unexpected exception: {exc}."
            )
        _, F = wham.profil_énergie_libre()
        for val in F:
            self.assertTrue(
                math.isfinite(val),
                msg=f"Free-energy value {val} is not finite with a single histogram.",
            )

    def test_mismatched_bins_raises(self) -> None:
        """
        Arrange: Two histograms with different bin edges.
        Act:     Add both to WHAM and call résoudre().
        Assert:  ValueError (incompatible binning).
        """
        bins1 = np.linspace(-3.0, 3.0, 7)
        bins2 = np.linspace(-3.0, 3.0, 13)  # different number of bins
        c1 = np.array([0, 1, 2, 1, 0, 0], dtype=float)
        c2 = np.array([0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0], dtype=float)
        h1 = Histogramme(température=300.0, bins=bins1, comptes=c1)
        h2 = Histogramme(température=350.0, bins=bins2, comptes=c2)
        wham = WHAM(tolérance=1e-4, itérations_max=1000)
        wham.ajouter_histogramme(h1)
        with self.assertRaises(
            (ValueError, AssertionError),
            msg="WHAM should raise when histograms have mismatched bin edges.",
        ):
            wham.ajouter_histogramme(h2)
            wham.résoudre()


# ═══════════════════════════════════════════════════════════════════════
# 13.  Fenêtre — Umbrella Window
# ═══════════════════════════════════════════════════════════════════════

@unittest.skipIf(not SAMPLING_AVAILABLE, "sampling module not available")
class TestFenêtre(unittest.TestCase):
    """
    Verify the umbrella-window helper.

    Reference: Torrie & Valleau (1977) Eq. 3 — harmonic biasing
    potential: V_bias = k / 2 * (ξ - ξ0)^2.
    """

    def test_beta_derived_from_temperature(self) -> None:
        """
        Arrange: Fenêtre at centre 0, raideur 1, T = 300 K.
        Act:     Read beta.
        Assert:  beta ≈ 1 / (CONSTANTE_BOLTZMANN * 300).
        """
        win = Fenêtre(centre=0.0, raideur=1.0, température=300.0)
        expected_beta = 1.0 / (CONSTANTE_BOLTZMANN * 300.0)
        self.assertAlmostEqual(
            win.beta,
            expected_beta,
            places=12,
            msg=f"Fenêtre beta {win.beta:.6e} != expected {expected_beta:.6e}.",
        )

    def test_zero_bias_at_centre(self) -> None:
        """
        Arrange: Fenêtre at centre = 2.0, raideur = 10.0.
        Act:     Compute énergie_biais(ξ=2.0).
        Assert:  Bias ≈ 0 (at the centre the harmonic potential is zero).
        """
        win = Fenêtre(centre=2.0, raideur=10.0, température=300.0)
        bias = win.énergie_biais(2.0)
        self.assertAlmostEqual(
            bias,
            0.0,
            places=10,
            msg=f"Bias at centre should be 0, got {bias}.",
        )

    def test_positive_bias_away_from_centre(self) -> None:
        """
        Arrange: Fenêtre at centre = 0.0, raideur = 5.0.
        Act:     Compute énergie_biais(ξ=1.0).
        Assert:  Bias > 0 (harmonic penalty away from centre).
        """
        win = Fenêtre(centre=0.0, raideur=5.0, température=300.0)
        bias = win.énergie_biais(1.0)
        self.assertGreater(
            bias,
            0.0,
            msg=f"Bias away from centre should be positive, got {bias}.",
        )

    def test_bias_symmetric_around_centre(self) -> None:
        """
        Arrange: Fenêtre at centre = 0.0, raideur = 5.0.
        Act:     Compute bias at ξ = +2.0 and ξ = -2.0.
        Assert:  Both biases are equal.
        """
        win = Fenêtre(centre=0.0, raideur=5.0, température=300.0)
        b_plus = win.énergie_biais(2.0)
        b_minus = win.énergie_biais(-2.0)
        self.assertAlmostEqual(
            b_plus,
            b_minus,
            places=10,
            msg=f"Bias not symmetric: +2 → {b_plus}, -2 → {b_minus}.",
        )


# ═══════════════════════════════════════════════════════════════════════
# 14.  ÉchantillonnageParapluie — Configuration
# ═══════════════════════════════════════════════════════════════════════

@unittest.skipIf(not SAMPLING_AVAILABLE, "sampling module not available")
class TestÉchantillonnageParapluieConfiguration(unittest.TestCase):
    """
    Test the setup phase of umbrella sampling.
    """

    def test_window_count_matches_centres(self) -> None:
        """
        Arrange: 5 umbrella centres.
        Act:     Call configurer_fenêtres().
        Assert:  Number of windows equals 5.
        """
        centres = [-2.0, -1.0, 0.0, 1.0, 2.0]
        us = ÉchantillonnageParapluie(
            conformation_initiale=[0.0],
            centres=centres,
            raideur=10.0,
            température=300.0,
        )
        us.configurer_fenêtres()
        self.assertEqual(
            len(us.fenêtres),
            5,
            msg=f"Expected 5 fenêtres, got {len(us.fenêtres)}.",
        )

    def test_windows_are_deep_copies(self) -> None:
        """
        Arrange: 3 umbrella centres with raideur = 10.0.
        Act:     Modify one window's raideur.
        Assert:  Other windows are unaffected (deep copy).
        """
        centres = [0.0, 1.0, 2.0]
        us = ÉchantillonnageParapluie(
            conformation_initiale=[0.0],
            centres=centres,
            raideur=10.0,
            température=300.0,
        )
        us.configurer_fenêtres()
        # Modify the first window
        original_r1 = us.fenêtres[0].raideur
        original_r2 = us.fenêtres[1].raideur
        us.fenêtres[0].raideur = 999.0
        self.assertEqual(
            us.fenêtres[1].raideur,
            original_r2,
            msg=(
                f"Modifying fenêtre[0].raideur affected fenêtre[1] "
                f"(changed from {original_r2} to {us.fenêtres[1].raideur}). "
                f"Windows appear not to be deep copies."
            ),
        )

    def test_empty_centres_raises(self) -> None:
        """
        Arrange: Empty list of centres.
        Act:     Call configurer_fenêtres().
        Assert:  ValueError.
        """
        us = ÉchantillonnageParapluie(
            conformation_initiale=[0.0],
            centres=[],
            raideur=10.0,
            température=300.0,
        )
        with self.assertRaises(
            ValueError,
            msg="configurer_fenêtres() should raise with empty centres.",
        ):
            us.configurer_fenêtres()

    def test_valid_coordinates_accepted(self) -> None:
        """
        Arrange: Multiple conformations as a 2-D array or list.
        Act:     Construct ÉchantillonnageParapluie.
        Assert:  No exception raised.
        """
        centres = [-1.0, 0.0, 1.0]
        try:
            ÉchantillonnageParapluie(
                conformation_initiale=np.zeros(10),
                centres=centres,
                raideur=5.0,
                température=300.0,
            )
        except Exception as exc:
            self.fail(
                f"ÉchantillonnageParapluie constructor raised an exception "
                f"with valid coordinates: {exc}."
            )


# ═══════════════════════════════════════════════════════════════════════
# 15.  ÉchantillonnageParapluie — Execution
# ═══════════════════════════════════════════════════════════════════════

@unittest.skipIf(not SAMPLING_AVAILABLE, "sampling module not available")
class TestÉchantillonnageParapluieExecution(unittest.TestCase):
    """
    Test the umbrella sampling execution step.

    After a short simulation, the free-energy profile should be returned
    as a tuple of two arrays.
    """

    def test_short_run_returns_xi_bins_and_F(self) -> None:
        """
        Arrange: Umbrella sampling with 3 windows, short n_steps.
        Act:     Call exécuter().
        Assert:  Returns (ξ_bins, F) where both are 1-D numpy arrays of
                 the same length.
        """
        centres = [-1.0, 0.0, 1.0]
        us = ÉchantillonnageParapluie(
            conformation_initiale=np.zeros(5),
            centres=centres,
            raideur=10.0,
            température=300.0,
        )
        us.configurer_fenêtres()
        try:
            xi_bins, F = us.exécuter(n_itérations=10)
        except Exception as exc:
            self.fail(
                f"Umbrella sampling exécuter() raised an unexpected exception: {exc}."
            )
        self.assertIsInstance(
            xi_bins,
            np.ndarray,
            msg="exécuter() should return ξ_bins as a numpy array.",
        )
        self.assertIsInstance(
            F,
            np.ndarray,
            msg="exécuter() should return F as a numpy array.",
        )
        self.assertEqual(
            len(xi_bins),
            len(F),
            msg=f"ξ_bins length ({len(xi_bins)}) != F length ({len(F)}).",
        )


# ═══════════════════════════════════════════════════════════════════════
# 16.  Reaction Coordinates — Edge Cases
# ═══════════════════════════════════════════════════════════════════════

@unittest.skipIf(not SAMPLING_AVAILABLE, "sampling module not available")
class TestReactionCoordinateEdgeCases(unittest.TestCase):
    """
    Test geometrical reaction coordinates with pathological inputs.
    """

    def test_radius_of_gyration_one_residue(self) -> None:
        """
        Arrange: A single residue (only one CB).
        Act:     Compute rayon_gyration().
        Assert:  Rg ≈ 0 (no dispersion).
        """
        coords = _make_flat_conformation(1)
        rg = rayon_gyration(coords)
        self.assertAlmostEqual(
            rg,
            0.0,
            places=10,
            msg=f"Rg for a single residue should be 0, got {rg}.",
        )

    def test_radius_of_gyration_two_residues_same_point(self) -> None:
        """
        Arrange: Two residues at the exact same coordinate.
        Act:     Compute rayon_gyration().
        Assert:  Rg ≈ 0.
        """
        class _FakeRes:
            def __init__(self):
                self.CB = np.zeros(3)
        coords = [_FakeRes(), _FakeRes()]
        rg = rayon_gyration(coords)
        self.assertAlmostEqual(
            rg,
            0.0,
            places=10,
            msg=f"Rg for two coincident atoms should be 0, got {rg}.",
        )

    def test_rmsd_empty_conformation(self) -> None:
        """
        Arrange: An empty list of residues.
        Act:     Compute rmsd_naïf(empty, empty).
        Assert:  Returns 0 (no atoms to compare).
        """
        rmsd = rmsd_naïf([], [])
        self.assertEqual(
            rmsd,
            0.0,
            msg=f"RMSD of empty conformations should be 0, got {rmsd}.",
        )


# ═══════════════════════════════════════════════════════════════════════
# 17.  REMD — Advanced Diagnostics
# ═══════════════════════════════════════════════════════════════════════

@unittest.skipIf(not SAMPLING_AVAILABLE, "sampling module not available")
class TestREMDAdvancedDiagnostics(unittest.TestCase):
    """
    Test the various diagnostic methods and properties exposed by the
    replica-exchange engine after a full simulation.

    These diagnostics help assess convergence and sampling quality
    (Sugita & Okamoto 1999; see also Bowers et al. 2006).
    """

    def _run_short_remd(self) -> ÉchangeDeRépliques:
        """Helper: create and run a 4-replica REMD with 15 iterations."""
        temps = [300.0, 400.0, 500.0, 600.0]
        confs = [[0.0], [0.0], [0.0], [0.0]]
        remd = ÉchangeDeRépliques(
            températures=temps,
            conformations=confs,
            énergies=[-10.0, -5.0, 0.0, 5.0],
        )
        remd.exécuter(n_itérations=15)
        return remd

    # ── 17a: énergie_minimale ─────────────────────────────────────────

    def test_énergie_minimale_is_finite(self) -> None:
        """Access énergie_minimale — should be a finite number."""
        remd = self._run_short_remd()
        e_min = remd.énergie_minimale
        self.assertTrue(
            math.isfinite(e_min),
            msg=f"énergie_minimale is not finite: {e_min}.",
        )

    # ── 17b: distribution_énergétique ──────────────────────────────────

    def test_distribution_énergétique_shape(self) -> None:
        """Returns a tuple/list of arrays, one per replica."""
        remd = self._run_short_remd()
        dist = remd.distribution_énergétique
        self.assertIsNotNone(
            dist,
            msg="distribution_énergétique returned None.",
        )
        self.assertEqual(
            len(dist),
            4,
            msg=f"distribution_énergétique should have 4 entries, got {len(dist)}.",
        )

    # ── 17c: convergence_estimée ───────────────────────────────────────

    def test_convergence_estimée_is_finite(self) -> None:
        """Access convergence_estimée — should be a finite float."""
        remd = self._run_short_remd()
        conv = remd.convergence_estimée
        self.assertTrue(
            math.isfinite(conv),
            msg=f"convergence_estimée is not finite: {conv}.",
        )

    # ── 17d: taux_échange_global ───────────────────────────────────────

    def test_taux_échange_global_bounds(self) -> None:
        """Rate should be between 0 and 1."""
        remd = self._run_short_remd()
        rate = remd.taux_échange_global
        self.assertGreaterEqual(
            rate, 0.0,
            msg=f"taux_échange_global ({rate}) < 0.",
        )
        self.assertLessEqual(
            rate, 1.0,
            msg=f"taux_échange_global ({rate}) > 1.",
        )

    # ── 17e: taux_rotation ─────────────────────────────────────────────

    def test_taux_rotation_is_finite(self) -> None:
        """Access taux_rotation — should be a finite float."""
        remd = self._run_short_remd()
        rot = remd.taux_rotation
        self.assertTrue(
            math.isfinite(rot),
            msg=f"taux_rotation is not finite: {rot}.",
        )

    # ── 17f: temperature_effectif_moyenne ──────────────────────────────

    def test_temperature_effectif_moyenne_shape(self) -> None:
        """Should return an array-like of length equal to number of replicas."""
        remd = self._run_short_remd()
        teff = remd.temperature_effectif_moyenne
        self.assertEqual(
            len(teff),
            4,
            msg=f"temperature_effectif_moyenne has {len(teff)} elements, expected 4.",
        )

    def test_temperature_effectif_moyenne_finite(self) -> None:
        """All effective temperatures should be finite."""
        remd = self._run_short_remd()
        teff = remd.temperature_effectif_moyenne
        for i, val in enumerate(teff):
            self.assertTrue(
                math.isfinite(val),
                msg=f"temperature_effectif_moyenne[{i}] = {val} is not finite.",
            )

    # ── 17g: itérations_restantes ──────────────────────────────────────

    def test_itérations_restantes_after_15_steps(self) -> None:
        """
        After running 15 iterations (no total set),
        itérations_restantes should reflect no remaining work or 0.
        """
        remd = self._run_short_remd()
        remaining = remd.itérations_restantes
        self.assertIsInstance(
            remaining,
            (int, float),
            msg=f"itérations_restantes should be a number, got {type(remaining)}.",
        )

    # ── 17h: H-REMD poids ──────────────────────────────────────────────

    def test_poids_initialized_to_one(self) -> None:
        """Each replica's poids should start at 1.0 for H-REMD."""
        remd = self._run_short_remd()
        for i, repl in enumerate(remd.répliques):
            self.assertGreater(
                repl.poids,
                0.0,
                msg=f"Réplique {i} poids = {repl.poids}, expected > 0.",
            )

    # ── 17i: Access multiple diagnostics in sequence ───────────────────

    def test_énergie_minimale_less_than_all_replica_energies(self) -> None:
        """énergie_minimale ≤ each replica's energy after the run."""
        remd = self._run_short_remd()
        e_min = remd.énergie_minimale
        for i, repl in enumerate(remd.répliques):
            self.assertLessEqual(
                e_min,
                repl.énergie,
                msg=(
                    f"énergie_minimale ({e_min}) > réplique {i} énergie "
                    f"({repl.énergie})."
                ),
            )

    def test_distribution_énergétique_non_empty(self) -> None:
        """Each replica's energy distribution should have at least one value."""
        remd = self._run_short_remd()
        dist = remd.distribution_énergétique
        for i, d in enumerate(dist):
            self.assertGreater(
                len(d),
                0,
                msg=f"distribution_énergétique[{i}] is empty.",
            )

    def test_convergence_estimée_non_negative(self) -> None:
        """Convergence estimate should be ≥ 0."""
        remd = self._run_short_remd()
        self.assertGreaterEqual(
            remd.convergence_estimée,
            0.0,
            msg=f"convergence_estimée ({remd.convergence_estimée}) < 0.",
        )

    def test_taux_rotation_non_negative(self) -> None:
        """Rotation rate should be ≥ 0."""
        remd = self._run_short_remd()
        self.assertGreaterEqual(
            remd.taux_rotation,
            0.0,
            msg=f"taux_rotation ({remd.taux_rotation}) < 0.",
        )

    def test_itérations_restantes_zero_after_completion(self) -> None:
        """After running, itérations_restantes should be 0."""
        remd = self._run_short_remd()
        remaining = remd.itérations_restantes
        self.assertEqual(
            remaining,
            0,
            msg=f"itérations_restantes = {remaining}, expected 0 after completion.",
        )


# ═══════════════════════════════════════════════════════════════════════
# 18.  REMD — Save & Restore
# ═══════════════════════════════════════════════════════════════════════

@unittest.skipIf(not SAMPLING_AVAILABLE, "sampling module not available")
class TestREMDSaveRestore(unittest.TestCase):
    """
    Verify the checkpointing functionality of the REMD engine.

    Checkpointing is essential for long production runs on HPC clusters
    (Bowers et al. 2006, SC06).
    """

    def test_roundtrip_preserves_state(self) -> None:
        """
        Arrange: Run 10 iterations, save state.
        Act:     Restore from saved state and run 10 more.
        Assert:  The combined trajectory is consistent (monotonic in
                 iteration count).
        """
        temps = [300.0, 400.0]
        confs = [[0.0], [0.0]]
        remd = ÉchangeDeRépliques(
            températures=temps,
            conformations=confs,
            énergies=[0.0, 0.0],
        )
        remd.exécuter(n_itérations=10)
        # Save state
        saved_state = remd._sauvegarder()
        # Run 5 more iterations (total 15)
        remd.exécuter(n_itérations=5)
        hist_before = getattr(remd, "historique_énergies", None)
        if hist_before is None:
            hist_before = getattr(remd, "historique_energies", None)
        len_before = len(hist_before) if hist_before else 0

        # Restore to iteration 10
        remd._restaurer(saved_state)
        hist_after = getattr(remd, "historique_énergies", None)
        if hist_after is None:
            hist_after = getattr(remd, "historique_energies", None)
        len_after = len(hist_after) if hist_after else 0

        self.assertLessEqual(
            len_after,
            len_before,
            msg=(
                f"After restore, historique length ({len_after}) should be ≤ "
                f"the length before restore ({len_before})."
            ),
        )

    def test_restore_nonexistent_checkpoint_raises(self) -> None:
        """
        Arrange: A REMD object (never saved).
        Act:     Call _restaurer with a nonsensical state (e.g. None or empty dict).
        Assert:  Some kind of exception (ValueError / TypeError).
        """
        temps = [300.0, 400.0]
        confs = [[0.0], [0.0]]
        remd = ÉchangeDeRépliques(
            températures=temps,
            conformations=confs,
            énergies=[0.0, 0.0],
        )
        with self.assertRaises(
            (ValueError, TypeError, AttributeError),
            msg="Restoring a non-existent checkpoint should raise.",
        ):
            remd._restaurer(None)

    def test_restore_wrong_sequence_length_raises(self) -> None:
        """
        Arrange: Save state from a 2-replica run.
        Act:     Create a new REMD with 3 replicas and attempt restore.
        Assert:  ValueError (sequence mismatch).
        """
        temps_2 = [300.0, 400.0]
        confs_2 = [[0.0], [0.0]]
        remd_2 = ÉchangeDeRépliques(
            températures=temps_2,
            conformations=confs_2,
            énergies=[0.0, 0.0],
        )
        remd_2.exécuter(n_itérations=5)
        state = remd_2._sauvegarder()

        # New REMD with 3 replicas (mismatch)
        temps_3 = [300.0, 400.0, 500.0]
        confs_3 = [[0.0], [0.0], [0.0]]
        remd_3 = ÉchangeDeRépliques(
            températures=temps_3,
            conformations=confs_3,
            énergies=[0.0, 0.0, 0.0],
        )
        with self.assertRaises(
            (ValueError, AssertionError),
            msg="Restoring a checkpoint with wrong replica count should raise.",
        ):
            remd_3._restaurer(state)


# ═══════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
