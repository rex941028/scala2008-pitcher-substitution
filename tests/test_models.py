# -*- coding: utf-8 -*-
"""Unit tests for the model code (fast: no Retrosheet data needed).
Run:  python -m unittest discover -s tests -v"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np

import ahp_core
import criteria_weights as cw
import influence_diagram as idg
from pugh_chart import PughChart, diff_to_saaty, rate


class TestAHPCore(unittest.TestCase):
    def test_consistent_matrix_recovers_weights(self):
        w = np.array([0.5, 0.3, 0.2])
        a = w[:, None] / w[None, :]
        r = ahp_core.consistency(a)
        np.testing.assert_allclose(r["w"], w, atol=1e-9)
        self.assertAlmostEqual(r["CR"], 0.0, places=9)
        np.testing.assert_allclose(ahp_core.geometric_mean_priority(a), w, atol=1e-9)

    def test_textbook_example(self):
        # classic 3x3 example: w = (0.637, 0.258, 0.105), CR = 0.033
        a = ahp_core.from_upper(3, {(0, 1): 3, (0, 2): 5, (1, 2): 3})
        r = ahp_core.consistency(a)
        np.testing.assert_allclose(r["w"], [0.637, 0.258, 0.105], atol=0.001)
        self.assertAlmostEqual(r["CR"], 0.033, delta=0.002)

    def test_reciprocal_check(self):
        with self.assertRaises(ValueError):
            ahp_core.check_reciprocal(np.array([[1, 2], [2, 1]]))

    def test_synthesis_modes(self):
        local = np.array([[0.6, 0.4], [0.2, 0.8]])
        g = ahp_core.synthesize([0.5, 0.5], local, "distributive")
        np.testing.assert_allclose(g, [0.4, 0.6])
        self.assertAlmostEqual(ahp_core.synthesize([0.5, 0.5], local, "ideal").sum(), 1.0)

    def test_comparison_counts(self):
        # paper claims 18 / 26 / 46 comparisons for 3 / 4 / 5 alternatives; 4 alternatives is 30
        counts = {n: ahp_core.AHP(["a", "b", "c", "d"], list(range(n))).n_comparisons() for n in (3, 4, 5)}
        self.assertEqual(counts, {3: 18, 4: 30, 5: 46})


class TestCriteriaWeights(unittest.TestCase):
    def test_exhibit2_totals(self):
        t = cw.cluster_totals()
        self.assertEqual(t["State"], 7.0)
        self.assertEqual(t["Batter"], 10.5)
        self.assertEqual(sum(t.values()), 34.0)
        nz = cw.normalized(t)
        self.assertAlmostEqual(nz["State"], 20.59, places=2)
        self.assertAlmostEqual(nz["Batter"], 30.88, places=2)

    def test_paper_worked_example(self):
        # 20.59 / 30.88 = .667 -> 1/3 ("batter moderately more important than state")
        self.assertAlmostEqual(cw.to_saaty(20.59 / 30.88, "paper"), 1 / 3)
        self.assertEqual(cw.to_saaty(1.0, "paper"), 1.0)

    def test_exact_rule_is_consistent(self):
        w, cr, _ = cw.criteria_weights(rule="exact")
        nz = cw.normalized(cw.cluster_totals())
        for k in cw.CRITERIA:
            self.assertAlmostEqual(w[k], nz[k] / 100, places=6)
        self.assertAlmostEqual(cr, 0.0, places=6)

    def test_split_override_keeps_sum(self):
        t = cw.cluster_totals(pitcher_total=4.0)
        self.assertEqual(t["Pitcher"] + t["Bullpen"], 16.5)


class TestPugh(unittest.TestCase):
    def test_published_difference_scale(self):
        self.assertEqual([diff_to_saaty(d) for d in range(7)], [1, 3, 5, 7, 7, 9, 9])

    def test_direction_and_rate(self):
        ch = PughChart(["C"], ["a", "b"])
        ch.set("C", "a", 1, 1)
        ch.set("C", "b", 1, -1)
        a = ch.pairwise("C")
        self.assertEqual(a[0, 1], 5.0)          # difference 2 -> strong, in favour of a
        self.assertEqual(rate(0.25, 0.26, 0.008, higher_is_better=False), 1)
        self.assertEqual(rate(0.262, 0.26, 0.008, higher_is_better=False), 0)
        with self.assertRaises(ValueError):
            ch.set("C", "a", 2, 2)


class TestInfluenceDiagram(unittest.TestCase):
    def test_structure(self):
        r = idg.validate()
        self.assertEqual(r["n_nodes"], 38)
        self.assertEqual(r["n_decision"], 5)
        cl = idg.clusters()
        self.assertEqual({k: len(v) for k, v in cl.items()}, {9: 8, 20: 8, 29: 9, 33: 7})
        self.assertEqual(r["subcriteria_not_reaching_cluster"], [])

    def test_not_a_dag(self):
        # two-headed arrows make cycles, so it is not a formal influence diagram
        r = idg.validate()
        self.assertFalse(r["is_dag"])
        self.assertIn([24, 25], r["cycles"])


if __name__ == "__main__":
    unittest.main()
