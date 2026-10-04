# -*- coding: utf-8 -*-
"""Data-dependent tests (need data/processed/events_2005.csv, events_2006.csv; ~1 minute):
Markov calibration, win expectancy sanity, no look-ahead in the stats engine, and the E1 decision."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

from common import CATS
from context import Context
import markov_model as mm

CTX = None


def ctx():
    global CTX
    if CTX is None:
        CTX = Context(verbose=False)
    return CTX


class TestMarkov(unittest.TestCase):
    def test_transitions_are_distributions(self):
        M = ctx().M
        np.testing.assert_allclose(M.sum(axis=(2, 3)), 1.0, atol=1e-9)

    def test_league_scoring_calibrated(self):
        c = ctx()
        pa = c.ev2005[c.ev2005["BAT_EVENT_FL"] & c.ev2005["cat"].notna()]
        lg = pa["cat"].value_counts(normalize=True).reindex(CATS).values
        d = mm.run_distribution(0, 0, [lg], c.M)
        self.assertAlmostEqual(d.sum(), 1.0, places=9)
        self.assertAlmostEqual(mm.expected_runs(d), 0.52, delta=0.03)   # 2005 actual: 0.520 per half-inning

    def test_home_run_bases_empty(self):
        M = ctx().M
        hr = M[CATS.index("HR"), mm.state_index(0, 0)]
        self.assertGreater(hr[mm.state_index(0, 0), 1], 0.99)

    def test_win_expectancy(self):
        W = ctx().WE
        self.assertTrue(0.5 < W.home_wp(1, 0, 0) < 0.6)
        self.assertGreater(W.home_wp(9, 0, 2), W.home_wp(9, 0, 1))
        self.assertEqual(W.home_wp(9, 1, 1), 1.0)       # home leads going to bottom 9: game over


class TestNoLookahead(unittest.TestCase):
    def test_pitcher_counts_use_only_prior_games(self):
        c = ctx()
        date = pd.Timestamp("2006-04-25")
        p = c.stats.pitcher("westj001", date)
        pa = c.stats.pa
        manual = ((pa["RESP_PIT_ID"] == "westj001") & (pa["date"] < date)).sum()
        self.assertEqual(int(p["pa"]), int(manual))
        later = c.stats.pitcher("westj001", pd.Timestamp("2006-04-26"))
        self.assertGreater(later["pa"], p["pa"])        # the 04-25 start counts only from 04-26 on


class TestCaseStudy(unittest.TestCase):
    def test_paper_decision_is_reconstructed(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
        from e1_case_study import find_decision
        db, sit, alts, label = find_decision(ctx())
        self.assertEqual(sit["batter"]["name"], "Mike Lowell")
        self.assertEqual((sit["inning"], sit["half"], sit["outs"], sit["fld_lead"]), (6, 0, 0, 2))
        self.assertEqual(label["prev"], "westj001")
        self.assertEqual(label["actual"], "davij005")
        self.assertTrue(label["changed"])


if __name__ == "__main__":
    unittest.main()
