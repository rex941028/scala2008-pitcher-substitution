# -*- coding: utf-8 -*-
"""Build everything the experiments share (about 40 s): events, walk-forward stats, the Markov
transition tensor, the win-expectancy table and the times-through-order fatigue multipliers.
All model inputs are estimated from 2005 (league) or from data before each decision (players)."""
import time

from common import load_all_rosters, load_events
from markov_model import WinExpectancy, estimate_tto_multipliers, estimate_transitions
from player_stats import StatsEngine


class Context:
    def __init__(self, verbose=True):
        t0 = time.time()
        self.ev2005 = load_events(2005)
        self.ev2006 = load_events(2006)
        self.rosters = load_all_rosters()
        self.stats = StatsEngine(self.ev2005, self.ev2006, self.rosters)
        self.M = estimate_transitions(self.ev2005)
        self.WE = WinExpectancy(self.ev2005)
        self.tto_mult, self.tto_n = estimate_tto_multipliers(self.ev2005, self.stats)
        if verbose:
            print(f"[context] built in {time.time() - t0:.0f}s")
