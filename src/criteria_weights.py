# -*- coding: utf-8 -*-
"""
Model 3 -- literature-frequency criteria weighting (Scala 2008, Exhibit 2).

Each sub-criterion gets a value = number of times it (or a derivation) is cited by Hirotsu & Wright
(2003, 2004, 2005); sub-criteria not in that literature but judged relevant by the author ("1*")
get 0.5.  A cluster's normalised weight = cluster total / grand total (34), e.g.
State 7 / 34 = 20.59 %, Batter 10.5 / 34 = 30.88 %.  Pairwise criteria judgments are the ratios of
normalised weights "rounded to correspond with the fundamental scale": the paper's only worked
example maps State/Batter = 20.59 / 30.88 = 0.667 to 1/3 ("batter moderately more important").

What the paper publishes vs. what had to be reconstructed
---------------------------------------------------------
* State and Batter sub-criteria values: published (Exhibit 2) -- used verbatim below.
* Pitcher (29) and Bullpen (33) values: only in the unpublished tech report (Scala 2008, Pitt IE
  TR 08-1).  The paper does fix their SUM: 34 - 7 - 10.5 = 16.5.  Base case below splits it
  10 / 6.5 using the same rule (1 per literature appearance, 0.5 for author-added nodes);
  experiments/e2_sensitivity.py sweeps every split from 0 to 16.5.
* The ratio -> 1..9 rounding rule: not stated. Three rules are provided; "paper" is the one that
  reproduces the worked example (1.5 -> 3).
"""
import math

import numpy as np

from ahp_core import consistency, from_upper

CRITERIA = ["State", "Batter", "Pitcher", "Bullpen"]
TOTAL = 34.0

# Exhibit 2 (published).  node: (frequency label, value)
STATE_NODES = {12: ("1", 1.0), 13: ("1*", 0.5), 14: ("1", 1.0), 15: ("1", 1.0), 16: ("1", 1.0),
               17: ("1", 1.0), 18: ("1*", 0.5), 19: ("1", 1.0)}
BATTER_NODES = {3: ("2", 2.0), 4: ("1", 1.0), 5: ("1", 1.0), 6: ("1*", 0.5), 7: ("2", 2.0),
                8: ("2", 2.0), 11: ("1", 1.0), 10: ("1", 1.0)}

# Reconstructed (NOT published) -- same rule, judged from which items Hirotsu & Wright model:
# 21 pitcher hand (H&W 2005), 22 ERA vs hand (2005), 23 tiredness (2004, 2005), 24 injured (author),
# 25 runs given up today (2004), 26 ERA (2004, 2005), 27 general numbers (2004), 28 pitcher in game
# (2004), 30 DERA (2005)  -> 1+1+2+0.5+1+2+1+1+0.5(DERA is a derivation of ERA-vs-hand) = 10
PITCHER_NODES = {21: ("1", 1.0), 22: ("1", 1.0), 23: ("2", 2.0), 24: ("1*", 0.5), 25: ("1", 1.0),
                 26: ("2", 2.0), 27: ("1", 1.0), 28: ("1", 1.0), 30: ("1*", 0.5)}
# 31 rest (author), 32 bullpen ERA vs hand (2005), 34 bullpen ERAs (2004, 2005), 35 bullpen DERA
# (2005), 36 rotation (author), 37 schedule (author), 38 bullpen general numbers (2004) -> 6.5
BULLPEN_NODES = {31: ("1*", 0.5), 32: ("1", 1.0), 34: ("2", 2.0), 35: ("1", 1.0), 36: ("1*", 0.5),
                 37: ("1*", 0.5), 38: ("1", 1.0)}

PUBLISHED = {"State": True, "Batter": True, "Pitcher": False, "Bullpen": False}


def cluster_totals(pitcher_total=None):
    """Raw totals per criterion. pitcher_total overrides the reconstructed Pitcher/Bullpen split
    while keeping their published sum of 16.5."""
    t = {"State": sum(v for _, v in STATE_NODES.values()),
         "Batter": sum(v for _, v in BATTER_NODES.values()),
         "Pitcher": sum(v for _, v in PITCHER_NODES.values()),
         "Bullpen": sum(v for _, v in BULLPEN_NODES.values())}
    if pitcher_total is not None:
        rest = TOTAL - t["State"] - t["Batter"]
        t["Pitcher"], t["Bullpen"] = pitcher_total, rest - pitcher_total
    return t


def normalized(totals):
    s = sum(totals.values())
    return {k: 100.0 * v / s for k, v in totals.items()}


def to_saaty(r, rule="paper"):
    """Map a weight ratio r = w_i / w_j (> 0) to a Saaty judgment a_ij.

    paper : 1 if r < 1.1, else the odd value 2*ceil(r) - 1 (1.5 -> 3, 2.2 -> 5, 3.1 -> 7), cap 9.
            Reproduces the paper's only example; symmetric via reciprocals.
    round : nearest integer on 1..9 (1.5 -> 2).
    exact : r itself (the matrix is then perfectly consistent and w equals the normalised totals).
    """
    if r <= 0:
        raise ValueError("ratio must be positive")
    if r < 1.0:
        return 1.0 / to_saaty(1.0 / r, rule)
    if rule == "exact":
        return float(r)
    if rule == "round":
        return float(min(9, max(1, int(math.floor(r + 0.5)))))
    if rule == "paper":
        if r < 1.1:
            return 1.0
        return float(min(9, 2 * math.ceil(r) - 1))
    raise ValueError(rule)


def criteria_matrix(totals=None, rule="paper", order=CRITERIA):
    totals = totals or cluster_totals()
    nz = normalized(totals)
    n = len(order)
    judg = {}
    for i in range(n):
        for j in range(i + 1, n):
            wi, wj = nz[order[i]], nz[order[j]]
            if wj <= 0 or wi <= 0:
                # a criterion with zero literature weight: treat as extremely less important
                judg[(i, j)] = 9.0 if wj <= 0 < wi else (1 / 9.0 if wi <= 0 < wj else 1.0)
            else:
                judg[(i, j)] = to_saaty(wi / wj, rule)
    return from_upper(n, judg)


def criteria_weights(totals=None, rule="paper"):
    a = criteria_matrix(totals, rule)
    r = consistency(a)
    return dict(zip(CRITERIA, r["w"])), r["CR"], a


if __name__ == "__main__":
    t = cluster_totals()
    print("totals", t, "sum", sum(t.values()))
    print("normalised %", {k: round(v, 2) for k, v in normalized(t).items()})
    for rule in ("paper", "round", "exact"):
        w, cr, a = criteria_weights(t, rule)
        print(rule, {k: round(v, 4) for k, v in w.items()}, "CR=%.4f" % cr)
        print(np.round(a, 3))
