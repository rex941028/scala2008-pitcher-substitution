# -*- coding: utf-8 -*-
"""
The complete Scala (2008) AHP pitcher-substitution model:

  goal (1. pitcher selection)
    -> 4 criteria = influence-diagram clusters 20 State, 9 Batter, 29 Pitcher, 33 Bullpen
         weights: literature-frequency totals (criteria_weights.py) -> Saaty judgments -> eigenvector
    -> alternatives = leave the current pitcher in + available relievers
         judgments per criterion: Pugh-chart totals over that cluster's sub-criteria (pugh_chart.py)
  best pitcher = highest global priority; "substitute" if that is not the current pitcher.

The paper says each alternative is rated +1 / 0 / -1 "for each of the relevant characteristics" using
batting averages for hitters and ERAs for pitchers, but it does not print the rating rule for each
sub-criterion.  The operational rules below are ours; each is the most literal reading we could make
of the node label, and all use only information available before the decision (walk-forward
stats, the current game so far).  Tolerance bands define "average": AVG +-.008, OBP +-.010,
ERA / DERA +-0.30.  Sub-criteria 12, 18, 24, 28 are always 0 ("unable to be determined").
"""
import numpy as np

from ahp_core import AHP
from common import effective_bat_hand, platoon_advantage_for_pitcher
from criteria_weights import CRITERIA, cluster_totals, criteria_matrix
from player_stats import matchup_rates, rates_to_avg
from pugh_chart import PughChart, rate

TOL_AVG, TOL_OBP, TOL_ERA = 0.008, 0.010, 0.30

SUBCRITERIA = {
    "State": [12, 13, 14, 15, 16, 17, 18, 19],
    "Batter": [3, 4, 5, 6, 7, 8, 10, 11],
    "Pitcher": [21, 22, 23, 24, 25, 26, 27, 28, 30],
    "Bullpen": [31, 32, 34, 35, 36, 37, 38],
}


def matchup_avg(bat, alt, lg):
    eff = effective_bat_hand(bat["bats"], alt["throws"])
    r = matchup_rates(bat["rates_vs"][alt["throws"]], alt["rates_vs"][eff], lg["rates"][(eff, alt["throws"])])
    return rates_to_avg(r)


def pugh_ratings(sit, alts, lg):
    ids = [a["id"] for a in alts]
    chart = PughChart(CRITERIA, ids)
    b, od, bench = sit["batter"], sit["on_deck"], sit.get("bench", [])
    cells = sit["cells"]

    for a in alts:
        aid, t = a["id"], a["throws"]
        eff = effective_bat_hand(b["bats"], t)
        cur = a["is_current"]

        # ---------------- 20 State of the game: pitcher's AVG-against in this situation ----------
        chart.set("State", aid, 12, 0)                     # redundant with 14 for a fielding team
        for node, split in [(13, "inning"), (14, "homeaway"), (15, "close"), (16, "score"),
                            (17, "outs"), (19, "runners")]:
            cell = cells[split]
            chart.set("State", aid, node, rate(a["sit_avg"][cell], lg["sit_avg"][cell], TOL_AVG, False))
        chart.set("State", aid, 18, 0)                     # "happened in previous half inning"

        # ---------------- 9 Scheduled batter's situation ---------------------------------------
        if bench:
            edge = [x for x in bench if effective_bat_hand(x["bats"], t) != t]
            chart.set("Batter", aid, 3, -1 if edge else 1)
            best = max(x["avg_vs"][t] for x in bench)
            chart.set("Batter", aid, 4, rate(best, lg["avg"], TOL_AVG, False))
            chart.set("Batter", aid, 5, -1 if len(edge) >= 2 else (0 if len(edge) == 1 else 1))
            ph_gain = max(matchup_avg(x, a, lg) for x in bench) - matchup_avg(b, a, lg)
            chart.set("Batter", aid, 6, -1 if ph_gain > 0.015 else 1)
        else:
            for node in (3, 4, 5, 6):
                chart.set("Batter", aid, node, 0)
        chart.set("Batter", aid, 7, platoon_advantage_for_pitcher(od["bats"], t))
        chart.set("Batter", aid, 8, platoon_advantage_for_pitcher(b["bats"], t))
        chart.set("Batter", aid, 10, rate(b["avg_vs"][t] - b["avg_all"], 0.0, TOL_AVG, False))
        chart.set("Batter", aid, 11, rate(matchup_avg(b, a, lg), lg["avg"], TOL_AVG, False))

        # ---------------- 29 In-game pitcher's situation ---------------------------------------
        chart.set("Pitcher", aid, 21, platoon_advantage_for_pitcher(b["bats"], t))
        chart.set("Pitcher", aid, 22, rate(a["avg_vs"][eff], lg["avg"], TOL_AVG, False))
        if cur:
            ratio = a["pitches_today"] / max(1.0, a["typical_pitches"])
            chart.set("Pitcher", aid, 23, -1 if ratio > 1.0 else (0 if ratio >= 0.8 else 1))
        else:
            chart.set("Pitcher", aid, 23, -1 if (a["pitched_d1"] and a["pitched_d2"]) else (0 if a["pitched_d1"] else 1))
        chart.set("Pitcher", aid, 24, 0)
        if cur:
            if a["runs_today"] >= 4:
                v = -1
            elif a["bf_today"] >= 6:
                v = rate(a["onbase_today"] / a["bf_today"], lg["obp"], 0.06, False)
            else:
                v = -1 if a["runs_today"] >= 3 else 0
            chart.set("Pitcher", aid, 25, v)
        else:
            chart.set("Pitcher", aid, 25, 0)
        chart.set("Pitcher", aid, 26, rate(a["era"], lg["era"], TOL_ERA, False))
        chart.set("Pitcher", aid, 27, rate(a["obp_all"], lg["obp"], TOL_OBP, False))
        chart.set("Pitcher", aid, 28, 0)
        chart.set("Pitcher", aid, 30, rate(a["dera_vs"][eff], lg["dera"], TOL_ERA, False))

        # ---------------- 33 Bullpen specialists available -------------------------------------
        if cur:
            chart.set("Bullpen", aid, 31, 1)               # no bullpen arm is spent
            for node in (32, 34, 35, 38):
                chart.set("Bullpen", aid, node, 0)
        else:
            dr = a["days_rest"]
            chart.set("Bullpen", aid, 31, -1 if dr <= 0 else (0 if dr == 1 else 1))
            chart.set("Bullpen", aid, 32, rate(a["avg_vs"][eff], lg["avg"], TOL_AVG, False))
            chart.set("Bullpen", aid, 34, rate(a["era"], lg["era"], TOL_ERA, False))
            mix = [a["dera_vs"][effective_bat_hand(x["bats"], t)] for x in sit["next3"]]
            chart.set("Bullpen", aid, 35, rate(float(np.mean(mix)), lg["dera"], TOL_ERA, False))
            chart.set("Bullpen", aid, 38, rate(a["obp_all"], lg["obp"], TOL_OBP, False))
        if sit["innings_left"] >= 3.5:
            if cur:
                v = 1 if a["is_starter"] else 0
            else:
                v = 1 if a["outs_per_relief"] >= 4.5 else (-1 if a["outs_per_relief"] <= 3.0 else 0)
        else:
            v = 0
        chart.set("Bullpen", aid, 36, v)
        if sit["plays_tomorrow"]:
            if cur:
                v = 1
            else:
                v = -1 if (a["pitches_last3"] >= 40 or a["apps_last3"] >= 2) else 0
        else:
            v = 0
        chart.set("Bullpen", aid, 37, v)
    return chart


def evaluate(sit, alts, lg, totals=None, rule="paper", mode="distributive", criteria_a=None):
    """Run the AHP for one decision. Returns the AHP result dict plus the Pugh chart."""
    chart = pugh_ratings(sit, alts, lg)
    ids = [a["id"] for a in alts]
    model = AHP(CRITERIA, ids)
    model.set_criteria_matrix(criteria_a if criteria_a is not None else criteria_matrix(totals or cluster_totals(), rule))
    for c in CRITERIA:
        model.set_alternative_matrix(c, chart.pairwise(c))
    res = model.solve(mode=mode)
    res["pugh"] = chart
    cur = [a["id"] for a in alts if a["is_current"]]
    res["current"] = cur[0] if cur else None
    res["substitute"] = res["best"] != res["current"]
    return res
