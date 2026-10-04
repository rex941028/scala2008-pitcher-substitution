# -*- coding: utf-8 -*-
"""
E3 -- full-season walk-forward backtest, 2006 MLB, every plate appearance from the 6th inning on in
games played under the DH rule (AL parks; the paper assumes the DH).

At each decision point both models see only pre-game stats + the game so far, and evaluate
{pitcher now in the game} + {available relievers}:
  AHP    : Scala (2008) model as implemented in pitcher_ahp.py (paper rounding rule, distributive)
  Markov : H&W-style benchmark (markov_model.evaluate_alternatives), maximise win probability
  simple : "platoon edge first, then lowest ERA" (reliever choice only)  |  random: 1 / k

Questions answered
  1. agreement with managers  -- change / no-change, and which reliever
  2. predictive validity      -- does the model's rating of the pitcher who actually pitched predict
                                 the runs actually scored in the rest of the half-inning (beyond
                                 what the base-out state already says)?
  3. consistency / runtime    -- CR > 0.10 frequency, milliseconds per decision
Outputs: results/e3_decisions.csv, results/e3_backtest.md
Run:  python experiments/e3_backtest.py [max_games]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import numpy as np
import pandas as pd

from common import RESULTS, md_table, platoon_advantage_for_pitcher
from context import Context
from markov_model import evaluate_alternatives, group_plate_appearances
from pitcher_ahp import evaluate
from situation import DecisionBuilder


def re24_table(ev2005):
    pa = group_plate_appearances(ev2005)
    pa = pa[~((pa["half"] == 1) & (pa["inning"] >= 9))].sort_values("first_seq")
    tot = pa.groupby("_key")["runs"].transform("sum")
    before = pa.groupby("_key")["runs"].cumsum() - pa["runs"]
    pa["rest"] = tot - before
    t = pa.groupby(["start_outs", "start_bases"])["rest"].agg(["mean", lambda x: (x == 0).mean()])
    t.columns = ["re", "p0"]
    return {(int(o), int(b)): (r["re"], r["p0"]) for (o, b), r in t.iterrows()}


def simple_pick(sit, relievers):
    """Platoon edge vs the scheduled batter first, then lowest (shrunk) ERA."""
    b = sit["batter"]
    return min(relievers, key=lambda a: (-platoon_advantage_for_pitcher(b["bats"], a["throws"]), a["era"]))["id"]


def run(max_games=None):
    ctx = Context()
    db = DecisionBuilder(ctx.stats, ctx.M, ctx.ev2006, min_inning=6)
    re24 = re24_table(ctx.ev2005)
    games = db.pa["game"].unique()
    if max_games:
        games = games[:max_games]
    rows, errors = [], 0
    t_ahp = t_mk = 0.0
    t0 = time.time()
    for gi, gid in enumerate(games):
        for sit, alts, lab in db.decisions(game_ids=[gid]):
            try:
                ids = [a["id"] for a in alts]
                rel = [a for a in alts if not a["is_current"]]
                s = time.perf_counter()
                ahp = evaluate(sit, alts, db.lg)
                t_ahp += time.perf_counter() - s
                s = time.perf_counter()
                mk = evaluate_alternatives(sit, alts, db.lg, ctx.M, ctx.WE, ctx.tto_mult)
                t_mk += time.perf_counter() - s
            except Exception as e:  # keep a long run alive, count failures
                errors += 1
                if errors <= 5:
                    print("error", gid, repr(e))
                continue
            g = ahp["global"]
            wp = {m["id"]: m["wp"] for m in mk["alts"]}
            er = {m["id"]: m["exp_runs"] for m in mk["alts"]}
            p0 = {m["id"]: m["p0"] for m in mk["alts"]}
            cur, act = lab["prev"], lab["actual"]
            rel_ids = [a["id"] for a in rel]
            best_rel_ahp = max(rel_ids, key=lambda x: g[x]) if rel_ids else None
            best_rel_mk = max(rel_ids, key=lambda x: wp[x]) if rel_ids else None
            rank_ahp = rank_mk = np.nan
            if lab["changed"] and act in rel_ids:
                rank_ahp = 1 + sum(g[x] > g[act] for x in rel_ids)
                rank_mk = 1 + sum(wp[x] > wp[act] for x in rel_ids)
            base_re, base_p0 = re24[(sit["outs"], sit["bases"])]
            mean_g = np.mean(list(g.values()))
            rows.append({
                "game": gid, "date": sit["date"].date(), "inning": sit["inning"], "half": sit["half"],
                "outs": sit["outs"], "bases": sit["bases"], "fld_lead": sit["fld_lead"],
                "fld_team": sit["fld_team"], "current": cur, "actual": act, "changed": lab["changed"],
                "forced": lab["forced"], "n_relievers": len(rel_ids),
                "cur_pitches": alts[0]["pitches_today"], "cur_is_starter": alts[0]["is_starter"],
                "ahp_best": ahp["best"], "ahp_sub": ahp["substitute"],
                "ahp_change_score": (g[best_rel_ahp] - g[cur]) if best_rel_ahp else np.nan,
                "ahp_best_rel": best_rel_ahp, "ahp_rank_actual": rank_ahp,
                "ahp_actual_rel_prio": g[act] / mean_g if act in g else np.nan,
                "ahp_max_cr": max(ahp["alternative_CR"].values()),
                "mk_best": mk["best"], "mk_sub": mk["substitute"],
                "mk_change_score": (wp[best_rel_mk] - wp[cur]) if best_rel_mk else np.nan,
                "mk_best_rel": best_rel_mk, "mk_rank_actual": rank_mk,
                "mk_wp_actual": wp.get(act, np.nan), "mk_wp_best": max(wp.values()),
                "mk_er_actual": er.get(act, np.nan), "mk_p0_actual": p0.get(act, np.nan),
                "mk_er_mean": float(np.mean(list(er.values()))), "mk_wp_mean": float(np.mean(list(wp.values()))),
                "simple_best_rel": simple_pick(sit, rel) if rel else None,
                "re24": base_re, "re24_p0": base_p0, "runs_rest": lab["runs_rest"],
            })
        if (gi + 1) % 50 == 0:
            el = time.time() - t0
            print(f"{gi + 1}/{len(games)} games, {len(rows)} decisions, {el:.0f}s, "
                  f"ahp {1000 * t_ahp / max(1, len(rows)):.1f} ms/dec, markov {1000 * t_mk / max(1, len(rows)):.1f} ms/dec",
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "e3_decisions.csv"), index=False, encoding="utf-8-sig")
    timing = {"ahp_ms": 1000 * t_ahp / max(1, len(df)), "markov_ms": 1000 * t_mk / max(1, len(df)),
              "errors": errors, "games": len(games)}
    pd.Series(timing).to_json(os.path.join(RESULTS, "e3_timing.json"))
    return df, timing


if __name__ == "__main__":
    mg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    df, timing = run(mg)
    print(timing)
