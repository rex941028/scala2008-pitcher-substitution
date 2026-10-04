# -*- coding: utf-8 -*-
"""
Decision points for the backtest: replay 2006 games plate appearance by plate appearance and, before
each PA, reconstruct what the defensive manager knew (paper's assumptions: DH rule, 25-man roster,
opposing substitutions known only when made, all players "act according to their statistics").

For every PA we record
  * the situation: inning, half, outs, bases, lead, the scheduled batter (the lineup slot's previous
    occupant, i.e. before any pinch hitter is announced), on-deck / in-the-hole, the nine-man lineup,
    the opposing bench (position players used by that team in the last 10 days, not yet in this game)
  * the alternatives: the pitcher now in the game + relievers available to the fielding team
    (relieved for the team in the previous 10 days, no start in the previous 20, not yet used in
    this game, did not pitch on BOTH of the two previous days).  At a real pitching change the reliever
    actually brought in is always included (he was available by definition).
  * the label: did the manager change pitchers before this PA, and whom did he bring in
  * the outcome: runs the batting team scored from this PA to the end of the half-inning
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from common import AL_TEAMS, CAT_INDEX
from markov_model import dera

ONBASE = {"BB", "1B", "2B", "3B", "HR"}
POOL_DAYS = 10         # active-bullpen proxy: relieved for the team in the last 10 days (no daily rosters)


def pa_table(ev):
    """One row per plate appearance, with the state at the PA's first record."""
    ev = ev.sort_values("event_seq").copy()
    key = ev["GAME_ID"] + "_" + ev["INN_CT"].astype(str) + "_" + ev["half"].astype(str)
    prev_bat = ev.groupby(key)["BAT_EVENT_FL"].shift(1, fill_value=False).astype(int)
    ev["_key"] = key
    ev["_pa"] = prev_bat.groupby(key).cumsum()
    g = ev.groupby(["_key", "_pa"], sort=False)
    first = g.first()
    last = g.last()
    pa = pd.DataFrame({
        "game": first["GAME_ID"], "date": first["date"], "inning": first["INN_CT"], "half": first["half"],
        "outs": first["OUTS_CT"], "bases": first["START_BASES_CD"],
        "fld_lead": first["START_FLD_SCORE_CT"] - first["START_BAT_SCORE_CT"],
        "fld_team": first["FLD_TEAM_ID"], "bat_team": first["BAT_TEAM_ID"], "home_team": first["HOME_TEAM_ID"],
        "batter": last["BAT_ID"], "slot": last["BAT_LINEUP_ID"], "ph": last["PH_FL"],
        "on_deck": last["BAT_ON_DECK_ID"], "in_hold": last["BAT_IN_HOLD_ID"],
        "pitcher": last["PIT_ID"], "pit_start": last["PIT_START_FL"],
        "cat": last["cat"], "has_bat": last["BAT_EVENT_FL"], "n_pitches": last["n_pitches"],
        "runs_pa": g["EVENT_RUNS_CT"].sum(),
        "r1": first["BASE1_RUN_ID"], "r2": first["BASE2_RUN_ID"], "r3": first["BASE3_RUN_ID"],
        "seq": first["event_seq"],
    }).reset_index(drop=True)
    pa = pa.sort_values("seq").reset_index(drop=True)
    hk = pa["game"] + "_" + pa["inning"].astype(str) + "_" + pa["half"].astype(str)
    tot = pa.groupby(hk)["runs_pa"].transform("sum")
    before = pa.groupby(hk)["runs_pa"].cumsum() - pa["runs_pa"]
    pa["runs_rest"] = tot - before
    pa["half_key"] = hk
    return pa


class DecisionBuilder:
    def __init__(self, stats, markov_M, ev2006, min_inning=6, dh_only=True):
        self.S = stats
        self.M = markov_M
        self.min_inning = min_inning
        self.pa = pa_table(ev2006)
        if dh_only:
            self.pa = self.pa[self.pa["home_team"].isin(AL_TEAMS)].reset_index(drop=True)
        app = stats.appearances
        self.app06 = app[app["year"] == 2006]
        self.team_dates = set(zip(self.app06["team"], self.app06["date"]))
        # position players by team and date (for bench)
        allpa = pa_table(ev2006)
        self.bat_days = allpa.groupby(["bat_team", "date"])["batter"].apply(set).to_dict()
        self._dera_cache, self._wl_cache, self._typ_cache, self._pool_cache = {}, {}, {}, {}
        self.lg = self._league_dict()

    def _league_dict(self):
        S = self.S
        lg_rates = S.lg_rates
        return {"rates": lg_rates, "avg": S.lg_avg, "obp": S.lg_obp, "era": S.lg_era,
                "sit_avg": S.lg_sit_avg, "dera": dera(S.lg_all, self.M), "avg_matchup": S.lg_avg_matchup}

    # ------------------------------------------------------------ cached enrichments
    def dera_vs(self, pid, date, pstats):
        key = (pid, date)
        if key not in self._dera_cache:
            self._dera_cache[key] = {h: dera(pstats["rates_vs"][h], self.M) for h in "LR"}
        return self._dera_cache[key]

    def workload(self, pid, date):
        key = (pid, date)
        if key not in self._wl_cache:
            self._wl_cache[key] = self.S.workload(pid, date)
        return self._wl_cache[key]

    def typical(self, pid, date, starter):
        key = (pid, date, starter)
        if key not in self._typ_cache:
            self._typ_cache[key] = self.S.typical_pitches(pid, date, starter)
        return self._typ_cache[key]

    def pitcher_alt(self, pid, date, is_current, ingame=None):
        p = dict(self.S.pitcher(pid, date))
        p["dera_vs"] = self.dera_vs(pid, date, p)
        p.update(self.workload(pid, date))
        p["is_current"] = is_current
        if is_current:
            p.update(ingame)
            p["typical_pitches"] = self.typical(pid, date, ingame["is_starter"])
        else:
            p["is_starter"] = False
        return p

    def reliever_pool(self, team, date, used):
        d = pd.Timestamp(date)
        key = (team, d)
        if key not in self._pool_cache:
            w = self.app06[(self.app06["team"] == team) & (self.app06["date"] < d) &
                           (self.app06["date"] >= d - pd.Timedelta(days=20))]
            starters = set(w.loc[w["started"], "pid"])
            recent = w[w["date"] >= d - pd.Timedelta(days=POOL_DAYS)]
            rel = sorted(set(recent.loc[~recent["started"], "pid"]) - starters)
            self._pool_cache[key] = [p for p in rel
                                     if not (self.workload(p, d)["pitched_d1"] and self.workload(p, d)["pitched_d2"])]
        return [p for p in self._pool_cache[key] if p not in used]

    def bench(self, team, date, used):
        d = pd.Timestamp(date)
        recent = set()
        for k in range(1, 11):
            recent |= self.bat_days.get((team, d - pd.Timedelta(days=k)), set())
        return sorted(recent - used)

    # ------------------------------------------------------------ main loop
    def decisions(self, game_ids=None, max_games=None):
        pa = self.pa
        games = pa["game"].unique() if game_ids is None else game_ids
        if max_games:
            games = games[:max_games]
        for gid in games:
            yield from self._game(pa[pa["game"] == gid])

    def _game(self, g):
        date = g["date"].iloc[0]
        slot_occ = {0: {}, 1: {}}          # batting half -> slot -> batter id
        used_bat = {0: set(), 1: set()}    # batting half -> players who appeared
        used_pit = {0: set(), 1: set()}    # FIELDING half index (0 = home fields in top) -> pitchers used
        cur_pit = {0: None, 1: None}
        ingame = defaultdict(lambda: {"pitches": 0, "bf": 0, "onbase": 0, "runs": 0, "started": False})
        rows = g.to_dict("records")
        for i, r in enumerate(rows):
            h = r["half"]            # batting side: 0 visitors bat (home fields), 1 home bats
            f = h
            actual = r["pitcher"]
            prev = cur_pit[f]
            changed = prev is not None and actual != prev
            sched = slot_occ[h].get(r["slot"], r["batter"])
            if r["inning"] >= self.min_inning and prev is not None and r["has_bat"]:
                yield self._decision(r, date, h, f, prev, actual, changed, sched, slot_occ, used_bat,
                                     used_pit, ingame)
            # ---- update game state with this PA
            if prev is None:
                ingame[actual]["started"] = True
            cur_pit[f] = actual
            used_pit[f].add(actual)
            used_bat[h].add(r["batter"])
            for rr in (r["r1"], r["r2"], r["r3"]):
                if rr:
                    used_bat[h].add(rr)
            slot_occ[h][r["slot"]] = r["batter"]
            st = ingame[actual]
            st["pitches"] += r["n_pitches"]
            st["bf"] += int(r["has_bat"])
            st["onbase"] += int(r["cat"] in ONBASE)
            st["runs"] += int(r["runs_pa"])

    def _decision(self, r, date, h, f, prev, actual, changed, sched, slot_occ, used_bat, used_pit, ingame):
        S = self.S
        st = ingame[prev]
        cur_ingame = {"pitches_today": st["pitches"], "bf_today": st["bf"], "onbase_today": st["onbase"],
                      "runs_today": st["runs"], "is_starter": st["started"], "tto": st["bf"] // 9}
        pool = self.reliever_pool(r["fld_team"], date, used_pit[f])
        forced = False
        if changed and actual not in pool:
            pool = pool + [actual]
            forced = True
        alts = [self.pitcher_alt(prev, date, True, cur_ingame)] + [self.pitcher_alt(p, date, False) for p in pool]
        # lineup from the scheduled batter onward (previous occupants: pre-pinch-hit)
        slot = int(r["slot"])
        order = [((slot - 1 + k) % 9) + 1 for k in range(9)]
        lineup_ids = [slot_occ[h].get(s) for s in order]
        lineup_ids[0] = sched
        lineup_ids = [x if x else sched for x in lineup_ids]
        lineup = [S.batter(b, date) for b in lineup_ids]
        bench_ids = self.bench(r["bat_team"], date, used_bat[h] | set(lineup_ids))
        bench = [S.batter(b, date) for b in bench_ids if S.rosters.get(b, {}).get("pos") != "P"]
        inn, outs = int(r["inning"]), int(r["outs"])
        fld_home = h == 0
        from player_stats import cell_of, SPLITS
        cells = {sp: cell_of(sp, fld_home, inn, outs, int(r["bases"]), int(r["fld_lead"])) for sp in SPLITS}
        sit = {
            "game": r["game"], "date": date, "inning": inn, "half": h, "outs": outs, "bases": int(r["bases"]),
            "fld_lead": int(r["fld_lead"]), "fld_home": fld_home, "fld_team": r["fld_team"],
            "bat_team": r["bat_team"], "batter": lineup[0], "on_deck": lineup[1], "next3": lineup[:3],
            "lineup": lineup, "bench": bench, "cells": cells,
            "plays_tomorrow": (r["fld_team"], date + pd.Timedelta(days=1)) in self.team_dates,
            "innings_left": (max(0, 9 - inn) + (3 - outs) / 3.0) if inn <= 9 else (3 - outs) / 3.0,
        }
        label = {"changed": changed, "prev": prev, "actual": actual, "forced": forced,
                 "runs_rest": int(r["runs_rest"]), "actual_batter": r["batter"], "sched_batter": sched,
                 "seq": int(r["seq"])}
        return sit, alts, label
