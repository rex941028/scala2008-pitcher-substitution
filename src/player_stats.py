# -*- coding: utf-8 -*-
"""
Walk-forward player statistics (no look-ahead): every query `pitcher(pid, date)` / `batter(pid, date)`
uses only plate appearances from 2005 plus 2006 games played strictly BEFORE `date`.

League baselines (event rates by batter-hand x pitcher-hand, situational averages, ERA) come from
2005 only, so nothing from the 2006 test season leaks into the priors.

Shrinkage ("regression to the mean") -- all estimates are Bayesian-style (n*obs + k*prior)/(n + k):
  pitcher overall rate per outcome, k in PA: K 70, BB 170, 1B 670, 2B 1450, 3B 1450, HR 1300
  batter  overall rate per outcome, k in PA: K 60, BB 120, 1B 290, 2B 1610, 3B 1610, HR 170
  platoon split (vs L / vs R) shrunk to overall x league platoon ratio with k = 1000 PA
  situational AVG splits (home/away, inning, outs, runners, score) shrunk with k = 800 AB
  ERA shrunk toward league ERA with k = 60 IP
The OIP ("out in play") share is the remainder so each rate vector sums to 1.
"""
import numpy as np
import pandas as pd

from common import CATS, CAT_INDEX, HIT_CATS, load_events, load_all_rosters

K_PIT = np.array([70, 170, 670, 1450, 1450, 1300, np.nan])
K_BAT = np.array([60, 120, 290, 1610, 1610, 170, np.nan])
K_SPLIT = 1000.0
K_SIT = 800.0
K_ERA_IP = 60.0

SPLITS = {
    "homeaway": ["home", "away"],
    "inning": ["inn1_3", "inn4_6", "inn7p"],
    "outs": ["outs0", "outs1", "outs2"],
    "runners": ["empty", "on"],
    "score": ["ahead", "tied", "behind"],
    "close": ["close", "notclose"],
}
SPLIT_CELLS = [c for cells in SPLITS.values() for c in cells]


def cell_of(split, fld_home, inning, outs, bases, fld_diff):
    if split == "homeaway":
        return "home" if fld_home else "away"
    if split == "inning":
        return "inn1_3" if inning <= 3 else ("inn4_6" if inning <= 6 else "inn7p")
    if split == "outs":
        return f"outs{min(int(outs), 2)}"
    if split == "runners":
        return "empty" if bases == 0 else "on"
    if split == "score":
        return "ahead" if fld_diff > 0 else ("tied" if fld_diff == 0 else "behind")
    if split == "close":
        return "close" if abs(fld_diff) <= 1 else "notclose"
    raise ValueError(split)


def rates_to_avg(r):
    """Batting average implied by an outcome-rate vector (AB ~ PA - BB/HBP)."""
    h = sum(r[CAT_INDEX[c]] for c in HIT_CATS)
    return h / max(1e-9, 1.0 - r[CAT_INDEX["BB"]])


def rates_to_obp(r):
    return sum(r[CAT_INDEX[c]] for c in HIT_CATS) + r[CAT_INDEX["BB"]]


def _shrink_vector(counts, n, prior, k):
    """counts (7,), n PA, prior rates (7,), k (7,) with NaN for the remainder category."""
    est = np.empty(7)
    for i in range(6):
        est[i] = (counts[i] + k[i] * prior[i]) / (n + k[i])
    est[6] = max(0.05, 1.0 - est[:6].sum())
    return est / est.sum()


def _pa_table(ev):
    pa = ev[ev["BAT_EVENT_FL"] & ev["cat"].notna()].copy()
    pa["bhand"] = pa["RESP_BAT_HAND_CD"]
    pa["phand"] = pa["RESP_PIT_HAND_CD"]
    pa["fld_home"] = pa["half"] == 0
    pa["hit"] = (pa["H_CD"] > 0).astype(int)
    pa["ab"] = pa["AB_FL"].astype(int)
    return pa


class StatsEngine:
    def __init__(self, ev2005=None, ev2006=None, rosters=None):
        self.ev2005 = load_events(2005) if ev2005 is None else ev2005
        self.ev2006 = load_events(2006) if ev2006 is None else ev2006
        self.rosters = load_all_rosters() if rosters is None else rosters
        ev = pd.concat([self.ev2005, self.ev2006], ignore_index=True)
        self.pa = _pa_table(ev)
        self._league(self.pa[self.pa["year"] == 2005], self.ev2005)
        self._build_pitcher_tables(ev)
        self._build_batter_tables()
        self._build_appearances(ev)
        self._cache = {}

    # ------------------------------------------------------------------ league (2005 only)
    def _league(self, pa05, ev05):
        self.lg_rates = {}
        for bh in "LR":
            for ph in "LR":
                sub = pa05[(pa05["bhand"] == bh) & (pa05["phand"] == ph)]
                cnt = sub["cat"].value_counts()
                v = np.array([cnt.get(c, 0) for c in CATS], dtype=float)
                self.lg_rates[(bh, ph)] = v / v.sum()
        cnt = pa05["cat"].value_counts()
        v = np.array([cnt.get(c, 0) for c in CATS], dtype=float)
        self.lg_all = v / v.sum()
        # league rates by pitcher hand (batter mix as observed) and by batter hand
        self.lg_by_phand, self.lg_by_bhand = {}, {}
        for h in "LR":
            c1 = pa05[pa05["phand"] == h]["cat"].value_counts()
            v1 = np.array([c1.get(c, 0) for c in CATS], dtype=float)
            self.lg_by_phand[h] = v1 / v1.sum()
            c2 = pa05[pa05["bhand"] == h]["cat"].value_counts()
            v2 = np.array([c2.get(c, 0) for c in CATS], dtype=float)
            self.lg_by_bhand[h] = v2 / v2.sum()
        self.lg_avg = pa05["hit"].sum() / pa05["ab"].sum()
        self.lg_avg_matchup = {}
        for bh in "LR":
            for ph in "LR":
                s = pa05[(pa05["bhand"] == bh) & (pa05["phand"] == ph)]
                self.lg_avg_matchup[(bh, ph)] = s["hit"].sum() / s["ab"].sum()
        self.lg_obp = rates_to_obp(self.lg_all)
        self.lg_sit_avg = {}
        for split in SPLITS:
            cells = pa05.apply(lambda r: cell_of(split, r["fld_home"], r["INN_CT"], r["OUTS_CT"],
                                                  r["START_BASES_CD"], r["fld_diff"]), axis=1)
            for cell in SPLITS[split]:
                m = cells == cell
                self.lg_sit_avg[cell] = pa05.loc[m, "hit"].sum() / pa05.loc[m, "ab"].sum()
        er, outs = self._er_outs(ev05)
        self.lg_era = 9.0 * er["ER"].sum() / (outs["OUTS"].sum() / 3.0)

    @staticmethod
    def _er_outs(ev):
        parts = []
        for dest, resp in [("BAT_DEST_ID", "RESP_PIT_ID"), ("RUN1_DEST_ID", "RUN1_RESP_PIT_ID"),
                           ("RUN2_DEST_ID", "RUN2_RESP_PIT_ID"), ("RUN3_DEST_ID", "RUN3_RESP_PIT_ID")]:
            m = ev[dest] == 4
            parts.append(pd.DataFrame({"pid": ev.loc[m, resp], "date": ev.loc[m, "date"], "ER": 1}))
        er = pd.concat(parts).groupby(["pid", "date"], as_index=False)["ER"].sum()
        outs = ev.groupby(["PIT_ID", "date"], as_index=False)["EVENT_OUTS_CT"].sum()
        outs.columns = ["pid", "date", "OUTS"]
        return er, outs

    # ------------------------------------------------------------------ cumulative tables
    @staticmethod
    def _cumulate(df, key, cols):
        """df has columns key, date, *cols (daily sums). Returns {id: (dates ndarray, cum ndarray)}."""
        out = {}
        df = df.sort_values([key, "date"])
        for pid, g in df.groupby(key, sort=False):
            out[pid] = (g["date"].values, np.cumsum(g[cols].values.astype(float), axis=0))
        return out

    @staticmethod
    def _lookup(table, pid, date, ncols):
        if pid not in table:
            return np.zeros(ncols)
        dates, cum = table[pid]
        i = np.searchsorted(dates, np.datetime64(date), side="left") - 1
        return cum[i] if i >= 0 else np.zeros(ncols)

    def _build_pitcher_tables(self, ev):
        pa = self.pa
        cols = []
        daily = pa[["RESP_PIT_ID", "date"]].copy()
        for h in "LR":
            m = pa["bhand"] == h
            daily[f"PA_{h}"] = m.astype(int)
            cols.append(f"PA_{h}")
            for c in CATS:
                daily[f"{c}_{h}"] = (m & (pa["cat"] == c)).astype(int)
                cols.append(f"{c}_{h}")
            daily[f"AB_{h}"] = pa["ab"] * m
            daily[f"H_{h}"] = pa["hit"] * m
            cols += [f"AB_{h}", f"H_{h}"]
        for split, cells in SPLITS.items():
            cell = [cell_of(split, a, b, c, d, e) for a, b, c, d, e in
                    zip(pa["fld_home"], pa["INN_CT"], pa["OUTS_CT"], pa["START_BASES_CD"], pa["fld_diff"])]
            cell = np.array(cell)
            for cname in cells:
                m = cell == cname
                daily[f"AB_{cname}"] = pa["ab"].values * m
                daily[f"H_{cname}"] = pa["hit"].values * m
                cols += [f"AB_{cname}", f"H_{cname}"]
        daily = daily.groupby(["RESP_PIT_ID", "date"], as_index=False)[cols].sum()
        daily = daily.rename(columns={"RESP_PIT_ID": "pid"})
        self.p_cols = cols
        self.p_col_index = {c: i for i, c in enumerate(cols)}
        self.p_table = self._cumulate(daily, "pid", cols)
        er, outs = self._er_outs(ev)
        eo = pd.merge(er, outs, on=["pid", "date"], how="outer").fillna(0)
        self.era_table = self._cumulate(eo, "pid", ["ER", "OUTS"])

    def _build_batter_tables(self):
        pa = self.pa
        daily = pa[["RESP_BAT_ID", "date"]].copy()
        cols = []
        for t in "LR":
            m = pa["phand"] == t
            daily[f"PA_{t}"] = m.astype(int)
            cols.append(f"PA_{t}")
            for c in CATS:
                daily[f"{c}_{t}"] = (m & (pa["cat"] == c)).astype(int)
                cols.append(f"{c}_{t}")
            daily[f"AB_{t}"] = pa["ab"] * m
            daily[f"H_{t}"] = pa["hit"] * m
            cols += [f"AB_{t}", f"H_{t}"]
        daily = daily.groupby(["RESP_BAT_ID", "date"], as_index=False)[cols].sum()
        daily = daily.rename(columns={"RESP_BAT_ID": "pid"})
        self.b_cols = cols
        self.b_col_index = {c: i for i, c in enumerate(cols)}
        self.b_table = self._cumulate(daily, "pid", cols)

    def _build_appearances(self, ev):
        """One row per pitcher appearance: date, game, team, started, pitches, BF, outs."""
        bat = ev[ev["BAT_EVENT_FL"]]
        trunc = ev[ev["INN_END_FL"] & ~ev["BAT_EVENT_FL"]]
        pitches = pd.concat([bat[["GAME_ID", "PIT_ID", "n_pitches"]], trunc[["GAME_ID", "PIT_ID", "n_pitches"]]])
        pitches = pitches.groupby(["GAME_ID", "PIT_ID"])["n_pitches"].sum()
        bf = bat.groupby(["GAME_ID", "PIT_ID"]).size()
        outs = ev.groupby(["GAME_ID", "PIT_ID"])["EVENT_OUTS_CT"].sum()
        first = ev.groupby(["GAME_ID", "PIT_ID"]).agg(date=("date", "first"), team=("FLD_TEAM_ID", "first"),
                                                     started=("PIT_START_FL", "max"),
                                                     first_seq=("event_seq", "min"))
        app = first.join(pitches.rename("pitches")).join(bf.rename("bf")).join(outs.rename("outs")).fillna(0)
        app = app.reset_index().rename(columns={"PIT_ID": "pid"})
        app["year"] = app["date"].dt.year
        self.appearances = app.sort_values(["pid", "date", "first_seq"]).reset_index(drop=True)
        self._app_by_pid = {pid: g for pid, g in self.appearances.groupby("pid")}

    # ------------------------------------------------------------------ queries
    def hand(self, pid, kind):
        r = self.rosters.get(pid)
        if r is None:
            return "R"
        return r["throws"] if kind == "throws" else r["bats"]

    def name(self, pid):
        r = self.rosters.get(pid)
        return r["name"] if r else pid

    def pitcher(self, pid, date):
        key = ("P", pid, pd.Timestamp(date))
        if key in self._cache:
            return self._cache[key]
        throws = self.hand(pid, "throws")
        v = self._lookup(self.p_table, pid, date, len(self.p_cols))
        ix = self.p_col_index
        cnt = {h: np.array([v[ix[f"{c}_{h}"]] for c in CATS]) for h in "LR"}
        n = {h: v[ix[f"PA_{h}"]] for h in "LR"}
        tot_cnt, tot_n = cnt["L"] + cnt["R"], n["L"] + n["R"]
        prior_all = self.lg_by_phand[throws]
        overall = _shrink_vector(tot_cnt, tot_n, prior_all, K_PIT)
        talent = overall / prior_all          # pitcher's multiplicative talent vs league
        rates_vs = {}
        for h in "LR":
            prior_h = self.lg_rates[(h, throws)] * talent
            prior_h = prior_h / prior_h.sum()
            est = (cnt[h] + K_SPLIT * prior_h) / (n[h] + K_SPLIT)
            rates_vs[h] = est / est.sum()
        # situational AVG splits, shrunk toward the pitcher's own overall AVG shifted by league cell effect
        avg_all = rates_to_avg(overall)
        sit = {}
        for cname in SPLIT_CELLS:
            ab, hh = v[ix[f"AB_{cname}"]], v[ix[f"H_{cname}"]]
            prior = avg_all + (self.lg_sit_avg[cname] - self.lg_avg)
            sit[cname] = (hh + K_SIT * prior) / (ab + K_SIT)
        eo = self._lookup(self.era_table, pid, date, 2)
        ip = eo[1] / 3.0
        era = 9.0 * (eo[0] + K_ERA_IP * self.lg_era / 9.0) / (ip + K_ERA_IP)
        out = {
            "id": pid, "name": self.name(pid), "throws": throws, "pa": tot_n, "ip": ip,
            "rates_all": overall, "rates_vs": rates_vs,
            "avg_all": avg_all, "avg_vs": {h: rates_to_avg(rates_vs[h]) for h in "LR"},
            "obp_all": rates_to_obp(overall), "k_rate": overall[CAT_INDEX["K"]],
            "sit_avg": sit, "era": era, "raw_era": 9.0 * eo[0] / ip if ip > 0 else np.nan,
        }
        self._cache[key] = out
        return out

    def batter(self, pid, date):
        key = ("B", pid, pd.Timestamp(date))
        if key in self._cache:
            return self._cache[key]
        bats = self.hand(pid, "bats")
        v = self._lookup(self.b_table, pid, date, len(self.b_cols))
        ix = self.b_col_index
        cnt = {t: np.array([v[ix[f"{c}_{t}"]] for c in CATS]) for t in "LR"}
        n = {t: v[ix[f"PA_{t}"]] for t in "LR"}
        tot_cnt, tot_n = cnt["L"] + cnt["R"], n["L"] + n["R"]
        base_hand = "L" if bats == "L" else "R"
        prior_all = self.lg_by_bhand[base_hand] if bats != "B" else self.lg_all
        overall = _shrink_vector(tot_cnt, tot_n, prior_all, K_BAT)
        talent = overall / prior_all
        rates_vs = {}
        for t in "LR":
            eff = bats if bats != "B" else ("L" if t == "R" else "R")
            prior_t = self.lg_rates[(eff, t)] * talent
            prior_t = prior_t / prior_t.sum()
            est = (cnt[t] + K_SPLIT * prior_t) / (n[t] + K_SPLIT)
            rates_vs[t] = est / est.sum()
        out = {"id": pid, "name": self.name(pid), "bats": bats, "pa": tot_n,
               "rates_all": overall, "rates_vs": rates_vs,
               "avg_all": rates_to_avg(overall), "avg_vs": {t: rates_to_avg(rates_vs[t]) for t in "LR"}}
        self._cache[key] = out
        return out

    def appearances_before(self, pid, date):
        g = self._app_by_pid.get(pid)
        if g is None:
            return g
        return g[g["date"] < pd.Timestamp(date)]

    def typical_pitches(self, pid, date, as_starter):
        g = self.appearances_before(pid, date)
        default = 95.0 if as_starter else 20.0
        if g is None or len(g) == 0:
            return default
        g = g[g["started"] == as_starter]
        if len(g) < 3:
            return default
        return float(g["pitches"].tail(30).mean())

    def workload(self, pid, date):
        """days_rest (0 = pitched yesterday), pitches and appearances over the previous 3 days,
        mean outs per relief appearance (role length)."""
        d = pd.Timestamp(date)
        g = self.appearances_before(pid, d)
        if g is None or len(g) == 0:
            return {"days_rest": 9, "pitches_last3": 0, "apps_last3": 0, "outs_per_relief": 3.0,
                    "pitched_d1": False, "pitched_d2": False}
        last = g["date"].max()
        recent = g[g["date"] >= d - pd.Timedelta(days=3)]
        rel = g[(~g["started"]) & (g["date"] >= d - pd.Timedelta(days=365))]
        dates = set(g["date"][g["date"] >= d - pd.Timedelta(days=2)])
        return {"days_rest": int((d - last).days) - 1,
                "pitches_last3": float(recent["pitches"].sum()),
                "apps_last3": int(len(recent)),
                "outs_per_relief": float(rel["outs"].mean()) if len(rel) else 3.0,
                "pitched_d1": (d - pd.Timedelta(days=1)) in dates,
                "pitched_d2": (d - pd.Timedelta(days=2)) in dates}


def matchup_rates(bat_rates, pit_rates, lg):
    """Generalised log5 / odds-ratio method (Tango): p_i proportional to b_i * p_i / l_i."""
    x = bat_rates * pit_rates / lg
    return x / x.sum()
