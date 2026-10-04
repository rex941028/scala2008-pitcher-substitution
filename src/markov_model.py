# -*- coding: utf-8 -*-
"""
Model 5 (benchmark) -- Markov-chain pitcher evaluation in the spirit of Hirotsu & Wright (2003, 2004,
2005), the prior work Scala (2008) positions the AHP against ("extremely long CPU run-times").

This is a deliberately SIMPLIFIED re-implementation, not H&W's full game-long dynamic program
(1,434,672 states).  What it keeps from H&W:
  * the half-inning as an absorbing Markov chain over the 24 base-out states
  * batter-specific and pitcher-specific outcome probabilities that depend on handedness
    (H&W 2005), combined with the odds-ratio (generalised log5) rule
  * DERA: expected runs a pitcher concedes over nine innings, conditional on batter hand
  * pitcher substitution evaluated by win probability
What it adds / simplifies:
  * base-out transitions are estimated empirically from 2005 Retrosheet data for each outcome
    type (includes double plays, sac flies, errors, and stolen bases / wild pitches that happen
    during the plate appearance) instead of fixed advancement rules
  * horizon = the rest of the current half-inning with the candidate pitcher, then a league-average
    win-expectancy table for the rest of the game (not an optimal policy for every later inning)
  * fatigue of the pitcher already in the game: times-through-the-order multipliers estimated
    from 2005 starters (observed / odds-ratio-expected outcome rates)
"""
import numpy as np
import pandas as pd

from common import CATS, CAT_INDEX

N_STATES = 24          # outs * 8 + bases (bases bitmask: 1 = 1B, 2 = 2B, 4 = 3B, as in cwevent)
ABSORB = 24
RMAX_PLAY = 4          # runs on a single PA
RMAX = 15              # half-inning run distribution support 0..RMAX (last bin = RMAX or more)


def state_index(outs, bases):
    return int(outs) * 8 + int(bases)


def group_plate_appearances(ev):
    """Collapse event records into plate appearances: start state = state at the first record of
    the PA, end state after the batter event, runs = all runs scored during the PA (incl. SB/WP)."""
    ev = ev.sort_values("event_seq")
    key = ev["GAME_ID"] + "_" + ev["INN_CT"].astype(str) + "_" + ev["half"].astype(str)
    prev_bat = ev.groupby(key)["BAT_EVENT_FL"].shift(1, fill_value=False).astype(int)
    pa_no = prev_bat.groupby(key).cumsum()
    g = ev.assign(_key=key, _pa=pa_no).groupby(["_key", "_pa"], sort=False)
    out = g.agg(start_outs=("OUTS_CT", "first"), start_bases=("START_BASES_CD", "first"),
                end_outs_before=("OUTS_CT", "last"), outs_on_last=("EVENT_OUTS_CT", "last"),
                end_bases=("END_BASES_CD", "last"), runs=("EVENT_RUNS_CT", "sum"),
                has_bat=("BAT_EVENT_FL", "last"), cat=("cat", "last"), inning=("INN_CT", "first"),
                half=("half", "first"), first_seq=("event_seq", "first"))
    out["end_outs"] = out["end_outs_before"] + out["outs_on_last"]
    return out.reset_index()


def estimate_transitions(ev, exclude_walkoff=True):
    """M[c, s, s', r] = P(next state s', r runs on the PA | state s, outcome category c)."""
    pa = group_plate_appearances(ev)
    pa = pa[pa["has_bat"] & pa["cat"].notna()]
    if exclude_walkoff:
        pa = pa[~((pa["half"] == 1) & (pa["inning"] >= 9))]
    counts = np.zeros((len(CATS), N_STATES, N_STATES + 1, RMAX_PLAY + 1))
    s = (pa["start_outs"].astype(int) * 8 + pa["start_bases"].astype(int)).values
    eo = pa["end_outs"].astype(int).values
    t = np.where(eo >= 3, ABSORB, eo * 8 + pa["end_bases"].astype(int).values)
    r = np.minimum(pa["runs"].astype(int).values, RMAX_PLAY)
    c = pa["cat"].map(CAT_INDEX).astype(int).values
    np.add.at(counts, (c, s, t, r), 1)
    # sparse cells (e.g. triple with bases loaded, 2 outs) borrow from the pooled-over-outs pattern
    probs = np.zeros_like(counts)
    for ci in range(len(CATS)):
        for si in range(N_STATES):
            tot = counts[ci, si].sum()
            if tot >= 30:
                probs[ci, si] = counts[ci, si] / tot
            else:
                probs[ci, si] = _fallback(ci, si, counts, tot)
    return probs


def _fallback(ci, si, counts, tot):
    """Blend sparse cells with a deterministic rule-of-thumb advancement so rows sum to 1."""
    outs, bases = divmod(si, 8)
    det = np.zeros((N_STATES + 1, RMAX_PLAY + 1))
    cat = CATS[ci]
    b = [(bases >> k) & 1 for k in range(3)]
    runners = sum(b)
    if cat == "HR":
        det[outs * 8 + 0, min(runners + 1, RMAX_PLAY)] = 1
    elif cat == "3B":
        det[outs * 8 + 4, min(runners, RMAX_PLAY)] = 1
    elif cat == "2B":
        det[outs * 8 + 2 + (4 if b[0] else 0), min(b[1] + b[2], RMAX_PLAY)] = 1
    elif cat == "1B":
        nb = 1 | (2 if b[0] else 0) | (4 if b[1] else 0)
        det[outs * 8 + nb, min(b[2], RMAX_PLAY)] = 1
    elif cat == "BB":  # forced advancement only
        runs = 0
        if not b[0]:
            nb = bases | 1
        elif not b[1]:
            nb = bases | 3
        elif not b[2]:
            nb = 7
        else:
            nb, runs = 7, 1
        det[outs * 8 + nb, runs] = 1
    else:  # K / OIP: batter out, runners hold
        det[ABSORB if outs == 2 else (outs + 1) * 8 + bases, 0] = 1
    w = tot / (tot + 30.0)
    emp = counts[ci, si] / tot if tot > 0 else 0
    return w * emp + (1 - w) * det


def pa_transition(q, M):
    """Mix transition tensors by outcome probabilities q (7,) -> (24, 25, 5)."""
    return np.tensordot(q, M, axes=1)


def run_distribution(outs, bases, lineup_probs, M, max_pa=40):
    """P(runs scored in the rest of the half-inning = r), r = 0..RMAX.
    lineup_probs: list of outcome-probability vectors for the upcoming batters in order
    (cycled if the half-inning goes past its end)."""
    D = np.zeros((N_STATES, RMAX + 1))
    D[state_index(outs, bases), 0] = 1.0
    final = np.zeros(RMAX + 1)
    k = len(lineup_probs)
    cache = {}
    for i in range(max_pa):
        j = i % k
        if j not in cache:
            cache[j] = pa_transition(lineup_probs[j], M)
        T = cache[j]
        newD = np.zeros_like(D)
        for r in range(RMAX_PLAY + 1):
            moved = np.einsum("sR,st->tR", D, T[:, :, r])   # (25, RMAX+1)
            if r:
                shifted = np.zeros_like(moved)
                shifted[:, r:] = moved[:, :-r]
                shifted[:, -1] += moved[:, -r:].sum(axis=1)
                moved = shifted
            newD += moved[:N_STATES]
            final += moved[ABSORB]
        D = newD
        if D.sum() < 1e-10:
            break
    return final / final.sum()


def expected_runs(dist):
    return float(np.dot(np.arange(len(dist)), dist))


def dera(rates, M):
    """H&W's DERA: expected runs conceded over nine innings when every batter produces `rates`."""
    return 9.0 * expected_runs(run_distribution(0, 0, [rates], M))


class WinExpectancy:
    """League-average win expectancy from half-inning run distributions (2005)."""

    def __init__(self, ev, max_inning=30, dmax=30):
        pa = group_plate_appearances(ev)
        half_runs = pa.groupby(["_key"]).agg(runs=("runs", "sum"), inning=("inning", "first"),
                                             half=("half", "first"))
        top = half_runs[half_runs["half"] == 0]["runs"].clip(upper=RMAX)
        bot = half_runs[(half_runs["half"] == 1) & (half_runs["inning"] < 9)]["runs"].clip(upper=RMAX)
        self.f_top = np.bincount(top, minlength=RMAX + 1) / len(top)
        self.f_bot = np.bincount(bot, minlength=RMAX + 1) / len(bot)
        self.max_inning, self.dmax = max_inning, dmax
        self._memo = {}

    def home_wp(self, inning, half, d):
        """P(home wins) at the START of half-inning (inning, half) with home lead d."""
        d = int(max(-self.dmax, min(self.dmax, d)))
        key = (inning, half, d)
        if key in self._memo:
            return self._memo[key]
        if inning > self.max_inning:
            v = 1.0 if d > 0 else (0.0 if d < 0 else 0.5)
        elif half == 1 and inning >= 9 and d > 0:
            v = 1.0
        elif half == 0:
            v = sum(p * self.home_wp(inning, 1, d - r) for r, p in enumerate(self.f_top) if p > 0)
        else:
            v = sum(p * self.after_bottom(inning, d + r) for r, p in enumerate(self.f_bot) if p > 0)
        self._memo[key] = v
        return v

    def after_bottom(self, inning, d):
        if inning >= 9:
            if d > 0:
                return 1.0
            if d < 0:
                return 0.0
        return self.home_wp(inning + 1, 0, d)

    def after_half(self, inning, half, home_lead_after):
        """P(home wins) right after the current half-inning ends with the given home lead."""
        if half == 0:
            return self.home_wp(inning, 1, home_lead_after)
        return self.after_bottom(inning, home_lead_after)

    def fielding_wp(self, inning, half, fld_lead, dist):
        """Fielding team's win probability given the distribution of runs it allows in the rest of
        the half-inning. fld_lead = fielding team's lead now."""
        fld_home = half == 0
        wp = 0.0
        for r, p in enumerate(dist):
            if p <= 0:
                continue
            lead_after = fld_lead - r
            home_lead = lead_after if fld_home else -lead_after
            h = self.after_half(inning, half, home_lead)
            wp += p * (h if fld_home else 1.0 - h)
        return wp


def evaluate_alternatives(sit, alts, lg, M, WE, tto_mult=None):
    """For each alternative pitcher: run distribution for the rest of this half-inning facing the
    scheduled lineup, expected runs, P(no run), and the fielding team's win probability."""
    from common import effective_bat_hand
    from player_stats import matchup_rates
    out = []
    for a in alts:
        t = a["throws"]
        mult = None
        if tto_mult is not None and a["is_current"] and a.get("is_starter"):
            mult = tto_mult[min(int(a.get("tto", 0)), 3)]
        probs = []
        for b in sit["lineup"]:
            eff = effective_bat_hand(b["bats"], t)
            q = matchup_rates(b["rates_vs"][t], a["rates_vs"][eff], lg["rates"][(eff, t)])
            if mult is not None:
                q = q * mult
                q = q / q.sum()
            probs.append(q)
        dist = run_distribution(sit["outs"], sit["bases"], probs, M)
        wp = WE.fielding_wp(sit["inning"], sit["half"], sit["fld_lead"], dist)
        out.append({"id": a["id"], "exp_runs": expected_runs(dist), "p0": float(dist[0]), "wp": wp})
    best = max(out, key=lambda x: x["wp"])
    cur = [a["id"] for a in alts if a["is_current"]]
    return {"alts": out, "best": best["id"], "current": cur[0] if cur else None,
            "substitute": best["id"] != (cur[0] if cur else None)}


def estimate_tto_multipliers(ev, stats_engine):
    """Observed / expected outcome rates for STARTERS by times through the order (1st, 2nd, 3rd,
    4th+), expected from the odds-ratio rule with each player's full-season 2005 rates (in-sample,
    which removes the selection effect that only good starters go deep)."""
    ev = ev[ev["BAT_EVENT_FL"] & ev["cat"].notna()].copy()
    ev = ev.sort_values("event_seq")
    ev["bf_before"] = ev.groupby(["GAME_ID", "PIT_ID"]).cumcount()
    starters = ev[ev["PIT_START_FL"]]
    obs = np.zeros((4, len(CATS)))
    exp = np.zeros((4, len(CATS)))
    date_end = pd.Timestamp("2006-01-01")
    from player_stats import matchup_rates
    lg = stats_engine.lg_rates
    for (pid, bid, bh, ph, cat, bf) in zip(starters["RESP_PIT_ID"], starters["RESP_BAT_ID"],
                                          starters["RESP_BAT_HAND_CD"], starters["RESP_PIT_HAND_CD"],
                                          starters["cat"], starters["bf_before"]):
        tto = min(bf // 9, 3)
        p = stats_engine.pitcher(pid, date_end)["rates_vs"][bh]
        b = stats_engine.batter(bid, date_end)["rates_vs"][ph]
        exp[tto] += matchup_rates(b, p, lg[(bh, ph)])
        obs[tto, CAT_INDEX[cat]] += 1
    mult = obs / exp
    return mult, obs.sum(axis=1)
