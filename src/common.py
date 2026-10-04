# -*- coding: utf-8 -*-
"""
Shared paths, Retrosheet loading and small helpers used by every model and experiment.

Data: Retrosheet play-by-play (2005, 2006) converted with Chadwick `cwevent` into
data/processed/events_YYYY.csv (see data_prep.py), plus the Retrosheet roster files (bats/throws).
"""
import os
import glob

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROCESSED = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "results")

# Retrosheet EVENT_CD -> the seven PA outcome categories used by the Markov model and the stats engine.
#   2 generic out, 3 K, 14 BB, 15 IBB, 16 HBP, 17 interference, 18 error, 19 fielder's choice,
#   20 1B, 21 2B, 22 3B, 23 HR.  Errors / FC / interference fold into "OIP" (out-in-play family):
#   their base-out transitions are estimated empirically, so a reach-on-error is still modelled.
CATS = ["K", "BB", "1B", "2B", "3B", "HR", "OIP"]
CAT_INDEX = {c: i for i, c in enumerate(CATS)}
EVENT_TO_CAT = {3: "K", 14: "BB", 15: "BB", 16: "BB", 20: "1B", 21: "2B", 22: "3B", 23: "HR",
                2: "OIP", 17: "OIP", 18: "OIP", 19: "OIP"}
HIT_CATS = ["1B", "2B", "3B", "HR"]

# Characters in PITCH_SEQ_TX that are real pitches (pickoff throws, runner-going marks etc. excluded).
PITCH_CHARS = set("BCFHIKLMOPQRSTUVXY")


def count_pitches(seq):
    if not isinstance(seq, str):
        return 0
    return sum(1 for ch in seq if ch in PITCH_CHARS)


def load_rosters(year):
    """player_id -> dict(name, bats, throws, team). Players traded mid-season appear in several
    team files; bats/throws are identical so the last one read wins."""
    rows = {}
    for fn in glob.glob(os.path.join(RAW, f"{year}eve", f"*{year}.ROS")):
        with open(fn, encoding="latin-1") as fh:
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                pid, last, first, bats, throws, team, pos = parts[:7]
                rows[pid] = {"name": f"{first} {last}", "bats": bats, "throws": throws,
                             "team": team, "pos": pos}
    return rows


def load_all_rosters():
    out = {}
    for y in (2005, 2006):
        out.update(load_rosters(y))
    return out


def load_events(year):
    """Load cwevent output and add convenience columns:
    date, half (0 top / 1 bottom), fielding-team score diff, PA grouping, category, pitches."""
    fn = os.path.join(PROCESSED, f"events_{year}.csv")
    if not os.path.exists(fn):          # the repository ships the gzipped copy
        fn += ".gz"
    df = pd.read_csv(fn, low_memory=False, keep_default_na=False, na_values=[])
    df["year"] = year
    df["date"] = pd.to_datetime(df["GAME_ID"].str[3:11], format="%Y%m%d")
    df["half"] = df["BAT_HOME_ID"].astype(int)
    for c in ["BASE1_RUN_ID", "BASE2_RUN_ID", "BASE3_RUN_ID"]:
        df[c] = df[c].astype(str)
    bool_cols = ["LEADOFF_FL", "PH_FL", "BAT_EVENT_FL", "AB_FL", "SH_FL", "SF_FL", "GAME_NEW_FL",
                 "GAME_END_FL", "INN_NEW_FL", "INN_END_FL", "BAT_START_FL", "PIT_START_FL"]
    for c in bool_cols:
        df[c] = df[c].astype(str).eq("T")
    df["cat"] = df["EVENT_CD"].map(EVENT_TO_CAT)
    df["n_pitches"] = df["PITCH_SEQ_TX"].map(count_pitches)
    # fielding team's run differential at the start of the play (positive = fielding team ahead)
    df["fld_diff"] = df["START_FLD_SCORE_CT"] - df["START_BAT_SCORE_CT"]
    df["event_seq"] = np.arange(len(df))
    return df


def md_table(df, index=False):
    """DataFrame -> GitHub markdown table (avoids the optional `tabulate` dependency)."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in d.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def league_of(team):
    """American League teams in 2005-2006 (DH rule in their parks)."""
    return "AL" if team in AL_TEAMS else "NL"


AL_TEAMS = {"ANA", "BAL", "BOS", "CHA", "CLE", "DET", "KCA", "MIN", "NYA", "OAK", "SEA", "TBA", "TEX", "TOR"}


def effective_bat_hand(bats, pitcher_throws):
    """Switch hitters bat from the side opposite the pitcher's throwing arm."""
    if bats == "B":
        return "L" if pitcher_throws == "R" else "R"
    return bats


def platoon_advantage_for_pitcher(bats, pitcher_throws):
    """+1 if the pitcher has the platoon edge (same side), -1 otherwise (switch hitters never give it)."""
    return 1 if effective_bat_hand(bats, pitcher_throws) == pitcher_throws else -1
