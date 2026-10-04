# -*- coding: utf-8 -*-
"""
E1 -- reproduce the paper's illustrative example (Scala 2008, p. 5):
  Boston 8 @ Cleveland 6, 2006-04-25, Jacobs Field. Westbrook (CLE starter) left after the 5th with
  CLE leading 4-2; first batter of the top of the 6th = Mike Lowell (RHB).
  Alternatives: leave Westbrook in / Jason Davis (RHP) / Scott Sauerbeck (LHP).
  Paper: AHP picks Jason Davis, which is what Eric Wedge did.

Stats are what Cleveland could know that night: 2005 + 2006 through 2006-04-24 (walk-forward).
Also runs the Markov benchmark on the same decision, and the AHP with the whole available bullpen.
Output: results/e1_case_study.md and results/e1_case_study.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import numpy as np

from common import RESULTS
from context import Context
from criteria_weights import CRITERIA, cluster_totals, criteria_weights
from influence_diagram import NODES
from markov_model import evaluate_alternatives
from pitcher_ahp import evaluate
from situation import DecisionBuilder

GAME = "CLE200604250"
PAPER_ALTS = ["westj001", "davij005", "saues001"]


def find_decision(ctx):
    db = DecisionBuilder(ctx.stats, ctx.M, ctx.ev2006, min_inning=6)
    for sit, alts, label in db.decisions(game_ids=[GAME]):
        if sit["inning"] == 6 and sit["half"] == 0:
            return db, sit, alts, label
    raise RuntimeError("decision not found")


def paper_alternatives(db, sit, alts):
    have = {a["id"]: a for a in alts}
    out = []
    for pid in PAPER_ALTS:
        if pid in have:
            out.append(have[pid])
        else:
            out.append(db.pitcher_alt(pid, sit["date"], False))
    return out


def fmt_pugh(res, names):
    lines = []
    for c in CRITERIA:
        tab = res["pugh"].table(c)
        lines.append(f"\n**{c}**\n")
        lines.append("| 子準則 | " + " | ".join(names) + " |")
        lines.append("|---|" + "---|" * len(names))
        for row in tab:
            sub = row[0]
            label = sub if sub == "Total" else f"{sub}. {NODES[sub][1]}"
            lines.append(f"| {label} | " + " | ".join(f"{v:+d}" if isinstance(v, (int, np.integer)) else str(v)
                                                    for v in row[1:]) + " |")
    return "\n".join(lines)


def main():
    ctx = Context()
    db, sit, alts, label = find_decision(ctx)
    S = ctx.stats
    out = {"situation": {k: sit[k] for k in ("game", "inning", "half", "outs", "bases", "fld_lead",
                                             "plays_tomorrow", "innings_left")},
           "batter": sit["batter"]["name"], "on_deck": sit["on_deck"]["name"],
           "actual_change": label["changed"], "actual_pitcher": S.name(label["actual"]),
           "runs_rest_of_inning": label["runs_rest"]}
    out["situation"]["date"] = str(sit["date"].date())
    palts = paper_alternatives(db, sit, alts)
    names = [a["name"] for a in palts]
    md = ["# E1：重現論文案例（2006-04-25 BOS @ CLE，6 局上，Lowell 打擊）\n",
          f"- 場上：CLE 領先 {sit['fld_lead']} 分，{sit['outs']} 出局，壘包代碼 {sit['bases']}，"
          f"打者 {sit['batter']['name']}（{sit['batter']['bats']}），下一棒 {sit['on_deck']['name']}（{sit['on_deck']['bats']}）",
          f"- 板凳：{', '.join(b['name'] + '(' + b['bats'] + ')' for b in sit['bench'])}",
          f"- Westbrook 今日：{palts[0]['pitches_today']} 球、{palts[0]['bf_today']} 打席、上壘 {palts[0]['onbase_today']}、"
          f"失分 {palts[0]['runs_today']}；平常先發用球 {palts[0]['typical_pitches']:.0f}",
          f"- 實際：{S.name(label['actual'])} 上場；本半局失 {label['runs_rest']} 分\n",
          "## 候選投手（賽前可得數據，已做收縮）\n",
          "| 投手 | 投 | ERA(收縮) | 被打擊率 vs L | vs R | 被上壘率 | DERA vs L | DERA vs R | 休息天數 | 近3日用球 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for a in palts:
        md.append(f"| {a['name']} | {a['throws']} | {a['era']:.2f} | {a['avg_vs']['L']:.3f} | {a['avg_vs']['R']:.3f} | "
                  f"{a['obp_all']:.3f} | {a['dera_vs']['L']:.2f} | {a['dera_vs']['R']:.2f} | {a['days_rest']} | {a['pitches_last3']:.0f} |")
    runs = {}
    for rule in ("paper", "round", "exact"):
        for mode in ("distributive", "ideal"):
            r = evaluate(sit, palts, db.lg, rule=rule, mode=mode)
            runs[f"{rule}/{mode}"] = r
    base = runs["paper/distributive"]
    md.append("\n## Pugh chart（本模型的 +1/0/-1 評分）")
    md.append(fmt_pugh(base, names))
    w, cr, _ = criteria_weights(cluster_totals(), "paper")
    md.append("\n## AHP 結果（基準：論文的四捨五入規則、distributive 合成）\n")
    md.append("準則權重：" + "，".join(f"{c} {w[c]:.3f}" for c in CRITERIA) + f"（CR = {cr:.3f}）\n")
    md.append("| 準則 | " + " | ".join(names) + " | CR |")
    md.append("|---|" + "---|" * (len(names) + 1))
    for c in CRITERIA:
        md.append(f"| {c} | " + " | ".join(f"{base['local'][c][a['id']]:.3f}" for a in palts) +
                  f" | {base['alternative_CR'][c]:.3f} |")
    md.append("| **整體** | " + " | ".join(f"**{base['global'][a['id']]:.3f}**" for a in palts) + " | |")
    md.append(f"\n→ AHP 建議：**{S.name(base['best'])}**（論文：Jason Davis；實際：{S.name(label['actual'])}）\n")
    md.append("## 不同換算規則／合成方式下的建議\n")
    md.append("| 規則/合成 | " + " | ".join(names) + " | 建議 |")
    md.append("|---|" + "---|" * (len(names) + 1))
    out["ahp"] = {}
    for k, r in runs.items():
        md.append(f"| {k} | " + " | ".join(f"{r['global'][a['id']]:.3f}" for a in palts) + f" | {S.name(r['best'])} |")
        out["ahp"][k] = {"global": {S.name(a): float(v) for a, v in r["global"].items()}, "best": S.name(r["best"])}
    mk = evaluate_alternatives(sit, palts, db.lg, ctx.M, ctx.WE, ctx.tto_mult)
    md.append("\n## Markov 基準模型（Hirotsu & Wright 式）\n")
    md.append("| 投手 | 本半局期望失分 | 本半局零失分機率 | 防守方勝率 |")
    md.append("|---|---|---|---|")
    for a, m in zip(palts, mk["alts"]):
        md.append(f"| {a['name']} | {m['exp_runs']:.3f} | {m['p0']:.3f} | {m['wp']:.4f} |")
    md.append(f"\n→ Markov 建議：**{S.name(mk['best'])}**\n")
    out["markov"] = {S.name(m["id"]): m for m in mk["alts"]}
    out["markov_best"] = S.name(mk["best"])
    # whole available bullpen
    r_all = evaluate(sit, alts, db.lg)
    mk_all = evaluate_alternatives(sit, alts, db.lg, ctx.M, ctx.WE, ctx.tto_mult)
    md.append("## 若把當晚所有可用牛棚都列為候選\n")
    md.append("| 投手 | 投 | AHP 整體權重 | Markov 防守方勝率 |")
    md.append("|---|---|---|---|")
    mk_map = {m["id"]: m for m in mk_all["alts"]}
    for a in alts:
        md.append(f"| {a['name']}{'（場上）' if a['is_current'] else ''} | {a['throws']} | {r_all['global'][a['id']]:.3f} | {mk_map[a['id']]['wp']:.4f} |")
    md.append(f"\n→ AHP：**{S.name(r_all['best'])}**；Markov：**{S.name(mk_all['best'])}**")
    out["full_pool"] = {"ahp_best": S.name(r_all["best"]), "markov_best": S.name(mk_all["best"]),
                        "pool": [a["name"] for a in alts]}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "e1_case_study.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(RESULTS, "e1_case_study.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print("\n".join(md))


if __name__ == "__main__":
    main()
