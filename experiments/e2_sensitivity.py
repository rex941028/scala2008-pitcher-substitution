# -*- coding: utf-8 -*-
"""
E2 -- how fragile is the AHP recommendation? (on the E1 decision, plus model-level properties)

 A. unpublished Pitcher / Bullpen weight split: sweep 0 .. 16.5 (their published sum) x 3 rounding rules
 B. random criteria weights: Dirichlet around the literature-frequency weights (5 000 draws)
 C. one-at-a-time changes to the Pugh operationalisation (each a defensible reading of the paper)
 D. rank reversal: add a 4th pitcher to the paper's three and see if the three reorder
 E. consistency: CR of every alternative matrix the Pugh-difference rule can produce (3-5 alternatives)
 F. the paper's comparison counts (18 / 26 / 46 for 3 / 4 / 5 alternatives)
Output: results/e2_sensitivity.md (+ csv for A and E)
"""
import copy
import itertools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import numpy as np
import pandas as pd

from ahp_core import AHP, consistency, from_upper
from common import RESULTS, md_table
from context import Context
from criteria_weights import CRITERIA, cluster_totals, criteria_matrix
from pitcher_ahp import evaluate, pugh_ratings
from pugh_chart import diff_to_saaty

sys.path.insert(0, HERE)
from e1_case_study import find_decision, paper_alternatives  # noqa: E402


def solve_chart(chart, crit_a, mode="distributive", diff_map=diff_to_saaty):
    ids = chart.alternatives
    m = AHP(CRITERIA, ids)
    m.set_criteria_matrix(crit_a)
    for c in CRITERIA:
        t = chart.totals(c)
        judg = {}
        for i in range(len(t)):
            for j in range(i + 1, len(t)):
                s = diff_map(t[i] - t[j])
                judg[(i, j)] = s if t[i] >= t[j] else 1.0 / s
        m.set_alternative_matrix(c, from_upper(len(t), judg))
    return m.solve(mode)


def main():
    ctx = Context()
    db, sit, alts, label = find_decision(ctx)
    S = ctx.stats
    palts = paper_alternatives(db, sit, alts)
    ids = [a["id"] for a in palts]
    nm = {a["id"]: a["name"].split()[-1] for a in palts}
    md = ["# E2：敏感度分析（論文案例 2006-04-25，候選 = Westbrook / Davis / Sauerbeck）\n"]

    # ---------------- A. Pitcher / Bullpen split
    rows = []
    for p in np.arange(0, 16.51, 0.5):
        tot = cluster_totals(pitcher_total=float(p))
        for rule in ("paper", "round", "exact"):
            r = evaluate(sit, palts, db.lg, totals=tot, rule=rule)
            rows.append({"pitcher_total": p, "bullpen_total": 16.5 - p, "rule": rule,
                         **{nm[k]: v for k, v in r["global"].items()}, "best": nm[r["best"]]})
    dfA = pd.DataFrame(rows)
    dfA.to_csv(os.path.join(RESULTS, "e2_A_weight_split.csv"), index=False, encoding="utf-8-sig")
    md.append("## A. 未公開的 Pitcher/Bullpen 權重拆分（兩者和固定 16.5）\n")
    piv = dfA.pivot_table(index="rule", columns="best", values="pitcher_total", aggfunc="count").fillna(0)
    md.append("每種換算規則下，34 種拆分中各投手勝出的次數：\n")
    md.append(md_table(piv.astype(int), index=True))
    for rule in ("paper", "round", "exact"):
        seg = dfA[dfA["rule"] == rule]
        spans = []
        for best, grp in itertools.groupby(zip(seg["pitcher_total"], seg["best"]), key=lambda x: x[1]):
            g = list(grp)
            spans.append(f"{best}: Pitcher={g[0][0]:.1f}~{g[-1][0]:.1f}")
        md.append(f"\n- {rule}: " + "；".join(spans))

    # ---------------- B. random weights
    rng = np.random.default_rng(2008)
    base_w = np.array([7, 10.5, 10, 6.5]) / 34.0
    wins = {k: 0 for k in nm.values()}
    chart = pugh_ratings(sit, palts, db.lg)
    local = []
    for c in CRITERIA:
        m = AHP(CRITERIA, ids)
        local.append(consistency(chart.pairwise(c))["w"])
    local = np.vstack(local)
    N = 5000
    for _ in range(N):
        w = rng.dirichlet(base_w * 20)
        g = w @ local
        wins[nm[ids[int(np.argmax(g))]]] += 1
    md.append("\n## B. 準則權重隨機擾動（Dirichlet，平均 = 文獻頻率權重，集中度 20，5000 次）\n")
    md.append("| 投手 | 勝出比例 |\n|---|---|")
    for k, v in wins.items():
        md.append(f"| {k} | {v / N:.1%} |")
    # which weights would make Davis win at all?
    dav = [i for i in ids if nm[i] == "Davis"][0]
    j = ids.index(dav)
    grid_hits = 0
    grid = 0
    for w in itertools.product(np.arange(0, 1.01, 0.05), repeat=3):
        if sum(w) > 1:
            continue
        ww = np.array([w[0], w[1], w[2], 1 - sum(w)])
        grid += 1
        if int(np.argmax(ww @ local)) == j:
            grid_hits += 1
    md.append(f"\n在整個權重單純形上（步長 0.05，{grid} 組），Davis 勝出的組合佔 {grid_hits / grid:.1%}。")

    # ---------------- C. operationalisation variants
    crit_a = criteria_matrix(cluster_totals(), "paper")
    variants = {}
    base_chart = pugh_ratings(sit, palts, db.lg)
    variants["基準（本研究的評分規則）"] = base_chart

    def modify(fn):
        ch = copy.deepcopy(base_chart)
        fn(ch)
        return ch

    cur = [a for a in palts if a["is_current"]][0]
    variants["場上投手在 Bullpen 準則全給 0（31/36/37 不給 +1）"] = modify(
        lambda ch: [ch.set("Bullpen", cur["id"], n, 0) for n in (31, 36, 37)])
    variants["疲勞門檻改為用球 ≥ 平常 90% 即 -1"] = modify(
        lambda ch: ch.set("Pitcher", cur["id"], 23, -1 if cur["pitches_today"] >= 0.9 * cur["typical_pitches"] else 0))
    variants["今日表現只看失分（2 分 → 0）"] = modify(
        lambda ch: ch.set("Pitcher", cur["id"], 25, -1 if cur["runs_today"] >= 3 else 0))
    variants["State 情境拆分全給 0（拆分樣本太小視為無法判定）"] = modify(
        lambda ch: [ch.set("State", a, n, 0) for a in ids for n in (13, 14, 15, 16, 17, 19)])
    variants["板凳代打節點 3–6 全給 0"] = modify(
        lambda ch: [ch.set("Batter", a, n, 0) for a in ids for n in (3, 4, 5, 6)])
    variants["Westbrook 用球 98 + 被上壘 .500 → 疲勞與今日表現都 -1 且 Bullpen 給 0"] = modify(
        lambda ch: [ch.set("Bullpen", cur["id"], n, 0) for n in (31, 36, 37)] + [ch.set("Pitcher", cur["id"], 23, -1)])
    rows = []
    for name, ch in variants.items():
        r = solve_chart(ch, crit_a)
        rows.append({"變體": name, **{nm[k]: round(v, 3) for k, v in r["global"].items()}, "建議": nm[r["best"]]})
    linear = lambda d: float(min(9, 1 + abs(round(d))))
    r = solve_chart(base_chart, crit_a, diff_map=linear)
    rows.append({"變體": "Pugh 差距改線性換算（差 1→2、2→3…）", **{nm[k]: round(v, 3) for k, v in r["global"].items()},
                 "建議": nm[r["best"]]})
    tol_rows = []
    import pitcher_ahp as pa
    for f in (0.5, 2.0):
        old = (pa.TOL_AVG, pa.TOL_OBP, pa.TOL_ERA)
        pa.TOL_AVG, pa.TOL_OBP, pa.TOL_ERA = old[0] * f, old[1] * f, old[2] * f
        rr = evaluate(sit, palts, db.lg)
        pa.TOL_AVG, pa.TOL_OBP, pa.TOL_ERA = old
        rows.append({"變體": f"「平均」容忍帶 ×{f}", **{nm[k]: round(v, 3) for k, v in rr["global"].items()},
                     "建議": nm[rr["best"]]})
    md.append("\n## C. 評分規則的一次一改（論文未公開逐項評分，這些都是合理讀法）\n")
    md.append(md_table(pd.DataFrame(rows)))

    # ---------------- D. rank reversal
    md.append("\n## D. 排序反轉（rank reversal）：在三位候選外再加一位投手\n")
    extra = [a for a in alts if a["id"] not in ids and not a["is_current"]]
    rows = []
    base_d = evaluate(sit, palts, db.lg, mode="distributive")
    base_order = [nm[x] for x in base_d["ranking"]]
    for e in extra:
        for mode in ("distributive", "ideal"):
            r = evaluate(sit, palts + [e], db.lg, mode=mode)
            order3 = [nm[x] for x in r["ranking"] if x in ids]
            b3 = [nm[x] for x in evaluate(sit, palts, db.lg, mode=mode)["ranking"]]
            rows.append({"加入": e["name"], "合成": mode, "原三人排序": " > ".join(b3),
                         "加入後三人排序": " > ".join(order3), "反轉": order3 != b3})
    dfD = pd.DataFrame(rows)
    md.append(md_table(dfD))
    md.append(f"\n反轉次數：distributive {int(dfD[dfD['合成'] == 'distributive']['反轉'].sum())}/{len(extra)}，"
              f"ideal {int(dfD[dfD['合成'] == 'ideal']['反轉'].sum())}/{len(extra)}")

    # ---------------- E. consistency of Pugh-derived matrices
    md.append("\n## E. Pugh 差距換算出的比較矩陣一致性（CR）\n")
    rows = []
    for n in (3, 4, 5):
        crs = []
        for tot in itertools.product(range(-4, 5), repeat=n):
            t = np.array(tot, dtype=float)
            judg = {}
            for i in range(n):
                for j in range(i + 1, n):
                    s = diff_to_saaty(t[i] - t[j])
                    judg[(i, j)] = s if t[i] >= t[j] else 1.0 / s
            crs.append(consistency(from_upper(n, judg))["CR"])
        crs = np.array(crs)
        rows.append({"候選數": n, "總分組合數": len(crs), "CR>0.10 比例": f"{(crs > 0.1).mean():.1%}",
                     "CR 中位數": round(float(np.median(crs)), 3), "CR 最大": round(float(crs.max()), 3)})
    md.append(md_table(pd.DataFrame(rows)))
    md.append("\n（每位候選在一個準則下的 Pugh 總分取 -4..+4 的所有組合）")

    # ---------------- F. comparison counts
    md.append("\n## F. 論文宣稱的兩兩比較次數\n")
    md.append("| 候選數 | 論文 | 實際（4 準則 6 次 + 每準則 n(n-1)/2） |\n|---|---|---|")
    for n, claim in ((3, 18), (4, 26), (5, 46)):
        md.append(f"| {n} | {claim} | {6 + 4 * n * (n - 1) // 2} |")

    with open(os.path.join(RESULTS, "e2_sensitivity.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
