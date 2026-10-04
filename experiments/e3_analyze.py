# -*- coding: utf-8 -*-
"""
E3 analysis -- reads results/e3_decisions.csv (from e3_backtest.py) and writes results/e3_backtest.md.
Confidence intervals: bootstrap over games (1 000 resamples), because PAs within a game are not
independent.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from common import RESULTS, md_table

B = 1000
RNG = np.random.default_rng(2006)


def boot(df, fn):
    """Point estimate and 95% CI of fn(df) resampling games."""
    games = df["game"].unique()
    idx = {g: np.where(df["game"].values == g)[0] for g in games}
    est = fn(df)
    vals = []
    for _ in range(B):
        pick = RNG.choice(games, len(games))
        rows = np.concatenate([idx[g] for g in pick])
        vals.append(fn(df.iloc[rows]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return est, lo, hi


def fmt(e, lo, hi, pct=False, d=3):
    if pct:
        return f"{e:.1%} [{lo:.1%}, {hi:.1%}]"
    return f"{e:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def main():
    df = pd.read_csv(os.path.join(RESULTS, "e3_decisions.csv"))
    timing = json.load(open(os.path.join(RESULTS, "e3_timing.json")))
    df["resid"] = df["runs_rest"] - df["re24"]
    md = ["# E3：2006 全季回測（AL 主場 DH 比賽、第 6 局起每個打席）\n"]
    ch = df[df["changed"]]
    md.append("## 0. 資料\n")
    md.append(md_table(pd.DataFrame([{
        "比賽數": df["game"].nunique(), "決策點（打席）": len(df), "實際換投": int(df["changed"].sum()),
        "換投率": f"{df['changed'].mean():.1%}", "平均可用牛棚數": round(df["n_relievers"].mean(), 2),
        "實際上場者不在推定名單（強制加入）": f"{ch['forced'].mean():.1%}",
        "計算錯誤而略過": timing["errors"]}])))

    # ------------------------------------------------------------ 1. change / no change
    md.append("\n## 1. 要不要換投：與總教練決策的一致性\n")
    rows = []
    for name, sub, score in [("AHP（Scala 2008）", "ahp_sub", "ahp_change_score"),
                             ("Markov（H&W 式）", "mk_sub", "mk_change_score")]:
        d = df.dropna(subset=[score])
        auc = boot(d, lambda x: roc_auc_score(x["changed"], x[score]))
        # threshold matched to the managers' change rate
        thr = np.quantile(d[score], 1 - d["changed"].mean())
        pred = d[score] > thr
        prec = (pred & d["changed"]).sum() / max(1, pred.sum())
        rows.append({"模型": name, "原始規則說要換的比例": f"{d[sub].mean():.1%}",
                     "總教練有換時模型也說換": f"{d.loc[d['changed'], sub].mean():.1%}",
                     "總教練沒換時模型說換": f"{d.loc[~d['changed'], sub].mean():.1%}",
                     "AUC（換投分數）": fmt(*auc),
                     "同換投率門檻下的命中率": f"{prec:.1%}"})
    d = df[df["cur_is_starter"]]
    auc_pc = boot(d, lambda x: roc_auc_score(x["changed"], x["cur_pitches"]))
    rows.append({"模型": "對照：只看場上先發投手用球數", "原始規則說要換的比例": "-", "總教練有換時模型也說換": "-",
                 "總教練沒換時模型說換": "-", "AUC（換投分數）": fmt(*auc_pc) + "（僅先發在場時）",
                 "同換投率門檻下的命中率": "-"})
    md.append(md_table(pd.DataFrame(rows)))
    md.append("\n換投分數：AHP = 最佳牛棚整體權重 − 場上投手整體權重；Markov = 最佳牛棚勝率 − 場上投手勝率。"
              "「同換投率門檻」= 把門檻調到模型換投比例等於實際換投率 "
              f"({df['changed'].mean():.1%}) 時，模型說換的打席裡總教練真的換的比例（隨機猜的命中率 = 換投率本身）。")

    # ------------------------------------------------------------ 2. which reliever
    md.append("\n## 2. 換誰：與總教練選的牛棚是否相同（實際換投、名單 ≥ 2 人）\n")
    c = df[df["changed"] & (df["n_relievers"] >= 2)].copy()
    rows = []
    for sub_name, sub in [("全部", c), ("排除強制加入", c[~c["forced"]])]:
        rnd = (1.0 / sub["n_relievers"]).mean()
        r = {"樣本": sub_name, "n": len(sub), "隨機猜": f"{rnd:.1%}"}
        for name, col in [("AHP", "ahp_best_rel"), ("Markov", "mk_best_rel"), ("左右+ERA 簡單法則", "simple_best_rel")]:
            e = boot(sub, lambda x: (x[col] == x["actual"]).mean())
            r[name] = fmt(*e, pct=True)
        rows.append(r)
    md.append(md_table(pd.DataFrame(rows)))
    for name, col in [("AHP", "ahp_rank_actual"), ("Markov", "mk_rank_actual")]:
        x = c.dropna(subset=[col])
        pct = ((x[col] - 1) / (x["n_relievers"] - 1).clip(lower=1)).mean()
        md.append(f"\n- {name}：總教練所選投手在模型排名的平均百分位 = {pct:.2f}（0 = 模型第一名，1 = 最後一名，隨機 = 0.50）")

    # ------------------------------------------------------------ 3. predictive validity
    md.append("\n## 3. 預測效度：模型對「實際上場投手」的評價，能否預測本半局剩餘失分？\n")
    md.append("殘差 = 實際本半局剩餘失分 − 同壘包/出局狀態的聯盟平均（RE24，2005 年估計）。"
              "若模型真的抓到投手/對戰優劣，評價越好的投手殘差應越低。\n")
    v = df.dropna(subset=["ahp_actual_rel_prio", "mk_er_actual"]).copy()
    v["mk_score"] = v["mk_er_actual"] - v["re24"]          # includes the quality of the coming hitters
    v["mk_rel"] = v["mk_er_actual"] - v["mk_er_mean"]      # pitcher only: vs the average alternative
    rho_ahp = boot(v, lambda x: spearmanr(x["ahp_actual_rel_prio"], x["resid"])[0])
    rho_mk = boot(v, lambda x: spearmanr(-x["mk_score"], x["resid"])[0])   # higher = better pitcher
    rho_mkr = boot(v, lambda x: spearmanr(-x["mk_rel"], x["resid"])[0])
    auc_mkr = boot(v, lambda x: roc_auc_score(x["runs_rest"] == 0, -x["mk_rel"]))
    mse_base = boot(v, lambda x: ((x["runs_rest"] - x["re24"]) ** 2).mean())
    mse_mk = boot(v, lambda x: ((x["runs_rest"] - x["mk_er_actual"]) ** 2).mean())
    gain = boot(v, lambda x: ((x["runs_rest"] - x["re24"]) ** 2).mean() - ((x["runs_rest"] - x["mk_er_actual"]) ** 2).mean())
    y0 = (v["runs_rest"] == 0).astype(float)
    v["y0"] = y0
    brier_base = boot(v, lambda x: ((x["y0"] - x["re24_p0"]) ** 2).mean())
    brier_mk = boot(v, lambda x: ((x["y0"] - x["mk_p0_actual"]) ** 2).mean())
    bgain = boot(v, lambda x: ((x["y0"] - x["re24_p0"]) ** 2).mean() - ((x["y0"] - x["mk_p0_actual"]) ** 2).mean())
    auc_ahp = boot(v, lambda x: roc_auc_score(x["y0"], x["ahp_actual_rel_prio"]))
    auc_mk = boot(v, lambda x: roc_auc_score(x["y0"], -x["mk_score"]))
    md.append("AHP 的分數只比較候選投手之間的高低（打者對所有候選都一樣）；為了公平，Markov 也列出「只比投手」的版本"
              "（實際投手的期望失分 − 所有候選的平均），另列含打線強弱的完整版本。\n")
    md.append(md_table(pd.DataFrame([
        {"指標": "Spearman ρ（模型評價 vs 殘差；越負越好）", "AHP": fmt(*rho_ahp),
         "Markov（只比投手）": fmt(*rho_mkr), "Markov（含打線）": fmt(*rho_mk), "只用 RE24": "0（定義）"},
        {"指標": "AUC：預測本半局零失分", "AHP": fmt(*auc_ahp), "Markov（只比投手）": fmt(*auc_mkr),
         "Markov（含打線）": fmt(*auc_mk), "只用 RE24": "-"},
    ])))
    md.append("\nMarkov 的機率/期望值校準（用完整預測取代 RE24）：\n")
    md.append(md_table(pd.DataFrame([
        {"指標": "MSE 剩餘失分", "Markov": fmt(*mse_mk), "只用 RE24": fmt(*mse_base)},
        {"指標": "MSE 改善（RE24 − Markov）", "Markov": fmt(*gain, d=4), "只用 RE24": "-"},
        {"指標": "Brier 零失分", "Markov": fmt(*brier_mk, d=4), "只用 RE24": fmt(*brier_base, d=4)},
        {"指標": "Brier 改善（RE24 − Markov）", "Markov": fmt(*bgain, d=5), "只用 RE24": "-"},
    ])))
    md.append("\n（AHP 的輸出是相對權重，不是機率或期望失分，所以無法算 MSE / Brier，也無法校準。）")
    v["ahp_q"] = pd.qcut(v["ahp_actual_rel_prio"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    v["mkr_q"] = pd.qcut((-v["mk_rel"]).rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    v["mk_q"] = pd.qcut((-v["mk_score"]).rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    q = pd.DataFrame({"五分位（1 = 模型評價最差）": [1, 2, 3, 4, 5],
                      "AHP：平均殘差": v.groupby("ahp_q", observed=True)["resid"].mean().values,
                      "Markov 只比投手：平均殘差": v.groupby("mkr_q", observed=True)["resid"].mean().values,
                      "Markov 含打線：平均殘差": v.groupby("mk_q", observed=True)["resid"].mean().values})
    md.append("\n依模型對實際上場投手的評價分五組，本半局剩餘失分殘差（分/半局）：\n")
    md.append(md_table(q))

    # ------------------------------------------------------------ 4. judging managers
    md.append("\n## 4. 用模型評價總教練（論文的第二個用途：validate the manager's decisions）\n")
    c = df[df["changed"] & (df["n_relievers"] >= 2)].copy()
    c["loss"] = c["mk_wp_best"] - c["mk_wp_actual"]
    agree_ahp = c["ahp_best_rel"] == c["actual"]
    agree_mk = c["mk_best_rel"] == c["actual"]
    rows = []
    for name, m in [("選到 AHP 第一名", agree_ahp), ("沒選 AHP 第一名", ~agree_ahp),
                    ("選到 Markov 第一名", agree_mk), ("沒選 Markov 第一名", ~agree_mk)]:
        sub = c[m]
        e = boot(sub, lambda x: x["resid"].mean())
        rows.append({"換投情況": name, "n": int(m.sum()), "實際剩餘失分殘差": fmt(*e)})
    md.append(md_table(pd.DataFrame(rows)))
    md.append(f"\n- Markov 判定：總教練的換投平均比 Markov 最佳選擇少 {c['loss'].mean() * 100:.2f} 個百分點勝率"
              f"（中位數 {c['loss'].median() * 100:.2f}）；{(c['loss'] < 0.005).mean():.1%} 的換投在最佳選擇 0.5 個百分點以內。")
    md.append("- 注意：「選到 / 沒選到」不是隨機分組（總教練會依情境選人），殘差差異只能當描述，不是因果效果。")

    # ------------------------------------------------------------ 4b. bullpen roles
    md.append("\n## 4b. 牛棚角色：模型會不會把同一位投手用到底？\n")
    rows = []
    c["inn"] = np.where(c["inning"] <= 7, "6-7", "8+")
    for name, col in [("總教練實際", "actual"), ("AHP", "ahp_best_rel"), ("Markov", "mk_best_rel"),
                      ("左右+ERA 簡單法則", "simple_best_rel")]:
        share = np.mean([g[col].value_counts(normalize=True).iloc[0] for _, g in c.groupby("fld_team")])
        same = []
        for _, g in c.groupby("fld_team"):
            a = g[g["inn"] == "6-7"][col].value_counts()
            b = g[g["inn"] == "8+"][col].value_counts()
            if len(a) and len(b):
                same.append(a.index[0] == b.index[0])
        rows.append({"決策者": name, "各隊換投中最常被選的那一位所佔比例": f"{share:.1%}",
                     "6–7 局與 8 局後最常選的是同一人（隊數）": f"{sum(same)}/{len(same)}"})
    md.append(md_table(pd.DataFrame(rows)))
    md.append("\n總教練會分角色（中繼、布局、終結者）；兩個模型都只看眼前這個打席／半局，"
              "所以幾乎永遠推薦同一位「數據最好」的投手，不會把終結者留到第 9 局。")

    # ------------------------------------------------------------ 5. consistency & runtime
    md.append("\n## 5. 一致性與運算時間\n")
    md.append(f"- AHP：{(df['ahp_max_cr'] > 0.1).mean():.1%} 的決策點至少有一個候選比較矩陣 CR > 0.10"
              "（Saaty 的標準下應重新評分，但 Pugh 差距換算是自動的，無法「重問」決策者）。")
    md.append(f"- 平均每個決策：AHP {timing['ahp_ms']:.1f} ms、Markov {timing['markov_ms']:.1f} ms（同一台電腦、Python/numpy）。"
              "論文說 Markov 方法「CPU 時間極長」，那是針對 H&W 整場 143 萬狀態的動態規劃；只算到半局結束的 Markov 已可即時計算。")
    with open(os.path.join(RESULTS, "e3_backtest.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
