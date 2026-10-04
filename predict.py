# -*- coding: utf-8 -*-
"""
換投預測：對 2006 年任一場 MLB 比賽的任一個打席，用 AHP（Scala 2008）與 Markov 兩個模型建議「續投或換誰」。
Pitcher-substitution prediction for any plate appearance of a 2006 MLB game, with both models.

用法 / usage
  python predict.py CLE200604250 --list                      # 列出這場比賽可以預測的打席
  python predict.py CLE200604250 --inning 6 --half top       # 6 局上第一個打席（論文的例子）
  python predict.py CLE200604250 --index 37                  # 用 --list 的編號
  python predict.py CLE200604250 --inning 6 --half top --only westj001,davij005,saues001
  python predict.py CLE200604250 --inning 6 --half top --json

比賽代碼 = 主隊代碼 + 日期 + 第幾場，例如 CLE200604250（Retrosheet 格式）。
只用該場比賽「之前」可得的資料（2005 全季 + 2006 當日以前），不偷看未來。
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from common import AL_TEAMS  # noqa: E402

BASES = ["壘上無人", "一壘", "二壘", "一、二壘", "三壘", "一、三壘", "二、三壘", "滿壘"]
CRIT_ZH = {"State": "比賽狀態", "Batter": "預定打者", "Pitcher": "場上投手", "Bullpen": "牛棚"}


def half_zh(h):
    return "上" if h == 0 else "下"


def load(game):
    if not (len(game) == 12 and game[3:7] == "2006"):
        raise SystemExit("目前只支援 2006 年的比賽（模型用 2005 年當先驗）。比賽代碼例：CLE200604250")
    if game[:3] not in AL_TEAMS:
        print("注意：論文假設有 DH（指定打擊）。國聯主場的比賽投手要打擊，結果僅供參考。", file=sys.stderr)
    print("載入資料與建立模型（約 30–60 秒）…", file=sys.stderr)
    t0 = time.time()
    from context import Context
    from situation import DecisionBuilder
    ctx = Context(verbose=False)
    db = DecisionBuilder(ctx.stats, ctx.M, ctx.ev2006, min_inning=1, dh_only=False)
    decisions = list(db.decisions(game_ids=[game]))
    if not decisions:
        raise SystemExit(f"找不到比賽 {game}，或這場沒有可預測的打席。")
    print(f"完成（{time.time() - t0:.0f} 秒）", file=sys.stderr)
    return ctx, db, decisions


def describe(sit, label, S):
    return (f"{sit['inning']} 局{half_zh(sit['half'])}　{sit['outs']} 出局　{BASES[sit['bases']]}　"
            f"防守方{'領先' if sit['fld_lead'] > 0 else '落後' if sit['fld_lead'] < 0 else '平手'}"
            f"{abs(sit['fld_lead']) if sit['fld_lead'] else ''}　打者 {sit['batter']['name']}（{sit['batter']['bats']}）"
            f"　場上投手 {S.name(label['prev'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game", help="Retrosheet 比賽代碼，例如 CLE200604250")
    ap.add_argument("--list", action="store_true", help="列出可以預測的打席")
    ap.add_argument("--index", type=int, help="--list 顯示的編號")
    ap.add_argument("--inning", type=int, help="局數")
    ap.add_argument("--half", choices=["top", "bottom"], help="上半局 top / 下半局 bottom")
    ap.add_argument("--pa", type=int, default=1, help="這個半局的第幾個打席（預設 1）")
    ap.add_argument("--only", help="只比較這些投手（Retrosheet 球員代碼，逗號分隔）；場上投手一定會列入")
    ap.add_argument("--rule", default="paper", choices=["paper", "round", "exact"], help="權重換算規則")
    ap.add_argument("--json", action="store_true", help="輸出 JSON")
    a = ap.parse_args()

    ctx, db, decisions = load(a.game)
    S = ctx.stats
    if a.list:
        for i, (sit, alts, lab) in enumerate(decisions):
            mark = f"→ 換上 {S.name(lab['actual'])}" if lab["changed"] else ""
            print(f"[{i:3d}] {describe(sit, lab, S)}　{mark}")
        return

    if a.index is not None:
        if not 0 <= a.index < len(decisions):
            raise SystemExit(f"--index 必須在 0 到 {len(decisions) - 1} 之間")
        sit, alts, lab = decisions[a.index]
    else:
        if a.inning is None or a.half is None:
            raise SystemExit("請指定 --index，或同時指定 --inning 與 --half（可先用 --list 查看）")
        h = 0 if a.half == "top" else 1
        rows = [d for d in decisions if d[0]["inning"] == a.inning and d[0]["half"] == h]
        if len(rows) < a.pa:
            raise SystemExit(f"{a.inning} 局{half_zh(h)}只有 {len(rows)} 個可預測的打席")
        sit, alts, lab = rows[a.pa - 1]

    if a.only:
        keep = set(x.strip() for x in a.only.split(","))
        extra = [db.pitcher_alt(p, sit["date"], False) for p in keep
                 if p not in {x["id"] for x in alts} and p != lab["prev"]]
        alts = [x for x in alts if x["is_current"] or x["id"] in keep] + extra

    from markov_model import evaluate_alternatives
    from pitcher_ahp import evaluate
    ahp = evaluate(sit, alts, db.lg, rule=a.rule)
    mk = evaluate_alternatives(sit, alts, db.lg, ctx.M, ctx.WE, ctx.tto_mult)
    mkd = {m["id"]: m for m in mk["alts"]}
    order = sorted(alts, key=lambda x: -ahp["global"][x["id"]])
    out = {
        "game": a.game, "situation": describe(sit, lab, S),
        "alternatives": [{
            "id": x["id"], "name": x["name"], "throws": x["throws"], "current": x["is_current"],
            "ahp_priority": round(float(ahp["global"][x["id"]]), 4),
            "ahp_pugh_totals": {c: int(sum(ahp["pugh"].ratings[c][x["id"]].values())) for c in CRIT_ZH},
            "markov_wp": round(mkd[x["id"]]["wp"], 4), "markov_exp_runs": round(mkd[x["id"]]["exp_runs"], 3),
            "markov_p0": round(mkd[x["id"]]["p0"], 3)} for x in order],
        "ahp_recommendation": S.name(ahp["best"]) + ("（續投）" if not ahp["substitute"] else ""),
        "markov_recommendation": S.name(mk["best"]) + ("（續投）" if not mk["substitute"] else ""),
        "actual": S.name(lab["actual"]) + ("（續投）" if not lab["changed"] else "（換投）"),
        "actual_runs_rest_of_half_inning": lab["runs_rest"],
    }
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return
    print("\n" + out["situation"])
    print(f"{'投手':<22}{'投':<3}{'AHP 權重':>9}{'Markov 勝率':>12}{'期望失分':>9}{'零失分':>8}　Pugh 小計（狀態/打者/投手/牛棚）")
    for r in out["alternatives"]:
        t = r["ahp_pugh_totals"]
        name = r["name"] + ("（場上）" if r["current"] else "")
        print(f"{name:<22}{r['throws']:<3}{r['ahp_priority']:>9.3f}{r['markov_wp']:>12.1%}{r['markov_exp_runs']:>9.2f}"
              f"{r['markov_p0']:>8.1%}　{t['State']:+d} / {t['Batter']:+d} / {t['Pitcher']:+d} / {t['Bullpen']:+d}")
    print(f"\nAHP 建議：{out['ahp_recommendation']}")
    print(f"Markov 建議：{out['markov_recommendation']}")
    print(f"實際：{out['actual']}，這半局之後失 {out['actual_runs_rest_of_half_inning']} 分")


if __name__ == "__main__":
    main()
