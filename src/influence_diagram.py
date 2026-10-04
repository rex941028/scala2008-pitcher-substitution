# -*- coding: utf-8 -*-
"""
Model 1 -- Scala (2008) Exhibit 1 "Influence Diagram" for pitcher substitution, transcribed node by
node and arrow by arrow from the published figure (paper/Scala_2008_ASEM.pdf, page 3).

  squares = decision nodes (1, 28, 31, 33, 36), circles = chance nodes (all others)
  item 1 = goal (pitcher selection); items 2, 9, 20, 29, 33 = main criteria; the rest are
  sub-criteria that roll up to their cluster's main criterion. The AHP (pitcher_ahp.py) uses the
  four clusters 9 / 20 / 29 / 33 as criteria; node 2 is the hub that joins them to the goal.

Arrows drawn with a head on both ends are stored as two directed edges.  A formal influence
diagram (Howard & Matheson 1984) must be a DAG; `validate()` reports where this one is not.
"""
from collections import defaultdict

# id: (English label, 中文, node type, cluster main-criterion id)
NODES = {
    1: ("Pitcher selection", "投手選擇（目標）", "decision", 1),
    2: ("Pitcher vs. batter matchups", "投打對決", "chance", 2),
    3: ("Bench batters' natural handedness", "板凳打者慣用手", "chance", 9),
    4: ("Bench batters' averages", "板凳打者打擊率", "chance", 9),
    5: ("Bench batters' handedness preference", "板凳打者左右投偏好", "chance", 9),
    6: ("Who the opposing team can PH", "對手可派的代打", "chance", 9),
    7: ("Next scheduled batter", "下一棒打者", "chance", 9),
    8: ("Scheduled batter handedness", "預定打者慣用手", "chance", 9),
    9: ("Scheduled batter's situation", "預定打者狀況", "chance", 9),
    10: ("Scheduled batter's handedness preference", "預定打者左右投偏好", "chance", 9),
    11: ("Scheduled batter's average", "預定打者打擊率", "chance", 9),
    12: ("Top or bottom", "上/下半局", "chance", 20),
    13: ("Inning", "局數", "chance", 20),
    14: ("Home or away", "主/客場", "chance", 20),
    15: ("Number runs ahead / behind", "領先/落後分數", "chance", 20),
    16: ("Behind or ahead", "領先或落後", "chance", 20),
    17: ("Number of outs", "出局數", "chance", 20),
    18: ("Happened in previous half inning", "上個半局發生的事", "chance", 20),
    19: ("Runners on", "壘上跑者", "chance", 20),
    20: ("State of the game", "比賽狀態", "chance", 20),
    21: ("Handedness of pitcher", "投手慣用手", "chance", 29),
    22: ("Pitcher's ERA vs. handedness", "投手對左右打防禦率", "chance", 29),
    23: ("Tiredness", "疲勞", "chance", 29),
    24: ("Injured?", "是否受傷", "chance", 29),
    25: ("How faring: number runs given up", "今日表現（失分）", "chance", 29),
    26: ("Pitcher's ERA", "投手防禦率", "chance", 29),
    27: ("Pitcher's general numbers", "投手一般數據", "chance", 29),
    28: ("Pitcher in the game", "場上投手", "decision", 29),
    29: ("In-game pitcher's situation", "場上投手狀況", "chance", 29),
    30: ("Pitcher's DERA", "投手 DERA", "chance", 29),
    31: ("Bullpen rest", "牛棚休息", "decision", 33),
    32: ("Bullpen ERA vs. handedness", "牛棚對左右打防禦率", "chance", 33),
    33: ("Bullpen specialists available", "可用牛棚專家", "decision", 33),
    34: ("Bullpen ERAs", "牛棚防禦率", "chance", 33),
    35: ("Bullpen DERA", "牛棚 DERA", "chance", 33),
    36: ("Starting rotation", "先發輪值", "decision", 33),
    37: ("Upcoming schedule", "後續賽程", "chance", 33),
    38: ("Bullpen's general numbers", "牛棚一般數據", "chance", 33),
}

GOAL = 1
HUB = 2
MAIN_CRITERIA = {9: "Batter", 20: "State", 29: "Pitcher", 33: "Bullpen"}

# (from, to); "<->" arrows appear twice.
_ONE_WAY = [
    # batter cluster
    (3, 5), (5, 4), (3, 6), (4, 6), (5, 6), (7, 6), (6, 9), (8, 9), (8, 10), (10, 11), (11, 9), (9, 2),
    # state cluster
    (14, 12), (13, 12), (12, 20), (13, 20), (14, 20), (15, 16), (15, 20), (16, 20), (17, 20),
    (18, 20), (19, 20), (20, 2),
    # pitcher cluster
    (21, 29), (22, 29), (23, 29), (23, 22), (23, 24), (24, 29), (25, 29), (25, 26), (22, 26),
    (27, 26), (26, 29), (29, 2),
    # bullpen cluster
    (31, 33), (32, 34), (34, 33), (38, 34), (36, 33), (37, 36), (37, 33), (33, 2),
]
_TWO_WAY = [(1, 2), (9, 29), (24, 25), (26, 30), (28, 29), (29, 30), (34, 35)]

EDGES = sorted(set(_ONE_WAY + _TWO_WAY + [(b, a) for a, b in _TWO_WAY]))


def successors():
    s = defaultdict(list)
    for a, b in EDGES:
        s[a].append(b)
    return s


def clusters():
    """main criterion id -> list of sub-criterion ids (the AHP hierarchy implied by the diagram)."""
    out = defaultdict(list)
    for nid, (_, _, _, cl) in NODES.items():
        if nid != cl and cl in MAIN_CRITERIA:
            out[cl].append(nid)
    return {k: sorted(v) for k, v in out.items()}


def find_cycles():
    """All elementary cycles (Johnson-style DFS, fine for 38 nodes). Each cycle as a node list."""
    succ = successors()
    nodes = sorted(NODES)
    cycles = []

    def dfs(start, v, path, on_path):
        for w in succ.get(v, []):
            if w == start:
                cycles.append(path[:])
            elif w > start and w not in on_path:
                on_path.add(w)
                path.append(w)
                dfs(start, w, path, on_path)
                path.pop()
                on_path.discard(w)

    for s in nodes:
        dfs(s, s, [s], {s})
    return cycles


def topological_order():
    """Kahn's algorithm; returns None if the graph has a cycle."""
    indeg = {n: 0 for n in NODES}
    for a, b in EDGES:
        indeg[b] += 1
    succ = successors()
    queue = sorted(n for n, d in indeg.items() if d == 0)
    order = []
    while queue:
        v = queue.pop(0)
        order.append(v)
        for w in succ.get(v, []):
            indeg[w] -= 1
            if indeg[w] == 0:
                queue.append(w)
    return order if len(order) == len(NODES) else None


def validate():
    """Structural checks a formal influence diagram must pass."""
    report = {}
    report["n_nodes"] = len(NODES)
    report["n_edges"] = len(EDGES)
    report["n_decision"] = sum(1 for v in NODES.values() if v[2] == "decision")
    report["n_chance"] = sum(1 for v in NODES.values() if v[2] == "chance")
    report["is_dag"] = topological_order() is not None
    report["cycles"] = find_cycles()
    report["two_way_arrows"] = list(_TWO_WAY)
    # every sub-criterion should reach its main criterion
    succ = successors()

    def reaches(a, target):
        seen, stack = {a}, [a]
        while stack:
            v = stack.pop()
            if v == target:
                return True
            for w in succ.get(v, []):
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return False

    report["subcriteria_not_reaching_cluster"] = [
        n for cl, subs in clusters().items() for n in subs if not reaches(n, cl)]
    # influences that cross clusters (the AHP treats clusters as independent: an assumption)
    report["cross_cluster_edges"] = [
        (a, b) for a, b in EDGES
        if NODES[a][3] in MAIN_CRITERIA and NODES[b][3] in MAIN_CRITERIA and NODES[a][3] != NODES[b][3]]
    # a decision node must have no chance-node parents it cannot observe -- report decision parents
    report["decision_node_parents"] = {
        n: sorted(a for a, b in EDGES if b == n) for n, v in NODES.items() if v[2] == "decision"}
    return report


def to_mermaid():
    lines = ["graph LR"]
    for nid, (en, zh, typ, _) in NODES.items():
        shape = ("[", "]") if typ == "decision" else ("((", "))")
        lines.append(f'  n{nid}{shape[0]}"{nid}. {zh}"{shape[1]}')
    two = set(_TWO_WAY)
    done = set()
    for a, b in EDGES:
        if (a, b) in two:
            lines.append(f"  n{a} <--> n{b}")
            done.add((b, a))
        elif (a, b) not in done and (b, a) not in two:
            lines.append(f"  n{a} --> n{b}")
    return "\n".join(lines)


def draw(path, highlight_cycles=False, title=True, font_scale=1.0):
    """Render the diagram with matplotlib (cluster layout, decision nodes as squares).
    highlight_cycles draws the arrows that form cycles in red."""
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    centers = {1: (9.6, 5.0), 2: (6.8, 5.0), 9: (3.0, 6.3), 20: (8.8, 8.3), 29: (4.6, 1.9), 33: (9.9, 1.9)}
    radii = {9: (2.35, 1.75), 20: (2.2, 1.55), 29: (2.55, 1.45), 33: (2.1, 1.45)}
    pos = {nid: centers[nid] for nid in centers}
    for cl, subs in clusters().items():
        cx, cy = centers[cl]
        rx, ry = radii[cl]
        for k, n in enumerate(subs):
            ang = 2 * math.pi * (k + 0.5) / len(subs)   # half-step offset keeps nodes off the hub arrows
            pos[n] = (cx + rx * math.cos(ang), cy + ry * math.sin(ang))
    cyc = set()
    if highlight_cycles:
        for c in find_cycles():
            for i, a in enumerate(c):
                cyc.add((a, c[(i + 1) % len(c)]))
    fig, ax = plt.subplots(figsize=(13, 10))
    for a, b in EDGES:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        red = (a, b) in cyc
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#b03f2c" if red else "#9a9f98",
                                    lw=2.0 if red else 0.8, shrinkA=16, shrinkB=16))
    cluster_color = {20: "#eb6834", 9: "#1baf7a", 29: "#eda100", 33: "#e87ba4"}
    for nid, (en, zh, typ, cl) in NODES.items():
        x, y = pos[nid]
        main = nid in MAIN_CRITERIA or nid in (GOAL, HUB)
        ec = cluster_color.get(cl, "#46515b")
        box = dict(boxstyle="square,pad=0.35" if typ == "decision" else "round,pad=0.35",
                   fc="#fbf3e4" if main else "#ffffff", ec=ec, lw=2.0 if main else 1.3)
        ax.text(x, y, f"{nid}. {zh}", ha="center", va="center", fontsize=(9 if not main else 11) * font_scale,
                fontweight="bold" if main else "normal", bbox=box, color="#16212b")
    ax.set_xlim(-0.5, 13)
    ax.set_ylim(-0.5, 10.5)
    ax.axis("off")
    if title:
        ax.set_title("Scala (2008) Exhibit 1 影響圖（方形 = 決策節點，圓角 = 機會節點）")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    import json
    r = validate()
    print(json.dumps({k: v for k, v in r.items() if k != "cycles"}, ensure_ascii=False, indent=1))
    print("cycles:", r["cycles"])
