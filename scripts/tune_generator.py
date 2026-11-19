"""Generator tuning tool (no rendering). Evaluates generator settings on a TUNING seed
(not the pool seed) and prints per-shortcut agreement, d=1 rates and answer balance.

Usage: python scripts/tune_generator.py [--n 1500] [--seed 1000]   (grid defined below)
"""
import argparse, copy, itertools, json, sys
from collections import Counter
from siftvqa.config import ROOT, load_config
from siftvqa.scene import sample_scene
from siftvqa.seeding import make_rng
from siftvqa.shortcuts import load_shortcuts, score_question

sys.path.insert(0, str(ROOT / "scripts"))
from gen_pool import make_questions  # noqa: E402


def evaluate(cfg, n, seed, shortcuts):
    rows = []
    for i in range(n):
        scene = sample_scene(cfg, make_rng(seed, "scene", i), f"s{i}")
        for q in make_questions(scene, cfg, make_rng(seed, "questions", i)):
            rows.append({**q, **score_question(q, scene, shortcuts)})
    m = {}
    for s in shortcuts:
        rs = [r for r in rows if s["name"] in r["shortcuts"]]
        m[s["name"]] = round(100 * sum(1 - r["d_per_shortcut"][s["name"]] for r in rs) / len(rs), 1)
    app = [r for r in rows if r["shortcuts"]]
    m["d1_applicable"] = round(100 * sum(r["d"] for r in app) / len(app), 1)
    cnt = [r for r in rows if r["template"] == "count_cs"]
    m["count_dist"] = dict(sorted(Counter(r["answer"] for r in cnt).items()))
    m["count_dist_d1"] = dict(sorted(Counter(r["answer"] for r in cnt if r["d"]).items()))
    for t in ("exists_cs", "compare_cs"):
        rs = [r for r in rows if r["template"] == t]
        m[t + "_yes"] = round(100 * sum(r["answer"] == "yes" for r in rs) / len(rs), 1)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    base = load_config(ROOT / "configs" / "generator.yaml")
    sc = load_shortcuts(ROOT / "configs" / "shortcuts.yaml")
    mixes = {"bal": {1: 0.0, 2: 0.5, 3: 0.5}, "lo": {1: 0.3, 2: 0.5, 3: 0.2}, "lower": {1: 0.5, 2: 0.4, 3: 0.1}}
    for (cn, cw), (sn, sw), pf in itertools.product(mixes.items(), mixes.items(), (0.4, 0.6)):
        cfg = copy.deepcopy(base)
        cfg["palette"].update(n_colors=cw, n_shapes=sw, p_focus=pf)
        m = evaluate(cfg, a.n, a.seed, sc)
        print(f"colors={cn:5} shapes={sn:5} focus={pf}", {k: v for k, v in m.items() if "dist" not in k}, flush=True)
