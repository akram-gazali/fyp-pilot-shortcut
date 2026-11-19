"""Stage 2: score every question (intended answer, shortcut answers, d flag) and write the
disagreement report. Every number in the report is computed here from scored.jsonl.

Usage: python scripts/score_pool.py
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from siftvqa import dsl
from siftvqa.config import ROOT, load_config, sha256_file
from siftvqa.runmeta import write_run_meta
from siftvqa.shortcuts import load_shortcuts, score_question

POOL = ROOT / "data" / "pool"
SHORTCUTS = ROOT / "configs" / "shortcuts.yaml"


def pct(a, b):
    return round(100 * a / b, 1) if b else None


def mutual_info_bits(pairs):
    n = len(pairs)
    cx, cy, cxy = Counter(x for x, _ in pairs), Counter(y for _, y in pairs), Counter(pairs)
    return sum(c / n * math.log2((c / n) / (cx[x] / n * cy[y] / n)) for (x, y), c in cxy.items())


def chi2_cramers_v(pairs):
    n = len(pairs)
    cx, cy, cxy = Counter(x for x, _ in pairs), Counter(y for _, y in pairs), Counter(pairs)
    chi2 = sum((cxy[(x, y)] - cx[x] * cy[y] / n) ** 2 / (cx[x] * cy[y] / n) for x in cx for y in cy)
    k = min(len(cx), len(cy)) - 1
    return chi2, (math.sqrt(chi2 / (n * k)) if k > 0 and n else 0.0)


def point_biserial(d, x):
    n = len(d)
    md, mx = sum(d) / n, sum(x) / n
    cov = sum((a - md) * (b - mx) for a, b in zip(d, x)) / n
    sd = math.sqrt(sum((a - md) ** 2 for a in d) / n)
    sx = math.sqrt(sum((b - mx) ** 2 for b in x) / n)
    return cov / (sd * sx) if sd and sx else None


def main():
    shortcuts = load_shortcuts(SHORTCUTS)
    sc_hash = sha256_file(SHORTCUTS)
    rows = []
    scene_cache = {}
    for line in open(POOL / "questions.jsonl"):
        q = json.loads(line)
        sid = q["scene_id"]
        if sid not in scene_cache:
            scene_cache[sid] = json.loads((POOL / "scenes" / f"{sid}.json").read_text())
        scene = scene_cache[sid]
        s = score_question(q, scene, shortcuts)
        rows.append({**q, **s, "n_objects": len(scene["objects"]), "shortcut_hash": sc_hash})
    (POOL / "scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    N = len(rows)
    templates = sorted({r["template"] for r in rows})
    rep = {"shortcuts_sha256": sc_hash, "n_questions": N, "pool": "pilot pool (all 4000 scenes, before Stage 3 splits)"}

    # --- overall d rate
    app = [r for r in rows if r["shortcuts"]]
    rep["d1_rate_pool_pct"] = pct(sum(r["d"] for r in rows), N)
    rep["d1_rate_shortcut_applicable_pct"] = pct(sum(r["d"] for r in app), len(app))
    rep["n_shortcut_applicable"] = len(app)

    # --- agreement per shortcut (on the questions that shortcut applies to, and pool-wide)
    rep["shortcut_agreement"] = {}
    for s in shortcuts:
        rs = [r for r in rows if s["name"] in r["shortcuts"]]
        agree = sum(1 - r["d_per_shortcut"][s["name"]] for r in rs)
        rep["shortcut_agreement"][s["name"]] = {
            "n_applicable": len(rs),
            "agree_pct_of_applicable": pct(agree, len(rs)),
            "agree_pct_of_whole_pool_counting_inapplicable_as_not_agreeing": pct(agree, N),
            "d1_pct_of_applicable": pct(len(rs) - agree, len(rs)),
        }

    # --- d=1 by template and by answer value
    rep["d1_by_template"] = {
        t: {"n": sum(r["template"] == t for r in rows),
            "d1": sum(r["d"] for r in rows if r["template"] == t),
            "d1_pct": pct(sum(r["d"] for r in rows if r["template"] == t), sum(r["template"] == t for r in rows))}
        for t in templates
    }
    rep["d1_by_template_and_answer"] = {}
    for t in templates:
        by = defaultdict(lambda: [0, 0])
        for r in rows:
            if r["template"] == t:
                by[r["answer"]][0] += 1
                by[r["answer"]][1] += r["d"]
        rep["d1_by_template_and_answer"][t] = {a: {"n": n, "d1": d, "d1_pct": pct(d, n)} for a, (n, d) in sorted(by.items())}

    # --- does d track the label / difficulty? (shortcut-applicable templates only)
    rep["label_confound"] = {}
    for t in [t for t in templates if any(r["template"] == t and r["shortcuts"] for r in rows)]:
        rs = [r for r in rows if r["template"] == t]
        pairs = [(r["d"], r["answer"]) for r in rs]
        chi2, v = chi2_cramers_v(pairs)
        d1 = [r for r in rs if r["d"]]
        best_acc = sum(max(Counter(r["answer"] for r in rs if r["d"] == dv).values()) for dv in (0, 1) if any(r["d"] == dv for r in rs))
        rep["label_confound"][t] = {
            "n": len(rs), "n_d1": len(d1),
            "answer_dist_when_d1": dict(sorted(Counter(r["answer"] for r in d1).items())),
            "answer_dist_when_d0": dict(sorted(Counter(r["answer"] for r in rs if not r["d"]).items())),
            "chi2": round(chi2, 1), "cramers_v": round(v, 3),
            "mutual_info_bits": round(mutual_info_bits(pairs), 4),
            "acc_predicting_answer_from_d_alone_pct": pct(best_acc, len(rs)),
            "acc_majority_answer_alone_pct": pct(max(Counter(r["answer"] for r in rs).values()), len(rs)),
        }
    ex = [r for r in rows if r["template"] == "exists_cs" and r["d"]]
    rep["exists_d1_all_answer_no"] = {"n_d1": len(ex), "n_d1_with_answer_no": sum(r["answer"] == "no" for r in ex)}
    # exists d=1 breakdown: answer "no" questions, how many have d=1
    exno = [r for r in rows if r["template"] == "exists_cs" and r["answer"] == "no"]
    exyes = [r for r in rows if r["template"] == "exists_cs" and r["answer"] == "yes"]
    rep["exists_p_d1_given_answer"] = {"no": pct(sum(r["d"] for r in exno), len(exno)), "yes": pct(sum(r["d"] for r in exyes), len(exyes))}

    # --- difficulty: number of objects and program length
    rep["d1_by_n_objects"] = {}
    for t in [t for t in templates if any(r["template"] == t and r["shortcuts"] for r in rows)]:
        by = defaultdict(lambda: [0, 0])
        for r in rows:
            if r["template"] == t:
                by[r["n_objects"]][0] += 1
                by[r["n_objects"]][1] += r["d"]
        rep["d1_by_n_objects"][t] = {k: {"n": n, "d1_pct": pct(d, n)} for k, (n, d) in sorted(by.items())}
    rep["difficulty_correlation"] = {
        "corr_d_vs_n_objects_(shortcut-applicable)": point_biserial([r["d"] for r in app], [r["n_objects"] for r in app]),
        "corr_d_vs_program_length_(shortcut-applicable)": point_biserial([r["d"] for r in app], [r["program_length"] for r in app]),
        "program_length_by_template": {t: sorted({r["program_length"] for r in rows if r["template"] == t}) for t in templates},
    }

    # --- combined count+exists view, and what the quotas can draw from
    ce = [r for r in rows if r["shortcuts"]]
    rep["combined_count_exists"] = {
        "n": len(ce), "d1": sum(r["d"] for r in ce), "d1_pct": pct(sum(r["d"] for r in ce), len(ce)),
        "agree_pct_any_shortcut_free": pct(sum(1 - r["d"] for r in ce), len(ce)),
    }
    ex_rows = [r for r in rows if r["template"] == "exists_cs"]
    cnt_d1 = [r for r in rows if r["template"] == "count_cs" and r["d"]]
    rep["quota_feasibility"] = {
        "exists_mode": load_config(ROOT / "configs" / "selection.yaml")["exists_mode"],
        "count_d1_available_by_answer": dict(sorted(Counter(r["answer"] for r in cnt_d1).items())),
        "count_answer_dist_all": dict(sorted(Counter(r["answer"] for r in rows if r["template"] == "count_cs").items())),
        "exists_d1_no": sum(r["d"] for r in ex_rows),
        "exists_d0_no": sum(1 for r in ex_rows if not r["d"] and r["answer"] == "no"),
        "exists_d0_yes": sum(1 for r in ex_rows if not r["d"] and r["answer"] == "yes"),
    }

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "stage2_disagreement.json").write_text(json.dumps(rep, indent=2))
    write_run_meta(out / "stage2", {"shortcuts": shortcuts}, stage="2-score", extra={"shortcuts_sha256": sc_hash})
    man = ROOT / "data" / "manifests" / "stage2_manifest.json"
    man.write_text(json.dumps({
        "stage": "2", "shortcuts_sha256": sc_hash,
        "scored_sha256": sha256_file(POOL / "scored.jsonl"),
        "questions_sha256": sha256_file(POOL / "questions.jsonl"),
        "d1_rate_pool_pct": rep["d1_rate_pool_pct"], "d1_rate_shortcut_applicable_pct": rep["d1_rate_shortcut_applicable_pct"],
    }, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
