"""Stage 3c: run the four selectors (all seeds, both exists modes), write selection manifests and
the arm-comparison report.

Usage: python scripts/run_selectors.py
"""
import copy
import json
import statistics
from collections import Counter

from siftvqa.config import ROOT, load_config, sha256_file
from siftvqa.runmeta import write_run_meta
from siftvqa.selectors import (ARMS, _selectable_templates, bucket_of, make_plan, quota_of,
                               run_all_arms, select_matched_random, shared_base)

SPL = ROOT / "data" / "splits"
MODES = ("stratified_quota", "fixed_shared_base")


def arm_metrics(rows, by_id, ids, ref_ids, buckets):
    rs = [by_id[i] for i in ids]
    m = {"n": len(rs), "by_template": {}}
    for t in sorted({r["template"] for r in rs}):
        ts = [r for r in rs if r["template"] == t]
        m["by_template"][t] = {
            "n": len(ts),
            "answers": dict(sorted(Counter(r["answer"] for r in ts).items())),
            "d1_share_pct": round(100 * sum(r["d"] for r in ts) / len(ts), 1),
            "mean_n_objects": round(sum(r["n_objects"] for r in ts) / len(ts), 2),
        }
        if t == "count_cs":
            m["by_template"][t]["both_shortcuts_disagree_pct"] = round(100 * sum(sum(r["d_per_shortcut"].values()) == 2 for r in ts) / len(ts), 1)
            m["by_template"][t]["mean_n_disagreeing_shortcuts"] = round(sum(sum(r["d_per_shortcut"].values()) for r in ts) / len(ts), 3)
    sel = [r for r in rs if r["template"] in ("count_cs", "exists_cs")]
    m["d1_share_all_pct"] = round(100 * sum(r["d"] for r in rs) / len(rs), 1)
    m["d1_share_count_exists_pct"] = round(100 * sum(r["d"] for r in sel) / len(sel), 1)
    m["overlap_with_shortcut_arm"] = len(set(ids) & set(ref_ids))
    return m


def gap(r):
    """Graded disagreement for count_cs: summed |intended - shortcut| over the two count shortcuts."""
    return sum(abs(int(v["answer"]) - int(r["intended"])) for v in r["shortcuts"].values())


def weak_contrast_diagnostics(pool, by_id, cfg0, seeds):
    """Measured numbers for the 'what if the contrast is weak' options. NOT arms; nothing here
    changes the design. Uses stratified_quota selections (count_cs is identical in both modes)."""
    cfg = copy.deepcopy(cfg0)
    cfg["exists_mode"] = "stratified_quota"
    cnt = [r for r in pool if r["template"] == "count_cs"]
    out = {"count_pool_n": len(cnt)}
    strict = [r for r in cnt if sum(r["d_per_shortcut"].values()) == 2]
    out["strict_d_both_shortcuts_disagree"] = {"pool_n": len(strict), "pool_share_pct": round(100 * len(strict) / len(cnt), 1)}
    per = {"shortcut_aware": [], "matched_random": [], "graded_top_gap_(diagnostic)": []}
    strict_short = []
    for seed in seeds:
        arms, quota, plan = run_all_arms(pool, cfg, seed)
        lab = Counter()
        for (t, a, _b), k in quota.items():
            if t == "count_cs":
                lab[a] += k
        graded = []
        for a, k in lab.items():
            cell = sorted((r for r in cnt if r["answer"] == a), key=lambda r: (-sum(r["d_per_shortcut"].values()), -gap(r), r["qid"]))
            graded += cell[:k]
            strict_short.append(max(0, k - sum(1 for r in strict if r["answer"] == a)))
        sets = {"shortcut_aware": [by_id[i] for i in arms["shortcut_aware"][0] if by_id[i]["template"] == "count_cs"],
                "matched_random": [by_id[i] for i in arms["matched_random"][0] if by_id[i]["template"] == "count_cs"],
                "graded_top_gap_(diagnostic)": graded}
        for k, rs in sets.items():
            per[k].append({"mean_abs_gap": sum(gap(r) for r in rs) / len(rs),
                           "mean_n_disagreeing": sum(sum(r["d_per_shortcut"].values()) for r in rs) / len(rs),
                           "both_disagree_pct": 100 * sum(sum(r["d_per_shortcut"].values()) == 2 for r in rs) / len(rs),
                           "d1_pct": 100 * sum(r["d"] for r in rs) / len(rs),
                           "mean_n_objects": sum(r["n_objects"] for r in rs) / len(rs)})
    out["count_arms_mean_over_seeds"] = {k: {m: round(statistics.mean(x[m] for x in v), 2) for m in v[0]} for k, v in per.items()}
    out["strict_d_would_be_short_by_examples_per_seed_(sum over answer cells)"] = round(sum(strict_short) / len(seeds), 1)
    return out


def main():
    cfg0 = load_config(ROOT / "configs" / "selection.yaml")
    pool = [json.loads(l) for l in open(SPL / "pool.jsonl")]
    by_id = {r["qid"]: r for r in pool}
    buckets = cfg0["n_objects_buckets"]
    seeds = cfg0["seeds"]
    base_hashes = {"shortcuts_sha256": sha256_file(ROOT / "configs" / "shortcuts.yaml"),
                   "pool_sha256": sha256_file(SPL / "pool.jsonl"),
                   "selection_config_sha256": sha256_file(ROOT / "configs" / "selection.yaml")}
    pool_ref = {t: round(100 * sum(r["d"] for r in pool if r["template"] == t) / sum(r["template"] == t for r in pool), 1)
                for t in ("count_cs", "exists_cs", "compare_shape")}
    report = {"pilot_note": "pilot numbers", "budget": cfg0["budget"], "seeds": seeds,
              "pool_size": len(pool), "pool_d1_share_pct_by_template": pool_ref,
              "pool_count_both_shortcuts_disagree_pct": round(100 * sum(sum(r["d_per_shortcut"].values()) == 2 for r in pool if r["template"] == "count_cs") / sum(r["template"] == "count_cs" for r in pool), 1),
              "modes": {}}
    for mode in MODES:
        cfg = copy.deepcopy(cfg0)
        cfg["exists_mode"] = mode
        per_seed = []
        for seed in seeds:
            arms, quota, plan = run_all_arms(pool, cfg, seed)
            sc_ids = arms["shortcut_aware"][0]
            # diagnostic only: random matched on answer counts but NOT on object-count buckets
            lab_only, _ = select_matched_random(pool, quota, plan, seed, mode, False, buckets)
            all_arms = {**{a: arms[a][0] for a in ARMS}, "random_label_only_(diagnostic)": lab_only}
            res = {a: arm_metrics(pool, by_id, ids, sc_ids, buckets) for a, ids in all_arms.items()}
            # exact label / bucket match checks (shortcut vs matched random)
            sel_t = _selectable_templates(mode)
            lm = lambda ids: Counter((by_id[i]["template"], by_id[i]["answer"]) for i in ids if by_id[i]["template"] in sel_t)
            bm = lambda ids: quota_of(by_id, ids, sel_t, buckets)
            res["checks"] = {
                "matched_random_label_counts_identical_to_shortcut_arm": lm(sc_ids) == lm(all_arms["matched_random"]),
                "difficulty_label_counts_identical": lm(sc_ids) == lm(all_arms["difficulty"]),
                "diversity_label_counts_identical": lm(sc_ids) == lm(all_arms["diversity"]),
                "matched_random_bucket_counts_identical": bm(sc_ids) == bm(all_arms["matched_random"]),
                "shared_base_identical_in_all_arms": all(set(shared_base(pool, plan, seed, mode)) <= set(ids) for ids in all_arms.values()),
                "shortcut_arm_info": arms["shortcut_aware"][1], "matched_random_info": arms["matched_random"][1],
                "all_count_examples_in_shortcut_arm_are_d1": all(by_id[i]["d"] == 1 for i in sc_ids if by_id[i]["template"] == "count_cs"),
            }
            per_seed.append(res)
            man = {"stage": "3-selection", "exists_mode": mode, "seed": seed, "budget": cfg["budget"], "plan": plan,
                   **base_hashes, "arms": {a: ids for a, ids in all_arms.items() if "diagnostic" not in a}}
            (ROOT / "data" / "manifests" / f"selection_{mode}_seed{seed}.json").write_text(json.dumps(man))
        # aggregate over seeds
        agg = {}
        for arm in per_seed[0]:
            if arm == "checks":
                continue
            def collect(f):
                v = [f(s[arm]) for s in per_seed]
                return {"mean": round(statistics.mean(v), 1), "sd": round(statistics.stdev(v), 1) if len(v) > 1 else None}
            agg[arm] = {
                "d1_share_all_pct": collect(lambda a: a["d1_share_all_pct"]),
                "d1_share_count_exists_pct": collect(lambda a: a["d1_share_count_exists_pct"]),
                "by_template": {t: {
                    "d1_share_pct": collect(lambda a, t=t: a["by_template"][t]["d1_share_pct"]),
                    "mean_n_objects": collect(lambda a, t=t: a["by_template"][t]["mean_n_objects"]),
                    **({"both_shortcuts_disagree_pct": collect(lambda a, t=t: a["by_template"][t]["both_shortcuts_disagree_pct"]),
                        "mean_n_disagreeing_shortcuts": collect(lambda a, t=t: a["by_template"][t]["mean_n_disagreeing_shortcuts"])} if t == "count_cs" else {}),
                } for t in per_seed[0][arm]["by_template"]},
                "overlap_with_shortcut_arm": collect(lambda a: a["overlap_with_shortcut_arm"]),
            }
        report["modes"][mode] = {"aggregate_over_seeds": agg, "seed0_detail": per_seed[0],
                                 "checks_all_seeds": [s["checks"] for s in per_seed]}
    report["weak_contrast_diagnostics"] = weak_contrast_diagnostics(pool, by_id, cfg0, seeds)
    (ROOT / "reports" / "stage3_selection.json").write_text(json.dumps(report, indent=2))
    write_run_meta(ROOT / "reports" / "stage3", {"selection": cfg0}, stage="3-selection", extra=base_hashes)
    print(json.dumps({m: {a: v["d1_share_count_exists_pct"] for a, v in d["aggregate_over_seeds"].items()} for m, d in report["modes"].items()}, indent=1))


if __name__ == "__main__":
    main()
