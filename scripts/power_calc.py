"""Stage 4: held-out test-size proposal from a power calculation (CPU).

Two endpoints are sized separately:
  plain accuracy  - on all held-out questions
  joint score     - on the paired subset only (original, relevant edit and irrelevant edit all right)
All discordance / seed-noise inputs are ASSUMPTIONS (labelled in the output); the paired-subset
coverage is measured (reports/stage3_edits.json, held-out template).

Usage: python scripts/power_calc.py
"""
import json
import math

from siftvqa.config import ROOT, load_config
from siftvqa.power import max_sigma_seed, min_detectable_delta, n_independent, n_paired, n_with_seed_noise, power_at

DELTA = 0.03                      # the pre-set 3-point margin
PSI_PLAIN = (0.10, 0.20, 0.30)    # ASSUMED share of items where the two arms disagree
PSI_JOINT = (0.20, 0.30, 0.40)    # joint needs 3 right answers, so arms disagree more often
SIGMA = (0.0, 0.01, 0.015, 0.02)  # ASSUMED sd of an arm's accuracy across seeds (0 = ignore seed noise)
KS = (3, 5, 8)
HEADLINE = {"psi_plain": 0.20, "psi_joint": 0.30, "sigma": 0.01, "k": 5}   # planning case


def table(psis):
    rows = []
    for psi in psis:
        for sig in SIGMA:
            for k in KS if sig > 0 else (1,):
                rows.append({"psi": psi, "sigma_seed": sig, "k_seeds": k,
                             "n_needed_80pct_power": n_with_seed_noise(DELTA, psi, sig, k) if sig > 0 else n_paired(DELTA, psi)})
    return rows


def main():
    edits = json.load(open(ROOT / "reports" / "stage3_edits.json"))
    scfg = load_config(ROOT / "configs" / "splits.yaml")
    n_test = scfg["n_test_scenes"]
    n_held_unpaired = edits["n_without_valid_pair"]["compare_cs"]
    coverage = (n_test - n_held_unpaired) / n_test            # measured: share of held-out questions with a pair
    n_paired_now = n_test - n_held_unpaired

    h = HEADLINE
    n_plain = n_with_seed_noise(DELTA, h["psi_plain"], h["sigma"], h["k"])
    n_joint = n_with_seed_noise(DELTA, h["psi_joint"], h["sigma"], h["k"])
    scenes_needed = None if None in (n_plain, n_joint) else max(n_plain, math.ceil(n_joint / coverage))
    out = {
        "assumed_inputs": {"delta": DELTA, "alpha": 0.05, "power": 0.8, "psi_plain": PSI_PLAIN, "psi_joint": PSI_JOINT,
                           "sigma_seed": SIGMA, "k_seeds": KS, "headline_case": h,
                           "note": "psi and sigma_seed are ASSUMED, not measured; the pilot fine-tunes will give a first rough sigma_seed"},
        "measured_inputs": {"current_heldout_test_questions": n_test, "paired_heldout_questions_now": n_paired_now,
                            "paired_coverage_of_heldout": round(coverage, 3)},
        "independent_proportions_reference_n_per_arm": {"p1=0.60 vs p2=0.63": n_independent(0.60, 0.63), "p1=0.50 vs p2=0.53": n_independent(0.50, 0.53)},
        "plain_accuracy_n_needed": table(PSI_PLAIN),
        "joint_score_paired_subset_n_needed": table(PSI_JOINT),
        "power_of_current_test_set": {
            "plain_n=%d" % n_test: {f"sigma={s},k={k}": round(power_at(n_test, DELTA, h["psi_plain"], s, k), 3) for s in SIGMA[1:] for k in KS},
            "joint_n=%d" % n_paired_now: {f"sigma={s},k={k}": round(power_at(n_paired_now, DELTA, h["psi_joint"], s, k), 3) for s in SIGMA[1:] for k in KS},
        },
        "min_detectable_gain_points_at_80pct_power": {
            "plain_n=%d_sigma=0.01_k=%d" % (n_test, h["k"]): _pts(min_detectable_delta(n_test, h["psi_plain"], h["sigma"], h["k"])),
            "joint_n=%d_sigma=0.01_k=%d" % (n_paired_now, h["k"]): _pts(min_detectable_delta(n_paired_now, h["psi_joint"], h["sigma"], h["k"])),
            "plain_n=%d_sigma=0.02_k=%d" % (n_test, h["k"]): _pts(min_detectable_delta(n_test, h["psi_plain"], 0.02, h["k"])),
        },
        "proposal": {
            "headline_case": h,
            "plain_questions_needed": n_plain, "joint_paired_questions_needed": n_joint,
            "held_out_test_scenes_proposed": scenes_needed,
            "rule": "scenes = max(plain n, joint n / paired coverage), one held-out question per scene",
            "seed_noise_ceiling_points": {f"k={k}": round(100 * max_sigma_seed(DELTA, k), 2) for k in KS},
            "seeds": "seed noise sets a floor: if one arm's accuracy varies across seeds by more than the ceiling above (sd, in points), NO test size gives 80% power for a 3-point gain with that many seeds; the pilot should measure sigma_seed before the test size is fixed",
            "scenes_needed_if_paired_coverage_were_0.9": None if n_joint is None else max(n_plain, math.ceil(n_joint / 0.9)),
            "note_one_question_per_scene": "held-out questions from the same scene are correlated; the test set uses one held-out question per scene, so n questions = n scenes",
            "reporting": "report plain accuracy on ALL held-out questions and the joint score on the PAIRED subset as two separate endpoints, each with its own n and interval; never pool them",
        },
    }
    (ROOT / "reports" / "stage4_power.json").write_text(json.dumps(out, indent=2))
    (ROOT / "reports" / "stage4_test_size_proposal.md").write_text(render_md(out))
    print(json.dumps(out["proposal"], indent=2))
    print(json.dumps(out["power_of_current_test_set"], indent=2))
    print(json.dumps(out["min_detectable_gain_points_at_80pct_power"], indent=2))


def render_md(o):
    p, a, m = o["proposal"], o["assumed_inputs"], o["measured_inputs"]
    L = ["# Held-out test-size proposal (power calculation; generated by scripts/power_calc.py)", "",
         f"Target: detect a **{int(a['delta'] * 100)}-point** gain of shortcut-aware over matched random, two-sided alpha {a['alpha']}, power {int(a['power'] * 100)}%, normal approximation.",
         "**Assumed (not measured):** discordance rate psi (share of items where the two arms disagree) and seed-to-seed sd of accuracy. **Measured:** paired coverage of the held-out questions "
         f"({m['paired_heldout_questions_now']} of {m['current_heldout_test_questions']} = {m['paired_coverage_of_heldout']}).", "",
         "## Proposal (planning case: " + ", ".join(f"{k}={v}" for k, v in p["headline_case"].items()) + ")", "",
         f"- Plain accuracy on ALL held-out questions: **{p['plain_questions_needed']}** questions.",
         f"- Joint score on the PAIRED subset: **{p['joint_paired_questions_needed']}** paired questions; at the measured coverage that needs more scenes than the plain endpoint.",
         f"- **Proposed held-out test scenes: {p['held_out_test_scenes_proposed']}** (one held-out question per scene), vs {m['current_heldout_test_questions']} now. If paired coverage were raised to 0.9: {p['scenes_needed_if_paired_coverage_were_0.9']}.",
         f"- Report plain accuracy (all questions) and joint score (paired subset) as two separate endpoints, each with its own n and interval.", "",
         "## What the current test set can detect (80% power)", ""]
    for k, v in o["min_detectable_gain_points_at_80pct_power"].items():
        L.append(f"- {k}: {v} points")
    L += ["", "## Power of the current test set for a 3-point gain", "", "| endpoint | " + " | ".join(next(iter(o["power_of_current_test_set"].values())).keys()) + " |", "|---|" + "---|" * 9]
    for k, v in o["power_of_current_test_set"].items():
        L.append(f"| {k} | " + " | ".join(str(x) for x in v.values()) + " |")
    L += ["", "## Seed noise sets a ceiling", "", f"Seed sd ceiling in points (above it no test size reaches 80% power): {p['seed_noise_ceiling_points']}. {p['seeds']}", "",
          "## Items needed, plain accuracy (rows: psi, seed sd, seeds)", "", "| psi | seed sd | seeds | items needed |", "|---|---|---|---|"]
    for r in o["plain_accuracy_n_needed"]:
        L.append(f"| {r['psi']} | {r['sigma_seed']} | {r['k_seeds']} | {r['n_needed_80pct_power'] if r['n_needed_80pct_power'] else 'unreachable'} |")
    L += ["", "## Items needed, joint score on paired subset", "", "| psi | seed sd | seeds | paired items needed |", "|---|---|---|---|"]
    for r in o["joint_score_paired_subset_n_needed"]:
        L.append(f"| {r['psi']} | {r['sigma_seed']} | {r['k_seeds']} | {r['n_needed_80pct_power'] if r['n_needed_80pct_power'] else 'unreachable'} |")
    L += ["", o["assumed_inputs"]["note"] + "."]
    return "\n".join(L) + "\n"


def _pts(x):
    return None if x is None else round(100 * x, 2)


if __name__ == "__main__":
    main()
