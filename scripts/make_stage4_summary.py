"""Stage 4 summary: renders reports/stage4_summary.md from the JSON files Colab produced
(reports/stage4/*.json). No number is typed by hand; missing files are reported as 'not run'."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "reports"
S4 = R / "stage4"


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    z, h, grid, cost, tiny = (load(S4 / f) for f in
                              ("zero_shot.json", "prepare_hash_check.json", "grid.json",
                               "cost_extrapolation.json", "tiny_finetune.json"))
    base = load(R / "stage3_leakage.json")["question_only_baseline"]
    L = ["# Stage 4 summary (Colab feasibility)", "",
         "All numbers are PILOT numbers from one run, generated from `reports/stage4/*.json`.", ""]
    if h:
        L += ["## Data hashes", f"Colab rebuild matches committed manifests: `{h['match']}`", "",
              f"Environment: `{h['env']}`", ""]
    if z:
        p, j, sh = z["plain_accuracy_all_questions"], z["joint_score_paired_subset"], z["shuffled_image_baseline"]
        L += [f"## Zero-shot reference ({z['model']}, {z['resolution']} px, greedy)", "",
              "| Plain accuracy, all questions | % |", "|---|---|",
              f"| held-out (compare_cs), n={p['n_held_out']} | {p['held_out_pct']} |",
              f"| seen templates, n={p['n_seen']} | {p['seen_pct']} |",
              f"| all, n={p['n_all']} | {p['all_pct']} |"]
        L += [f"| {t} | {v} |" for t, v in p["by_template_pct"].items()]
        L += ["", "| Joint score, paired subset | n | joint % | original % | relevant edit % | irrelevant edit % |",
              "|---|---|---|---|---|---|"]
        L += [f"| {k} | {v['n_paired']} | {v['joint_pct']} | {v['original_pct']} | {v['relevant_edit_pct']} | {v['irrelevant_edit_pct']} |"
              for k, v in j.items()]
        L += ["", f"Shuffled-image control: all {sh['all_pct']}%, seen {sh['seen_pct']}%, held-out {sh['held_out_pct']}%.",
              f"Question-only majority baselines (Stage 3): `{base}`.",
              f"Speed: {z['timing']['sec_per_example']} s/example at batch {z['timing']['batch_size']} "
              f"({z['timing']['n_items']} items). Unparsed outputs: {z['unparsed_outputs']}.", "",
              "Caveat: the normaliser takes the first yes/no/number in the output, so replies such as "
              "`0` or `Yes, 1` to a yes/no question are scored wrong. Low exists/compare accuracy may "
              "be answer-format failure, not ability. See `raw_output_examples` in zero_shot.json.", ""]
    for name, d in (("Memory/speed grid", grid), ("Cost extrapolation", cost), ("Tiny fine-tune", tiny)):
        L += [f"## {name}", "```json", json.dumps(d, indent=1), "```" if d else "NOT RUN yet", ""] if d else [f"## {name}", "NOT RUN yet", ""]
    (R / "stage4_summary.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
