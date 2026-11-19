"""Stage 3b: paired visual edits (relevant / irrelevant) for every test question; render + audit.

Usage: python scripts/build_edits.py
"""
import json
from collections import Counter

from siftvqa import dsl
from siftvqa.audit import audit_scene
from siftvqa.config import ROOT, load_config, sha256_file
from siftvqa.edits import make_paired_edits
from siftvqa.runmeta import write_run_meta
from siftvqa.scene import render
from siftvqa.seeding import make_rng

SPL, POOL = ROOT / "data" / "splits", ROOT / "data" / "pool"
OUT = ROOT / "data" / "test_edits"


def main():
    gcfg = load_config(ROOT / "configs" / "generator.yaml")
    scfg = load_config(ROOT / "configs" / "splits.yaml")
    OUT.mkdir(parents=True, exist_ok=True)
    test = [json.loads(l) for l in open(SPL / "test.jsonl")]
    recs, problems, no_pair = [], [], Counter()
    kinds = Counter()
    for q in test:
        scene = json.loads((POOL / "scenes" / f"{q['scene_id']}.json").read_text())
        scene["scene_id"] = q["qid"]
        pair = make_paired_edits(scene, dsl.from_dict(q["program"]), gcfg, make_rng(scfg["seed"], "edit", q["qid"]))
        if pair is None:
            no_pair[q["template"]] += 1
            continue
        for which in ("relevant", "irrelevant"):
            sc = pair[which]
            img = render(sc)
            img.save(OUT / f"{sc['scene_id']}.png")
            (OUT / f"{sc['scene_id']}.json").write_text(json.dumps(sc))
            aq = {"qid": sc["scene_id"], "template": q["template"], "params": q["params"], "answer": pair["answers"][which]}
            problems += [f"{sc['scene_id']}: {m}" for m in audit_scene(img, sc, [aq])]
            kinds[(which, sc["edit"]["kind"])] += 1
        recs.append({"qid": q["qid"], "scene_id": q["scene_id"], "template": q["template"], "question": q["question"],
                     "params": q["params"], "answers": pair["answers"],
                     "relevant_scene": pair["relevant"]["scene_id"], "irrelevant_scene": pair["irrelevant"]["scene_id"],
                     "relevant_edit": pair["relevant"]["edit"], "irrelevant_edit": pair["irrelevant"]["edit"]})
    p = SPL / "test_edits.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    rep = {"n_test_questions": len(test), "n_paired": len(recs), "n_without_valid_pair": dict(no_pair),
           "edit_kinds": {f"{a}:{b}": c for (a, b), c in sorted(kinds.items())},
           "audit_mismatches_on_edited_images": len(problems), "audit_examples": problems[:10],
           "relevant_answer_changed": sum(r["answers"]["relevant"] != r["answers"]["original"] for r in recs),
           "irrelevant_answer_same": sum(r["answers"]["irrelevant"] == r["answers"]["original"] for r in recs)}
    (ROOT / "reports" / "stage3_edits.json").write_text(json.dumps(rep, indent=2))
    write_run_meta(OUT, {"generator": gcfg, "splits": scfg}, stage="3-edits")
    (ROOT / "data" / "manifests" / "stage3_edits_manifest.json").write_text(json.dumps(
        {"stage": "3-edits", "test_edits_sha256": sha256_file(p), **{k: v for k, v in rep.items() if k.startswith("n_")}}, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
