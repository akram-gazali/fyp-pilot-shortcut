"""Stage 3a: scene-disjoint train / pool / test splits, held-out signature, test questions, leakage report.

Usage: python scripts/build_splits.py
"""
import json
from pathlib import Path

from siftvqa import dsl
from siftvqa.config import ROOT, load_config, sha256_file
from siftvqa.executor import answer_to_str, execute
from siftvqa.leakage import (duplicate_questions_across_splits, duplicate_scenes, question_only_baseline,
                             scene_group_overlap, shuffled_scene_accuracy, signature_overlap)
from siftvqa.runmeta import write_run_meta
from siftvqa.seeding import make_rng
from siftvqa.shortcuts import load_shortcuts, score_question
from siftvqa.splits import assign_scenes, signature, template_signature
from siftvqa.templates import TEMPLATES, build_program, question_text, sample_params

POOL = ROOT / "data" / "pool"
OUT = ROOT / "data" / "splits"


def scene_feats(scene):
    return {"n_objects": len(scene["objects"]),
            "n_distinct_colors": len({o["color"] for o in scene["objects"]}),
            "n_distinct_shapes": len({o["shape"] for o in scene["objects"]})}


def main():
    scfg = load_config(ROOT / "configs" / "splits.yaml")
    gcfg = load_config(ROOT / "configs" / "generator.yaml")
    shortcuts = load_shortcuts(ROOT / "configs" / "shortcuts.yaml")
    held_sig = template_signature(scfg["held_out_template"])
    held_t = scfg["held_out_template"]

    scored = [json.loads(l) for l in open(POOL / "scored.jsonl")]
    scene_ids = sorted({r["scene_id"] for r in scored})
    split_of = assign_scenes(scene_ids, scfg)
    scenes = {sid: json.loads((POOL / "scenes" / f"{sid}.json").read_text()) for sid in split_of}

    rows = {"train": [], "pool": [], "test": []}
    dropped_heldout = {"train": 0, "pool": 0}
    for r in scored:
        sp = split_of.get(r["scene_id"])
        if sp in ("train", "pool"):
            r = {**r, "signature": signature(r["program"]), **scene_feats(scenes[r["scene_id"]]), "split": sp}
            if r["signature"] == held_sig:
                dropped_heldout[sp] += 1      # held-out combination never enters train/pool
                continue
            r.pop("shortcut_hash", None)
            rows[sp].append(r)

    # test: one question per template per test scene (held-out template included)
    for sid, sp in sorted(split_of.items()):
        if sp != "test":
            continue
        scene = scenes[sid]
        for t in TEMPLATES:
            for k in range(scfg["test_questions_per_template_per_scene"]):
                params = sample_params(t, scene, gcfg, make_rng(scfg["seed"], "testq", sid, t, k))
                prog = build_program(t, params)
                q = {"qid": f"{sid}_t_{t}_{k}", "scene_id": sid, "template": t, "params": params,
                     "program": dsl.to_dict(prog), "program_str": dsl.to_str(prog), "program_length": dsl.length(prog),
                     "question": question_text(prog), "answer": answer_to_str(execute(prog, scene))}
                s = score_question(q, scene, shortcuts)
                rows["test"].append({**q, **s, "signature": signature(prog), **scene_feats(scene), "split": "test"})

    OUT.mkdir(parents=True, exist_ok=True)
    for sp, rs in rows.items():
        (OUT / f"{sp}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    (OUT / "scene_split.json").write_text(json.dumps(split_of))

    scenes_by_split = {sp: {sid: scenes[sid] for sid, s in split_of.items() if s == sp} for sp in rows}
    test_scene_by_id = {sid: scenes[sid] for sid in scenes_by_split["test"]}
    leak = {
        "pilot_note": "pilot numbers; model-free checks only",
        "split_sizes": {sp: {"scenes": len(scenes_by_split[sp]), "questions": len(rs)} for sp, rs in rows.items()},
        "held_out_questions_dropped_from_train_pool": dropped_heldout,
        "signature_overlap": signature_overlap(rows, held_sig),
        "scene_group_overlap_counts": scene_group_overlap(rows),
        "duplicate_scenes": duplicate_scenes(scenes_by_split),
        "duplicate_questions_across_splits": duplicate_questions_across_splits(rows, scenes_by_split),
        "question_only_baseline": question_only_baseline(rows["train"], rows["test"], held_t),
        "shuffled_scene_oracle_acc_pct": shuffled_scene_accuracy(rows["test"], test_scene_by_id, held_t, scfg["seed"]),
        "test_answer_dist_by_template": {
            t: dict(sorted(__import__("collections").Counter(r["answer"] for r in rows["test"] if r["template"] == t).items()))
            for t in TEMPLATES},
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "stage3_leakage.json").write_text(json.dumps(leak, indent=2))
    write_run_meta(OUT, {"splits": scfg}, stage="3-splits")
    man = {
        "stage": "3-splits", "held_out_template": held_t, "held_out_signature": held_sig,
        "shortcuts_sha256": sha256_file(ROOT / "configs" / "shortcuts.yaml"),
        "splits_config_sha256": sha256_file(ROOT / "configs" / "splits.yaml"),
        "generator_config_sha256": sha256_file(ROOT / "configs" / "generator.yaml"),
        "questions_sha256": sha256_file(POOL / "questions.jsonl"),
        **{f"{sp}_sha256": sha256_file(OUT / f"{sp}.jsonl") for sp in rows},
        "sizes": leak["split_sizes"],
    }
    (ROOT / "data" / "manifests" / "stage3_splits_manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps(leak, indent=2))


if __name__ == "__main__":
    main()
