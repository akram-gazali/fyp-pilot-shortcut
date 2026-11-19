"""Stage 1: generate scenes (image + scene JSON side by side), questions, and run the label audit.

Usage: python scripts/gen_pool.py [--config configs/generator.yaml] [--out data/pool] [--n N]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from siftvqa import dsl
from siftvqa.audit import audit_scene
from siftvqa.config import ROOT, load_config, sha256_file
from siftvqa.executor import answer_to_str, execute
from siftvqa.runmeta import write_run_meta
from siftvqa.scene import render, sample_scene
from siftvqa.seeding import make_rng
from siftvqa.templates import TEMPLATES, build_program, question_text, sample_params


def make_questions(scene, cfg, rng):
    weights = [cfg["templates"][t] for t in TEMPLATES]
    out = []
    for i in range(cfg["questions_per_scene"]):
        t = rng.choices(TEMPLATES, weights)[0]
        params = sample_params(t, scene, cfg, rng)
        prog = build_program(t, params)
        out.append(
            {
                "qid": f"{scene['scene_id']}_q{i}",
                "scene_id": scene["scene_id"],
                "template": t,
                "params": params,
                "program": dsl.to_dict(prog),
                "program_str": dsl.to_str(prog),
                "program_length": dsl.length(prog),
                "question": question_text(prog),
                "answer": answer_to_str(execute(prog, scene)),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "generator.yaml"))
    ap.add_argument("--out", default=str(ROOT / "data" / "pool"))
    ap.add_argument("--n", type=int, default=None, help="override n_scenes")
    ap.add_argument("--no-audit", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.n:
        cfg["n_scenes"] = args.n
    out = Path(args.out)
    scene_dir = out / "scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)

    all_q, mismatches = [], []
    for i in range(cfg["n_scenes"]):
        sid = f"s{i:05d}"
        rng = make_rng(cfg["seed"], "scene", i)
        scene = sample_scene(cfg, rng, sid)
        img = render(scene)
        img.save(scene_dir / f"{sid}.png")
        (scene_dir / f"{sid}.json").write_text(json.dumps(scene))
        qs = make_questions(scene, cfg, make_rng(cfg["seed"], "questions", i))
        all_q.extend(qs)
        if not args.no_audit:
            mismatches += [f"{sid}: {m}" for m in audit_scene(img, scene, qs)]

    qpath = out / "questions.jsonl"
    qpath.write_text("".join(json.dumps(q) + "\n" for q in all_q))

    audit = {"ran": not args.no_audit, "n_scenes": cfg["n_scenes"], "n_questions": len(all_q),
             "n_mismatches": len(mismatches), "mismatches": mismatches[:50]}
    (out / "label_audit.json").write_text(json.dumps(audit, indent=2))
    write_run_meta(out, cfg, stage="1-generate")

    manifest = {
        "stage": "1",
        "n_scenes": cfg["n_scenes"],
        "n_questions": len(all_q),
        "questions_sha256": sha256_file(qpath),
        "generator_config_sha256": sha256_file(args.config),
        "template_counts": dict(Counter(q["template"] for q in all_q)),
        "label_audit_mismatches": None if args.no_audit else len(mismatches),
    }
    mdir = ROOT / "data" / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "stage1_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
