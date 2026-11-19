# SIFT-VQA pilot

Pilot pipeline for shortcut-aware example selection in VQA on synthetic 2D scenes.
All pilot defaults live in `configs/` (not final until the protocol lock). Every run writes
`run_meta.json` (config + git commit) next to its outputs. All numbers in `reports/` are produced by code.

## Setup
```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest
```

## Reproduce Stages 1-3 (CPU, local, ~10 minutes; `data/` is git-ignored except small manifests)
```bash
.venv/bin/python scripts/gen_pool.py              # Stage 1: scenes, questions, pixel-based label audit
.venv/bin/python scripts/answer_stats.py data/pool/questions.jsonl reports/answer_stats_tuned.json
.venv/bin/python scripts/score_pool.py            # Stage 2: shortcut answers, d flag, disagreement report
.venv/bin/python scripts/make_stage2_summary.py   # reports/stage2_summary.md
.venv/bin/python scripts/build_splits.py          # Stage 3: splits, held-out signature, leakage checks
.venv/bin/python scripts/build_edits.py           # Stage 3: paired relevant/irrelevant test edits (~6 min)
.venv/bin/python scripts/run_selectors.py         # Stage 3: four selectors x 3 seeds x both exists modes
.venv/bin/python scripts/make_stage3_summary.py   # reports/stage3_summary.md
```
Optional: `scripts/tune_generator.py` (generator tuning on separate seeds, no rendering).

## Layout
`src/siftvqa/` library (dsl, executor, scene, templates, audit, shortcuts, splits, leakage, edits, selectors),
`configs/` (generator, shortcuts [frozen, hashed], splits, selection), `scripts/` (one CLI per step),
`tests/`, `reports/`, `data/manifests/` (small tracked manifests with config and shortcut-file hashes).
