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
## Stage 4 (GPU needed; Colab or Kaggle) and the final report
Colab step list: `colab/COLAB_STEPS.md`; notebook: `colab/stage4_feasibility.ipynb` (regenerate with `scripts/make_colab_notebook.py`;
do not use Colab's "Save a copy to GitHub", it overwrites the notebook). On the GPU machine (needs `pip uninstall -y torchao` on Colab):
```bash
python colab/stage4_feasibility.py prepare      # rebuild data, check hashes against data/manifests
python colab/stage4_feasibility.py zero-shot    # frozen-model reference (headroom), shuffled-image control
python colab/stage4_feasibility.py grid         # memory/speed grid (writes grid.json only when it finishes)
python colab/stage4_feasibility.py cost         # GPU-hour extrapolation (CPU is enough)
python colab/stage4_feasibility.py tiny-finetune  # OPTIONAL, ~16 GPU-hours on a T4, never run yet
```
Copy `reports/stage4/*.json` back into the repo. Then, locally (CPU):
```bash
.venv/bin/python scripts/power_calc.py            # held-out test-size proposal
.venv/bin/python scripts/make_stage4_summary.py   # reports/stage4_summary.md
.venv/bin/python scripts/make_pilot_report.py     # reports/pilot_report.md (reads only the JSON above)
```
Status: zero-shot, grid and cost are measured on a Colab T4. The `grid.json` in this repo was transcribed from the printed log
(the original file was lost in a runtime restart) and is labelled as such. The tiny fine-tune has not been run.

Optional: `scripts/tune_generator.py` (generator tuning on separate seeds, no rendering).

## Layout
`src/siftvqa/` library (dsl, executor, scene, templates, audit, shortcuts, splits, leakage, edits, selectors),
`configs/` (generator, shortcuts [frozen, hashed], splits, selection), `scripts/` (one CLI per step),
`tests/`, `reports/`, `data/manifests/` (small tracked manifests with config and shortcut-file hashes).
