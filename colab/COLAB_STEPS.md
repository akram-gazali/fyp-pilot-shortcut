# Stage 5 on Colab (alternative to Kaggle; use `kaggle/KAGGLE_STEPS.md` first)

The GPU code has not been run since the Stage 5 rewrite; the first run may need a small fix (send me the last 15 to 20 lines of the error, never a token).
Colab free tier can refuse a GPU ("Cannot connect to GPU backend"); then use Kaggle.

1. Colab > File > Open notebook > GitHub tab > `akram-gazali/fyp-pilot-shortcut` > `colab/stage4_feasibility.ipynb` (regenerated from `scripts/make_colab_notebook.py`).
2. Runtime > Change runtime type > **T4 GPU**. No Hugging Face token is needed (the model is public).
3. Run the cells in order. Each long step ends with `save()`, which copies `reports/stage5/` to `MyDrive/siftvqa_stage5/`, so a runtime restart cannot lose results (that is how `grid.json` was lost in Stage 4, and why it was transcribed by hand).
   Expected: `hash match ... True` for all keys after `prepare`; `CHOSEN: <method> <resolution>` after `dev-format`; one dict per row from `grid` (OOM rows are normal).
4. **Never use Colab's "Save a copy to GitHub"**: it replaces the generated notebook with an executed copy (this already happened once, commit dfc8ee7).
5. Download the JSON files from Drive into the repo's `reports/stage5/`, then `python scripts/make_pilot_report.py`, commit and push (no tokens, no data).

Tiny fine-tune (OPTIONAL, last cell): read `reports/stage5/cost_extrapolation.json` first. It prints one result per arm and saves after each arm.
CPU-side check before any GPU run: `pytest tests/test_stage5_train.py -s` prints the LoRA dtype check (adapters fp32 on an fp16 base, GradScaler step, no LoRA in the vision tower) and the loss-mask test.
