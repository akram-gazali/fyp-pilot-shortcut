# What you need to do in Colab (Stage 4)

Stage 4 is written but **has not been run on a GPU**. The first run may need small fixes (library versions); send me the error text if so.
This is a feasibility check. It does not test the hypothesis.

## 0. One-time setup (about 5 minutes)
1. **Hugging Face token.** huggingface.co > Settings > Access Tokens > New token, type **Read**. Copy it once.
2. **Colab secret.** In Colab, left sidebar > key icon (Secrets) > Add new secret: name `HF_TOKEN`, value = the token, switch **Notebook access** on.
   Never paste the token into a cell, a file, or git. The scripts read it from the secret and never write it anywhere.
3. If the model download fails with 401/403, open `huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct` while logged in and accept any licence prompt, then rerun.
4. **Only if the GitHub repo is private:** create a fine-grained token with read-only *Contents* access to this one repo and add it as Colab secret `GH_TOKEN`. (Public repo: skip.)

## 1. Open the notebook with a GPU
1. Colab > File > Open notebook > GitHub tab > `akram-gazali/fyp-pilot-shortcut` > `colab/stage4_feasibility.ipynb`.
2. Runtime > Change runtime type > **T4 GPU** (free tier works; an A100/L4 on Pro is faster, but numbers then describe that GPU).

## 2. Run the cells in order (or use the commands below in a terminal/`!` cell)
```bash
python colab/stage4_feasibility.py prepare                      # rebuild data on CPU, check hashes (~10-20 min)
python colab/stage4_feasibility.py zero-shot --max-questions 64 --out reports/stage4_smoke   # smoke test (minutes)
python colab/stage4_feasibility.py zero-shot                    # frozen reference: plain accuracy + joint score + shuffled-image
python colab/stage4_feasibility.py grid                         # peak memory and seconds/step; add --load-4bit if everything is OOM
python colab/stage4_feasibility.py cost                         # GPU-hour extrapolation vs the 20 h budget
python colab/stage4_feasibility.py tiny-finetune                # OPTIONAL, only if `cost` says it is cheap
```
Useful flags: `--resolution 336` (default 448), `--batch-size 8` (inference), `--held-out-only`, `--skip-edits` (on `prepare`, if you only need plain accuracy),
`--exists-mode fixed_shared_base` (tiny fine-tune), `--dry-run` (no GPU, checks plumbing).
Time guesses (ASSUMED until measured): prepare 10-20 min; smoke test a few minutes; full zero-shot is ~7,300 forward passes (originals, edited images, shuffled images), likely 20-60 min on a T4; grid 10-20 min.

## 3. Get the results back (no tokens in git)
Run the last notebook cell: it copies `reports/stage4/` to `MyDrive/siftvqa_stage4/`. Then on your computer copy those files into the repo's `reports/stage4/` and:
```bash
git add reports/stage4 && git commit -m "Stage 4 Colab results" && git push
```
Files: `prepare_hash_check.json`, `zero_shot.json`, `grid.json`, `cost_extrapolation.json`, optionally `tiny_finetune.json`, plus `run_meta.json` (config, git commit, GPU).

## 4. What to send me
The first lines of any error, or the four JSON files. I will then write the Stage 4 section of the pilot report from them (numbers come from the JSON only).

## Things to watch
- `prepare_hash_check.json` should say every hash matches. If not, the Colab Python differs from your Mac's and results are not comparable to the local Stage 1-3 numbers.
- `zero_shot.json` > `unparsed_outputs` should be near 0; a large number means the answer parser needs a fix.
- Free Colab can disconnect. Copy results to Drive after each step.
