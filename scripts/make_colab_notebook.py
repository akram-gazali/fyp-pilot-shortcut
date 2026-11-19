"""Build colab/stage4_feasibility.ipynb (a thin wrapper around colab/stage4_feasibility.py)."""
import json
from siftvqa.config import ROOT

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}

cells = [
    md("""
# SIFT-VQA Stage 4: model feasibility (Qwen2.5-VL-3B-Instruct + LoRA)
**Feasibility check only, not evidence for the hypothesis.** Written but not yet run on a GPU: expect small fixes on the first run.
Before you start: Runtime > Change runtime type > **GPU** (T4 is fine); add Colab secret **HF_TOKEN** (key icon, left sidebar, turn on *Notebook access*).
Never paste a token into a cell. Full steps: `colab/COLAB_STEPS.md`.
"""),
    code("""
!nvidia-smi
"""),
    md("## 1. Get the code"),
    code("""
import os, subprocess
REPO = "akram-gazali/fyp-pilot-shortcut"
DEST = "/content/fyp-pilot-shortcut"
if not os.path.exists(DEST):
    url = f"https://github.com/{REPO}.git"
    tok = None
    try:
        from google.colab import userdata
        tok = userdata.get("GH_TOKEN")      # only needed if the repo is private; read-only token
    except Exception:
        pass
    cmd = ["git", "clone", f"https://x-access-token:{tok}@github.com/{REPO}.git" if tok else url, DEST]
    r = subprocess.run(cmd, capture_output=True, text=True)           # output hidden so a token cannot leak
    if r.returncode:
        raise SystemExit("clone failed. If the repo is private, add a read-only Colab secret GH_TOKEN and rerun.")
    subprocess.run(["git", "-C", DEST, "remote", "set-url", "origin", url], check=True)   # drop token from .git/config
%cd /content/fyp-pilot-shortcut
"""),
    code("""
!pip -q install "transformers>=4.49" peft accelerate bitsandbytes
!pip -q install -e .
"""),
    code("""
# check the secret exists WITHOUT printing it
from google.colab import userdata
print("HF_TOKEN secret found:", bool(userdata.get("HF_TOKEN")))
"""),
    md("## 2. Rebuild the data (CPU, ~10-20 min; checks hashes against the committed manifests)"),
    code("""
!python colab/stage4_feasibility.py prepare
"""),
    md("## 3. Zero-shot reference on the held-out set\nSmoke test first (64 questions), then the full run."),
    code("""
!python colab/stage4_feasibility.py zero-shot --max-questions 64 --out reports/stage4_smoke
"""),
    code("""
!python colab/stage4_feasibility.py zero-shot
"""),
    md("## 4. Memory and speed grid (LoRA training step)\nIf a row shows OOM everywhere, rerun with `--load-4bit`."),
    code("""
!python colab/stage4_feasibility.py grid
"""),
    md("## 5. Cost extrapolation (uses grid.json and zero_shot.json)"),
    code("""
!python colab/stage4_feasibility.py cost
"""),
    md("## 6. OPTIONAL: tiny fine-tune per arm (run only if the cost table says it is cheap)\nOne seed, 400 examples, 1 epoch per arm. Proves the code path runs, nothing more."),
    code("""
!python colab/stage4_feasibility.py tiny-finetune
"""),
    md("## 7. Save results to Drive (then copy into the repo's `reports/stage4/` on your computer and commit; no tokens)"),
    code("""
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/siftvqa_stage4 && cp -r reports/stage4/. /content/drive/MyDrive/siftvqa_stage4/
!ls -la /content/drive/MyDrive/siftvqa_stage4
"""),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(ROOT / "colab" / "stage4_feasibility.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote colab/stage4_feasibility.ipynb with", len(cells), "cells")
