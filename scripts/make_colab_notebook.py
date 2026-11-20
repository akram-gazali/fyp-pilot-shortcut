"""Build the GPU notebooks from one source (do not edit the .ipynb files by hand):
  kaggle/stage5_kaggle.ipynb   Kaggle, runs top to bottom ("Save & Run All"), results in /kaggle/working
  colab/stage4_feasibility.ipynb  Colab (kept for later use), results copied to Drive after every long step
NB: Colab's "Save a copy to GitHub" overwrites the .ipynb with an executed copy; always regenerate from this script.
"""
import json
from siftvqa.config import ROOT

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}

REPO = "akram-gazali/fyp-pilot-shortcut"

CLONE = '''
import os, subprocess
REPO = "{repo}"
DEST = "{dest}"
if not os.path.exists(DEST):
    url = f"https://github.com/{{REPO}}.git"
    tok = None
    try:
        {secret_get}
    except Exception:
        pass
    cmd = ["git", "clone", f"https://x-access-token:{{tok}}@github.com/{{REPO}}.git" if tok else url, DEST]
    r = subprocess.run(cmd, capture_output=True, text=True)          # output hidden so a token cannot leak
    if r.returncode:
        raise SystemExit("clone failed. If the repo is private, add a read-only secret GH_TOKEN and rerun.")
    subprocess.run(["git", "-C", DEST, "remote", "set-url", "origin", url], check=True)   # drop token from .git/config
os.chdir(DEST)
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
'''

HEAD_KAGGLE = md("""
# SIFT-VQA Stage 5 (Kaggle): evaluation format, resolution, grid, cost
Runs top to bottom with **Save & Run All** (T4, Internet on). Each result JSON is written as soon as it exists in
`/kaggle/working/stage5_results`. A failing cell stops the run on purpose: send me the last lines of the log.
Feasibility check only, not evidence for the hypothesis. Steps: `kaggle/KAGGLE_STEPS.md`.
""")


def kaggle():
    out = "/kaggle/working/stage5_results"
    cells = [
        HEAD_KAGGLE,
        code('''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # one GPU only, even in a T4 x2 session
os.environ["HF_HOME"] = "/kaggle/temp/hf"         # 7 GB model cache stays out of the saved outputs
!nvidia-smi
!python --version
'''),
        code(CLONE.format(repo=REPO, dest="/kaggle/temp/fyp-pilot-shortcut",
                          secret_get='from kaggle_secrets import UserSecretsClient\n        tok = UserSecretsClient().get_secret("GH_TOKEN")   # only if the repo is private')),
        code("""
# install the pinned NON-torch packages first; torch is never touched. A conflict stops the run and reports.
!python scripts/install_pinned.py
!pip -q install -e .
"""),
        code(f"""
!python scripts/freeze_env.py --out {out}
!python colab/stage4_feasibility.py check-env --out {out}
"""),
        md("## Rebuild the data on the CPU and check hashes (about 10 to 20 minutes)"),
        code(f"!python colab/stage4_feasibility.py prepare --out {out}"),
        md("## Dev-set study (train split): suffix vs choice scoring, then 224 vs 336 px"),
        code(f"!python colab/stage4_feasibility.py dev-format --out {out}"),
        md("## Frozen model on the test set with the chosen method (the headroom number)"),
        code(f"!python colab/stage4_feasibility.py zero-shot --out {out}"),
        md("## Memory and speed grid (writes grid.json after every row)"),
        code(f"!python colab/stage4_feasibility.py grid --out {out}"),
        code(f"!python colab/stage4_feasibility.py cost --out {out}"),
        md("## OPTIONAL tiny fine-tune (one seed, 400 examples, 1 epoch per arm; saved after every arm)\nOff by default. Set `RUN_TINY = True` only after reading the cost table."),
        code(f"""
RUN_TINY = False
if RUN_TINY:
    !python colab/stage4_feasibility.py tiny-finetune --out {out}
"""),
        code(f"!ls -la {out}"),
    ]
    return cells, {"accelerator": "GPU", "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True}}


def colab():
    out = "reports/stage5"
    save = 'save = lambda: __import__("subprocess").run("mkdir -p /content/drive/MyDrive/siftvqa_stage5 && cp -r reports/stage5/. /content/drive/MyDrive/siftvqa_stage5/", shell=True)'
    cells = [
        md("""
# SIFT-VQA Stage 5 (Colab version)
Runtime > Change runtime type > **T4 GPU**. Results are copied to Google Drive after every long step so a runtime restart cannot lose them.
Do NOT use Colab's "Save a copy to GitHub": it overwrites this generated notebook. Steps: `colab/COLAB_STEPS.md`.
"""),
        code("!nvidia-smi"),
        code(f'''
from google.colab import drive
drive.mount('/content/drive')
{save}
'''),
        code(CLONE.format(repo=REPO, dest="/content/fyp-pilot-shortcut",
                          secret_get='from google.colab import userdata\n        tok = userdata.get("GH_TOKEN")   # only if the repo is private')),
        code("""
!python scripts/install_pinned.py
!pip -q install -e .
"""),
        code(f"""
!python scripts/freeze_env.py --out {out}
!python colab/stage4_feasibility.py check-env
save()
"""),
        code("!python colab/stage4_feasibility.py prepare\nsave()"),
        code("!python colab/stage4_feasibility.py dev-format\nsave()"),
        code("!python colab/stage4_feasibility.py zero-shot\nsave()"),
        code("!python colab/stage4_feasibility.py grid\nsave()   # if every row is OOM: add --load-4bit"),
        code("!python colab/stage4_feasibility.py cost\nsave()"),
        md("## OPTIONAL: tiny fine-tune (see the cost table first)"),
        code("!python colab/stage4_feasibility.py tiny-finetune\nsave()"),
    ]
    return cells, {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []}}


def write(path, cells, meta):
    nb = {"cells": cells, "metadata": {**meta, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(nb, indent=1))
    print("wrote", path.relative_to(ROOT), "with", len(cells), "cells")


if __name__ == "__main__":
    write(ROOT / "kaggle" / "stage5_kaggle.ipynb", *kaggle())
    write(ROOT / "colab" / "stage4_feasibility.ipynb", *colab())
