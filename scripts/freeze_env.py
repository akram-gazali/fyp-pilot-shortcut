"""Print the installed versions of every package the GPU script uses, as requirements lines.
Run it on the machine where the pilot ran, then paste the output into requirements-gpu.txt (pins come from the
environment we actually run in, never from memory). Writes reports/stage5/env_freeze.json when --out is given.
Usage: python scripts/freeze_env.py [--out DIR]
"""
import argparse
import json
from importlib import metadata
from pathlib import Path

PKGS = ["torch", "torchvision", "transformers", "peft", "accelerate", "bitsandbytes", "huggingface_hub",
        "safetensors", "tokenizers", "pillow", "numpy", "pyyaml", "qwen-vl-utils"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    got = {}
    for p in PKGS:
        try:
            got[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            got[p] = None
    for p, v in got.items():
        print(f"{p}=={v}" if v else f"# {p}: not installed")
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "env_freeze.json").write_text(json.dumps(got, indent=2))


if __name__ == "__main__":
    main()
