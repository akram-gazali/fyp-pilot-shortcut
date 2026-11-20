"""Install the pinned non-torch packages from requirements-gpu.txt WITHOUT touching torch.

torch is frozen with a constraint at the version already installed on this machine. If a pin cannot be
satisfied with that torch, pip fails and this script stops and reports (it never forces torch).
Also removes torchao (Colab ships 0.10, which peft rejects).
Usage: python scripts/install_pinned.py
"""
import importlib.metadata as md
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    try:
        torch_v = md.version("torch")
    except md.PackageNotFoundError:
        sys.exit("torch is not installed on this machine; use a GPU runtime image that ships torch.")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(f"torch=={torch_v}\n")
        cons = f.name
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements-gpu.txt"), "-c", cons],
                       capture_output=True, text=True)
    if r.returncode:
        print((r.stdout + r.stderr)[-3000:])
        sys.exit(f"STOP: a pin in requirements-gpu.txt conflicts with this machine's torch {torch_v}. "
                 "Not forcing torch. Send me the lines above.")
    chk = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    print("pip check:", (chk.stdout + chk.stderr).strip()[-1500:] or "ok")
    print(f"pinned packages installed; torch left at {torch_v}")


if __name__ == "__main__":
    main()
