"""Write run metadata (config + git commit) next to every run's outputs."""
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT


def git_info():
    def run(*args):
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD") or None,
        "dirty": bool(run("status", "--porcelain")),
    }


def write_run_meta(out_dir, config, stage, extra=None):
    """Write <out_dir>/run_meta.json with the config used and the git commit hash."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "stage": stage,
        "git": git_info(),
        "python": platform.python_version(),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        **(extra or {}),
    }
    path = out_dir / "run_meta.json"
    path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return path
