"""Answer-format helpers for the VLM evaluation (pure Python, CPU-testable).

Two decoding methods are compared on the dev set (train split), never on the test set:
  suffix  per-template instruction + greedy generation, answer parsed from the text
  choice  same prompt, one forward pass, argmax over the log-probability of each candidate answer
          (no generation, so no wrong-format output is possible by construction)
'legacy' (old one-size suffix, free generation) is kept only as a diagnostic.
Training and evaluation use the SAME prompt text (`prompt_text`), so a fine-tuned model sees at test
time exactly the format it was trained on.
"""
import json
import os
import re
from pathlib import Path

COUNT_DIGITS = [str(i) for i in range(9)]          # candidates 0..8 (generator answers are 0..5)
YESNO = ["yes", "no"]
LEGACY_SUFFIX = " Answer with a single word or number."
SUFFIX_YESNO = " Answer with only yes or no."
SUFFIX_NUMBER = " Answer with only a number."


def answer_kind(template):
    """count_cs -> number; every other template (exists_cs, compare_shape, compare_cs) -> yes/no."""
    return "number" if template.startswith("count") else "yesno"


def prompt_text(question, template, method="suffix"):
    if method == "legacy":
        return question + LEGACY_SUFFIX
    return question + (SUFFIX_NUMBER if answer_kind(template) == "number" else SUFFIX_YESNO)


def candidates(template):
    return COUNT_DIGITS if answer_kind(template) == "number" else YESNO


def surface_forms(answer):
    """Strings whose first token counts as this answer when scoring (case variants of yes/no)."""
    return [answer, answer.capitalize()] if answer in YESNO else [answer]


def strict_format_ok(text, template):
    """True only if the model output is exactly an allowed answer (ignoring case, spaces, a final period)."""
    t = re.sub(r"[.\s]+$", "", text.strip().lower())
    return t in candidates(template)


def macro(per_template):
    vals = [v for v in per_template.values() if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def pick_method(results, tie_pts=1.0):
    """results: {method: {"macro_acc": float, "sec_per_item": float}} over the compared methods.
    Highest macro dev accuracy wins; anything within `tie_pts` of the best is a tie and the faster one is chosen."""
    best = max(r["macro_acc"] for r in results.values())
    tied = {m: r for m, r in results.items() if best - r["macro_acc"] <= tie_pts}
    chosen = min(tied, key=lambda m: (tied[m]["sec_per_item"], m))
    return {"chosen": chosen, "best_macro_acc": best, "tied_methods": sorted(tied), "tie_pts": tie_pts}


def pick_resolution(acc_by_res, base=224, margin_pts=2.0):
    """Use the native resolution unless a larger one beats it by MORE than `margin_pts` macro-accuracy points."""
    better = {r: a for r, a in acc_by_res.items() if r != base and a - acc_by_res[base] > margin_pts}
    chosen = max(better, key=lambda r: better[r]) if better else base
    return {"chosen": chosen, "base": base, "margin_pts": margin_pts, "acc_by_res": acc_by_res}


# ----------------------------------------------------------------------------- environment / pins
def parse_pins(text):
    """Lines like `transformers==5.22.1` -> {name: version}. Comments and blanks ignored."""
    pins = {}
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, ver = line.split("==", 1)
            pins[name.strip().lower().replace("_", "-")] = ver.strip()
    return pins


def version_mismatches(pins, installed):
    """pins/installed: {name: version}. Returns list of (name, pinned, installed_or_None). torch is never enforced."""
    out = []
    for name, want in pins.items():
        if name == "torch":
            continue
        have = installed.get(name)
        if have != want:
            out.append((name, want, have))
    return out


def installed_versions(names):
    from importlib import metadata
    got = {}
    for n in names:
        try:
            got[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            got[n] = None
    return got


def check_pins(path):
    """Print installed versions; raise SystemExit if a pinned non-torch package differs."""
    pins = parse_pins(Path(path).read_text())
    inst = installed_versions(list(pins) + ["torch"])
    print("library versions:", json.dumps(inst))
    bad = version_mismatches(pins, inst)
    if inst.get("torch") and pins.get("torch") and inst["torch"] != pins["torch"]:
        print(f"WARNING: torch {inst['torch']} differs from the run environment ({pins['torch']}); torch is not enforced.")
    if bad:
        raise SystemExit("version mismatch (name, pinned, installed): %s\nRun scripts/install_pinned.py first." % bad)
    return inst


def choose_dtype(requested, compute_capability_major):
    """Never assume bf16: auto means fp16 on GPUs below compute capability 8.0 (T4 = 7.5), bf16 otherwise."""
    if requested in ("fp16", "bf16"):
        return requested
    return "bf16" if compute_capability_major >= 8 else "fp16"


def write_json_atomic(path, obj):
    """Write to a temp file then rename, so a killed session never leaves a half-written result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def best_candidate(token_logprobs, cand_ids):
    """token_logprobs: {token_id: log-prob at the answer position}; cand_ids: {answer: [token ids of its surface forms]}.
    Score of an answer = log-sum-exp of its surface forms; returns the best answer and all scores."""
    import math
    scores = {}
    for ans, ids in cand_ids.items():
        vals = [token_logprobs[i] for i in ids]
        m = max(vals)
        scores[ans] = m + math.log(sum(math.exp(v - m) for v in vals))
    return max(scores, key=scores.get), scores
