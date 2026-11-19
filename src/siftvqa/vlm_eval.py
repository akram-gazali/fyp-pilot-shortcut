"""Scoring of model answers: answer normalisation, plain accuracy, and the paired-subset joint
score. No model or GPU needed here, so it is unit-tested locally. The Colab script calls it.

Two endpoints are always kept separate:
  plain accuracy  - original image, ALL questions
  joint score     - PAIRED subset only: original, relevant edit and irrelevant edit all correct
"""
import re
from collections import defaultdict

from .edits import joint_correct

_WORDS = {"zero": "0", "none": "0", "no objects": "0", "one": "1", "two": "2", "three": "3", "four": "4",
          "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def normalize_answer(text):
    """Map free-form model output to the canonical answer string ("yes"/"no"/digits).
    Takes the first yes/no/number found; anything else is returned stripped (and scores wrong)."""
    t = text.strip().lower()
    m = re.search(r"\b(yes|no)\b", t)
    d = re.search(r"\b\d+\b", t)
    w = re.search(r"\b(" + "|".join(k for k in _WORDS if " " not in k) + r")\b", t)
    cands = [(x.start(), x.group(1) if x is m or x is w else x.group(0)) for x in (m, d, w) if x]
    if not cands:
        return t
    _, val = min(cands)
    return _WORDS.get(val, val)


def accuracy(rows, preds):
    """% correct over rows (dicts with qid, answer); preds: qid -> normalised answer. Missing = wrong."""
    return round(100 * sum(preds.get(r["qid"]) == r["answer"] for r in rows) / len(rows), 2) if rows else None


def plain_accuracy_report(test_rows, preds, held_out_template):
    """Plain accuracy on ALL questions, split seen / held-out and by template."""
    seen = [r for r in test_rows if r["template"] != held_out_template]
    held = [r for r in test_rows if r["template"] == held_out_template]
    by_t = defaultdict(list)
    for r in test_rows:
        by_t[r["template"]].append(r)
    return {"n_all": len(test_rows), "all_pct": accuracy(test_rows, preds),
            "seen_pct": accuracy(seen, preds), "n_seen": len(seen),
            "held_out_pct": accuracy(held, preds), "n_held_out": len(held),
            "by_template_pct": {t: accuracy(rs, preds) for t, rs in sorted(by_t.items())}}


def joint_report(edit_recs, preds_by_scene, held_out_template):
    """Joint score on the paired subset. preds_by_scene: key -> normalised answer, where the keys
    are the original qid, relevant_scene id and irrelevant_scene id of each edit record.
    Records without all three predictions count as wrong."""
    def one(recs):
        if not recs:
            return None
        ok = [joint_correct(preds_by_scene.get(r["qid"]), preds_by_scene.get(r["relevant_scene"]),
                            preds_by_scene.get(r["irrelevant_scene"]), r["answers"]) for r in recs]
        orig = [preds_by_scene.get(r["qid"]) == r["answers"]["original"] for r in recs]
        rel = [preds_by_scene.get(r["relevant_scene"]) == r["answers"]["relevant"] for r in recs]
        irr = [preds_by_scene.get(r["irrelevant_scene"]) == r["answers"]["irrelevant"] for r in recs]
        pc = lambda xs: round(100 * sum(xs) / len(xs), 2)
        return {"n_paired": len(recs), "joint_pct": pc(ok), "original_pct": pc(orig),
                "relevant_edit_pct": pc(rel), "irrelevant_edit_pct": pc(irr)}
    held = [r for r in edit_recs if r["template"] == held_out_template]
    seen = [r for r in edit_recs if r["template"] != held_out_template]
    return {"held_out": one(held), "seen": one(seen), "all_paired": one(edit_recs)}
