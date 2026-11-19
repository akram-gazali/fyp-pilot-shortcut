"""Shortcut programs: plausible wrong rules, derived from the intended program.

A shortcut is the intended program with one filter removed. It is executed with the
same executor, so shortcut answers are computed, never typed in.
"""
from dataclasses import replace

from . import dsl
from .config import load_config
from .executor import answer_to_str, execute


def load_shortcuts(path):
    return load_config(path)["shortcuts"]


def apply_rule(p, rule):
    if isinstance(p, dsl.Scene):
        return p
    if isinstance(p, dsl.FilterColor) and rule == "drop_color":
        return apply_rule(p.child, rule)
    if isinstance(p, dsl.FilterShape) and rule == "drop_shape":
        return apply_rule(p.child, rule)
    if isinstance(p, (dsl.FilterColor, dsl.FilterShape, dsl.Count, dsl.Exists)):
        return replace(p, child=apply_rule(p.child, rule))
    raise ValueError(f"shortcut rule {rule!r} not defined for {type(p).__name__}")


def score_question(q, scene, shortcuts):
    """Intended answer, each applicable shortcut's answer, and the d flag (any disagreement)."""
    intended = dsl.from_dict(q["program"])
    ans = answer_to_str(execute(intended, scene))
    assert ans == q["answer"], f"{q['qid']}: stored answer differs from executor"
    sc = {}
    for s in shortcuts:
        if s["template"] == q["template"]:
            prog = apply_rule(intended, s["rule"])
            sc[s["name"]] = {"program_str": dsl.to_str(prog), "answer": answer_to_str(execute(prog, scene))}
    flags = {n: int(v["answer"] != ans) for n, v in sc.items()}
    return {"intended": ans, "shortcuts": sc, "d_per_shortcut": flags, "d": int(any(flags.values()))}
