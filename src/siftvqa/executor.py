"""Executor: run a DSL program on a scene JSON. Answers are int or bool."""
from . import dsl


def execute(p, scene):
    p.out_type()  # type-check first
    return _run(p, scene)


def _run(p, scene):
    if isinstance(p, dsl.Scene):
        return list(scene["objects"])
    if isinstance(p, dsl.FilterColor):
        return [o for o in _run(p.child, scene) if o["color"] == p.color]
    if isinstance(p, dsl.FilterShape):
        return [o for o in _run(p.child, scene) if o["shape"] == p.shape]
    if isinstance(p, dsl.Count):
        return len(_run(p.child, scene))
    if isinstance(p, dsl.Exists):
        return len(_run(p.child, scene)) > 0
    if isinstance(p, dsl.GreaterThan):
        return _run(p.left, scene) > _run(p.right, scene)
    raise ValueError(f"unknown node {p!r}")


def answer_to_str(a):
    """Canonical answer string: integers as digits, booleans as yes/no."""
    if isinstance(a, bool):
        return "yes" if a else "no"
    return str(a)
