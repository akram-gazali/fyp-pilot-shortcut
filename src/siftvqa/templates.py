"""Question templates. Each template builds a *program*; the English question is
generated from the program (never the other way round)."""
from . import dsl
from .executor import answer_to_str, execute

TEMPLATES = ["count_cs", "exists_cs", "compare_shape", "compare_cs"]


def _cs(color, shape):
    return dsl.FilterColor(dsl.FilterShape(dsl.Scene(), shape), color)


def _s(shape):
    return dsl.FilterShape(dsl.Scene(), shape)


def build_program(template, params):
    if template == "count_cs":
        return dsl.Count(_cs(params["color"], params["shape"]))
    if template == "exists_cs":
        return dsl.Exists(_cs(params["color"], params["shape"]))
    if template == "compare_shape":
        return dsl.GreaterThan(dsl.Count(_s(params["shape_a"])), dsl.Count(_s(params["shape_b"])))
    if template == "compare_cs":  # colour+shape on one side only: the held-out combination
        return dsl.GreaterThan(
            dsl.Count(_cs(params["color"], params["shape_a"])), dsl.Count(_s(params["shape_b"]))
        )
    raise ValueError(template)


def _phrase(p, plural=True):
    """Noun phrase for an ObjSet program, e.g. 'blue squares'."""
    color = shape = None
    while not isinstance(p, dsl.Scene):
        if isinstance(p, dsl.FilterColor):
            color = p.color
        elif isinstance(p, dsl.FilterShape):
            shape = p.shape
        p = p.child
    noun = (shape + ("s" if plural else "")) if shape else ("objects" if plural else "object")
    return f"{color} {noun}" if color else noun


def question_text(p):
    """Natural-language question generated from a program."""
    if isinstance(p, dsl.Count):
        return f"How many {_phrase(p.child)} are there?"
    if isinstance(p, dsl.Exists):
        return f"Is there a {_phrase(p.child, plural=False)}?"
    if isinstance(p, dsl.GreaterThan):
        return f"Are there more {_phrase(p.left.child)} than {_phrase(p.right.child)}?"
    raise ValueError(p)


def _candidates(template, cfg):
    colors, shapes = cfg["colors"], cfg["shapes"]
    if template in ("count_cs", "exists_cs"):
        return [{"color": c, "shape": s} for c in colors for s in shapes]
    if template == "compare_shape":
        return [{"shape_a": a, "shape_b": b} for a in shapes for b in shapes if a != b]
    if template == "compare_cs":
        return [
            {"color": c, "shape_a": a, "shape_b": b}
            for c in colors for a in shapes for b in shapes if a != b
        ]
    raise ValueError(template)


def sample_params(template, scene, cfg, rng):
    """Choose the desired answer class first, then a parameter that gives it.

    count_cs: answer class uniform over the feasible values in 0..count_max_target.
    exists_cs / compare_*: "yes" with probability p_yes (falls back to the other class
    if the scene cannot give the wanted one). Answer balance is on purpose, not luck.
    """
    by_answer = {}
    for params in _candidates(template, cfg):
        a = answer_to_str(execute(build_program(template, params), scene))
        by_answer.setdefault(a, []).append(params)
    if template == "count_cs":
        feasible = sorted(a for a in by_answer if int(a) <= cfg["count_max_target"])
        # answer 0 is always feasible and would dominate; its weight is a config knob
        weights = [cfg.get("count_zero_weight", 1.0) if a == "0" else 1.0 for a in feasible]
        wanted = rng.choices(feasible, weights)[0]
    else:
        wanted = "yes" if rng.random() < cfg["p_yes"] else "no"
        if wanted not in by_answer:
            wanted = "no" if wanted == "yes" else "yes"
    return rng.choice(by_answer[wanted])
