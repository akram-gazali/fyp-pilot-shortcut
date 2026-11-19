"""Question templates. Each template builds a *program*; the English question is
generated from the program (never the other way round)."""
from . import dsl

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


def sample_params(template, scene, cfg, rng):
    """Pick template parameters. With prob p_target_present the asked colour+shape
    occurs in the scene (for exists: 'yes'-type); otherwise it is chosen to be absent
    when possible. Answer-value balance is handled later by quotas, not here."""
    colors, shapes = cfg["colors"], cfg["shapes"]
    present = sorted({(o["color"], o["shape"]) for o in scene["objects"]})
    absent = [(c, s) for c in colors for s in shapes if (c, s) not in present]

    def pick_cs():
        if rng.random() < cfg["p_target_present"] or not absent:
            return rng.choice(present)
        return rng.choice(absent)

    if template in ("count_cs", "exists_cs"):
        c, s = pick_cs()
        return {"color": c, "shape": s}
    if template == "compare_shape":
        a, b = rng.sample(shapes, 2)
        return {"shape_a": a, "shape_b": b}
    if template == "compare_cs":
        c, a = pick_cs()
        b = rng.choice([s for s in shapes if s != a])
        return {"color": c, "shape_a": a, "shape_b": b}
    raise ValueError(template)
