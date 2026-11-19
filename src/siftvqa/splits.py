"""Operation-combination signatures and scene-group splits."""
from . import dsl
from .seeding import make_rng
from .templates import build_program


def signature(p):
    """Operation signature of a program with attribute values erased, e.g.
    gt(count(fc(fs(S))),count(fs(S))). Two questions share a signature iff they use the
    same combination of operations (whatever colours/shapes they mention)."""
    if isinstance(p, dict):
        p = dsl.from_dict(p)
    if isinstance(p, dsl.Scene):
        return "S"
    if isinstance(p, dsl.FilterColor):
        return f"fc({signature(p.child)})"
    if isinstance(p, dsl.FilterShape):
        return f"fs({signature(p.child)})"
    if isinstance(p, dsl.Count):
        return f"count({signature(p.child)})"
    if isinstance(p, dsl.Exists):
        return f"exists({signature(p.child)})"
    return f"gt({signature(p.left)},{signature(p.right)})"


_EXAMPLE_PARAMS = {"color": "red", "shape": "square", "shape_a": "square", "shape_b": "circle"}


def template_signature(template):
    return signature(build_program(template, _EXAMPLE_PARAMS))


def assign_scenes(scene_ids, cfg):
    """Disjoint scene-group split: every scene (with all its questions) goes to exactly one split."""
    ids = sorted(scene_ids)
    make_rng(cfg["seed"], "split").shuffle(ids)
    n_tr, n_po, n_te = cfg["n_train_scenes"], cfg["n_pool_scenes"], cfg["n_test_scenes"]
    assert n_tr + n_po + n_te <= len(ids), "not enough scenes"
    out = {}
    for i, sid in enumerate(ids[: n_tr + n_po + n_te]):
        out[sid] = "train" if i < n_tr else ("pool" if i < n_tr + n_po else "test")
    return out
