"""Paired visual edits for the test set.

For a test question on scene S we make two single-object edits of S:
  relevant   - the executor answer CHANGES (the edit touches what the question asks about)
  irrelevant - the executor answer STAYS the same
Labels come from the executor on the edited scene JSON (never typed in). Edits keep the
scene valid: objects never overlap, stay inside the canvas, object count stays in range.
Joint correctness = 1 only if original, relevant and irrelevant are all answered correctly.
"""
import copy

from . import dsl
from .executor import answer_to_str, execute
from .scene import BOUND, place

REL_KINDS = ("recolor", "reshape", "remove", "add")
IRR_KINDS = ("recolor", "reshape", "remove", "add", "move", "resize")


def _reindex(objs):
    for i, o in enumerate(objs):
        o["id"] = i
    return objs


def _with(scene, objs, edit, suffix):
    new = copy.deepcopy(scene)
    new["objects"] = _reindex(objs)
    new["scene_id"] = f"{scene['scene_id']}__{suffix}"
    new["edit"] = edit
    return new


def candidate_edits(scene, cfg, rng):
    """All single-object edits (a bounded sample for add/move/resize): list of (kind, new_scene)."""
    objs = scene["objects"]
    n = len(objs)
    out = []

    def edited(i, **ch):
        o2 = copy.deepcopy(objs)
        o2[i].update(ch)
        return o2

    for o in objs:
        i = o["id"]
        for c in cfg["colors"]:
            if c != o["color"]:
                out.append(("recolor", _with(scene, edited(i, color=c), {"kind": "recolor", "object": i, "to": c}, "e")))
        for s in cfg["shapes"]:
            if s != o["shape"]:
                out.append(("reshape", _with(scene, edited(i, shape=s), {"kind": "reshape", "object": i, "to": s}, "e")))
        if n > cfg["min_objects"]:
            o2 = [copy.deepcopy(x) for x in objs if x["id"] != i]
            out.append(("remove", _with(scene, o2, {"kind": "remove", "object": i}, "e")))
        # move: same attributes, new free position at least 20 px away
        pos = place(objs, o["size"], cfg, rng, skip_id=i)
        if pos and (pos[0] - o["x"]) ** 2 + (pos[1] - o["y"]) ** 2 >= 20 ** 2:
            out.append(("move", _with(scene, edited(i, x=pos[0], y=pos[1]),
                                      {"kind": "move", "object": i, "to": list(pos)}, "e")))
        # resize: new size (>=3 px different) that still fits without overlap
        lo, hi = cfg["size_range"]
        sizes = [s for s in range(lo, hi + 1) if abs(s - o["size"]) >= 3]
        if sizes:
            s = rng.choice(sizes)
            m = int(BOUND * s)
            W = cfg["image_size"]
            fits = (m + 1 <= o["x"] <= W - m - 2 and m + 1 <= o["y"] <= W - m - 2 and all(
                x["id"] == i or (o["x"] - x["x"]) ** 2 + (o["y"] - x["y"]) ** 2
                >= (BOUND * (s + x["size"]) + cfg["placement_margin"]) ** 2 for x in objs))
            if fits:
                out.append(("resize", _with(scene, edited(i, size=s), {"kind": "resize", "object": i, "to": s}, "e")))
    if n < cfg["max_objects"]:
        for _ in range(6):
            size = rng.randint(*cfg["size_range"])
            pos = place(objs, size, cfg, rng)
            if pos:
                new = {"id": n, "shape": rng.choice(cfg["shapes"]), "color": rng.choice(cfg["colors"]),
                       "x": pos[0], "y": pos[1], "size": size}
                out.append(("add", _with(scene, copy.deepcopy(objs) + [new],
                                         {"kind": "add", "shape": new["shape"], "color": new["color"], "at": list(pos)}, "e")))
    return out


def _multi_relevant(scene, program, orig, cfg, rng, max_steps=4, trials=40):
    """When no single edit changes the answer (e.g. several matching objects), chain up to
    max_steps answer-relevant-kind edits; keep the shortest chain that flips the answer."""
    best = None
    for _ in range(trials):
        cur, steps = scene, []
        for _step in range(max_steps):
            cands = [c for c in candidate_edits(cur, cfg, rng) if c[0] in REL_KINDS]
            if not cands:
                break
            _kind, cur = rng.choice(cands)
            steps.append(cur["edit"])
            if execute(program, cur) != orig:
                if best is None or len(steps) < len(best[1]):
                    out = copy.deepcopy(cur)
                    out["edit"] = {"kind": "multi", "steps": steps}
                    best = (out, steps)
                break
    return best[0] if best else None


def make_paired_edits(scene, program, cfg, rng):
    """Return {"relevant": scene', "irrelevant": scene'', "answers": {...}} or None if no valid pair."""
    orig = execute(program, scene)
    rel, irr = {}, {}
    for kind, sc in candidate_edits(scene, cfg, rng):
        a = execute(program, sc)
        (irr if a == orig else rel).setdefault(kind, []).append((sc, a))
    rel = {k: v for k, v in rel.items() if k in REL_KINDS}
    irr = {k: v for k, v in irr.items() if k in IRR_KINDS}
    if not irr:
        return None
    if rel:
        rk = rng.choice(sorted(rel))
        r_sc, r_a = rng.choice(rel[rk])
    else:
        r_sc = _multi_relevant(scene, program, orig, cfg, rng)
        if r_sc is None:
            return None
        r_a = execute(program, r_sc)
    ik = rng.choice(sorted(irr))
    i_sc, i_a = rng.choice(irr[ik])
    r_sc["scene_id"] = scene["scene_id"] + "__rel"
    i_sc["scene_id"] = scene["scene_id"] + "__irr"
    return {"relevant": r_sc, "irrelevant": i_sc,
            "answers": {"original": answer_to_str(orig), "relevant": answer_to_str(r_a), "irrelevant": answer_to_str(i_a)}}


def joint_correct(pred_original, pred_relevant, pred_irrelevant, answers):
    """1 only if all three predictions match their executor answers, else 0."""
    return int(pred_original == answers["original"] and pred_relevant == answers["relevant"]
               and pred_irrelevant == answers["irrelevant"])
