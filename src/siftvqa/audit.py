"""Independent label audit.

Recomputes every answer WITHOUT the DSL, executor or scene JSON objects:
  1. objects are re-detected from the rendered pixels (flat-colour connected
     components; shape from the fill ratio of the component's bounding box:
     square = 1.0, circle ~ 0.785, triangle ~ 0.5);
  2. the answer is computed from the question's template + params with plain
     loops (a second implementation of the template semantics).
Any disagreement with the stored answer (or with the scene JSON) is reported.
"""
from collections import deque

import numpy as np

from .scene import BACKGROUND, COLOR_RGB

_RGB_TO_COLOR = {v: k for k, v in COLOR_RGB.items()}


def objects_from_pixels(img):
    a = np.asarray(img.convert("RGB"))
    H, W, _ = a.shape
    seen = np.zeros((H, W), bool)
    found = []
    for rgb, color in _RGB_TO_COLOR.items():
        mask = np.all(a == np.array(rgb), axis=2)
        for y, x in zip(*np.nonzero(mask)):
            if seen[y, x]:
                continue
            q, comp = deque([(y, x)]), []
            seen[y, x] = True
            while q:
                cy, cx = q.popleft()
                comp.append((cy, cx))
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            ys, xs = zip(*comp)
            box = (max(ys) - min(ys) + 1) * (max(xs) - min(xs) + 1)
            ratio = len(comp) / box
            shape = "square" if ratio > 0.9 else ("circle" if ratio > 0.65 else "triangle")
            found.append({"color": color, "shape": shape})
    # any non-background pixel with an unknown colour is an error
    known = np.zeros((H, W), bool)
    for rgb in _RGB_TO_COLOR:
        known |= np.all(a == np.array(rgb), axis=2)
    stray = int(np.sum(~known & ~np.all(a == np.array(BACKGROUND), axis=2)))
    return found, stray


def _n(objs, color=None, shape=None):
    return sum(1 for o in objs if (color is None or o["color"] == color) and (shape is None or o["shape"] == shape))


def answer_from_params(template, params, objs):
    """Second implementation of template semantics (plain loops, no DSL)."""
    if template == "count_cs":
        return str(_n(objs, params["color"], params["shape"]))
    if template == "exists_cs":
        return "yes" if _n(objs, params["color"], params["shape"]) > 0 else "no"
    if template == "compare_shape":
        return "yes" if _n(objs, shape=params["shape_a"]) > _n(objs, shape=params["shape_b"]) else "no"
    if template == "compare_cs":
        left = _n(objs, params["color"], params["shape_a"])
        return "yes" if left > _n(objs, shape=params["shape_b"]) else "no"
    raise ValueError(template)


def audit_scene(img, scene, questions):
    """Return a list of mismatch descriptions (empty = all labels verified)."""
    problems = []
    objs, stray = objects_from_pixels(img)
    if stray:
        problems.append(f"{stray} stray pixels of unknown colour")
    key = lambda o: (o["color"], o["shape"])
    if sorted(map(key, objs)) != sorted(map(key, scene["objects"])):
        problems.append("pixel-detected objects differ from scene JSON")
    for q in questions:
        got = answer_from_params(q["template"], q["params"], objs)
        if got != q["answer"]:
            problems.append(f"{q['qid']}: stored {q['answer']!r} vs audit {got!r}")
    return problems
