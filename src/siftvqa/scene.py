"""Scene sampling and rendering (Pillow). The scene JSON is the ground truth.

Every object is drawn inside a bounding circle of radius 1.42*size, and bounding
circles never overlap (plus a margin), so objects never touch. Drawing uses flat
colours with no anti-aliasing, so pixel colours are exact (used by the label audit).
"""
from PIL import Image, ImageDraw

BACKGROUND = (240, 240, 240)
COLOR_RGB = {
    "red": (220, 30, 30),
    "blue": (30, 60, 220),
    "green": (30, 170, 50),
    "yellow": (240, 220, 30),
    "purple": (150, 40, 170),
    "orange": (250, 140, 20),
}
BOUND = 1.42  # bounding-circle radius / size


def sample_scene(cfg, rng, scene_id):
    n = rng.randint(cfg["min_objects"], cfg["max_objects"])
    W = cfg["image_size"]
    lo, hi = cfg["size_range"]
    pal = cfg["palette"]
    colors = rng.sample(cfg["colors"], rng.randint(*pal["n_colors"]))
    shapes = rng.sample(cfg["shapes"], rng.randint(*pal["n_shapes"]))
    focus = (rng.choice(colors), rng.choice(shapes))
    objs = []
    for _ in range(n):
        if rng.random() < pal["p_focus"]:
            color, shape = focus
        else:
            color, shape = rng.choice(colors), rng.choice(shapes)
        for _attempt in range(1000):
            size = rng.randint(lo, hi)
            x = rng.randint(int(BOUND * size) + 1, W - int(BOUND * size) - 2)
            y = rng.randint(int(BOUND * size) + 1, W - int(BOUND * size) - 2)
            ok = all(
                (x - o["x"]) ** 2 + (y - o["y"]) ** 2
                >= (BOUND * (size + o["size"]) + cfg["placement_margin"]) ** 2
                for o in objs
            )
            if ok:
                break
        else:
            raise RuntimeError("could not place object")
        objs.append(
            {
                "id": len(objs),
                "shape": shape,
                "color": color,
                "x": x,
                "y": y,
                "size": size,
            }
        )
    return {"scene_id": scene_id, "image_size": W, "objects": objs}


def render(scene):
    W = scene["image_size"]
    img = Image.new("RGB", (W, W), BACKGROUND)
    d = ImageDraw.Draw(img)
    for o in scene["objects"]:
        x, y, s = o["x"], o["y"], o["size"]
        fill = COLOR_RGB[o["color"]]
        if o["shape"] == "square":
            d.rectangle([x - s, y - s, x + s, y + s], fill=fill)
        elif o["shape"] == "circle":
            d.ellipse([x - s, y - s, x + s, y + s], fill=fill)
        else:  # triangle
            d.polygon([(x, y - s), (x - s, y + s), (x + s, y + s)], fill=fill)
    return img
