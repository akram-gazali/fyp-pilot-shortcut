import json
import sys

import pytest

from siftvqa import dsl
from siftvqa.audit import audit_scene, objects_from_pixels
from siftvqa.config import ROOT, load_config
from siftvqa.executor import answer_to_str, execute
from siftvqa.scene import render, sample_scene
from siftvqa.seeding import make_rng
from siftvqa.templates import build_program, question_text

sys.path.insert(0, str(ROOT / "scripts"))
from gen_pool import make_questions  # noqa: E402


def obj(i, shape, color, x, y, size=12):
    return {"id": i, "shape": shape, "color": color, "x": x, "y": y, "size": size}


# Hand-made scene: 2 blue squares, 1 red square, 1 blue circle, 1 red triangle.
SCENE = {
    "scene_id": "hand",
    "image_size": 224,
    "objects": [
        obj(0, "square", "blue", 30, 30),
        obj(1, "square", "blue", 90, 30),
        obj(2, "square", "red", 150, 30),
        obj(3, "circle", "blue", 30, 100),
        obj(4, "triangle", "red", 100, 100),
    ],
}
S = dsl.Scene()


def cs(color, shape):
    return dsl.FilterColor(dsl.FilterShape(S, shape), color)


def test_count():
    assert execute(dsl.Count(cs("blue", "square")), SCENE) == 2
    assert execute(dsl.Count(dsl.FilterShape(S, "square")), SCENE) == 3
    assert execute(dsl.Count(cs("green", "square")), SCENE) == 0
    assert execute(dsl.Count(S), SCENE) == 5


def test_exists():
    assert execute(dsl.Exists(cs("red", "triangle")), SCENE) is True
    assert execute(dsl.Exists(cs("red", "circle")), SCENE) is False


def test_greater_than_is_strict():
    sq, tr = dsl.Count(dsl.FilterShape(S, "square")), dsl.Count(dsl.FilterShape(S, "triangle"))
    assert execute(dsl.GreaterThan(sq, tr), SCENE) is True
    assert execute(dsl.GreaterThan(tr, sq), SCENE) is False
    one = dsl.Count(cs("red", "square"))
    assert execute(dsl.GreaterThan(one, tr), SCENE) is False  # 1 > 1 is False


def test_filter_order_does_not_matter():
    a = dsl.Count(dsl.FilterColor(dsl.FilterShape(S, "square"), "blue"))
    b = dsl.Count(dsl.FilterShape(dsl.FilterColor(S, "blue"), "square"))
    assert execute(a, SCENE) == execute(b, SCENE) == 2


def test_type_errors():
    with pytest.raises(dsl.ProgramTypeError):
        execute(dsl.Count(dsl.Count(S)), SCENE)
    with pytest.raises(dsl.ProgramTypeError):
        execute(dsl.GreaterThan(S, S), SCENE)
    with pytest.raises(dsl.ProgramTypeError):
        execute(dsl.Exists(dsl.Exists(S)), SCENE)


def test_answer_strings():
    assert answer_to_str(True) == "yes" and answer_to_str(False) == "no"
    assert answer_to_str(0) == "0" and answer_to_str(7) == "7"


def test_program_roundtrip_and_length():
    p = build_program("compare_cs", {"color": "blue", "shape_a": "square", "shape_b": "triangle"})
    assert dsl.from_dict(json.loads(json.dumps(dsl.to_dict(p)))) == p
    assert dsl.length(dsl.Count(cs("blue", "square"))) == 4


def test_question_text_from_program():
    assert question_text(build_program("count_cs", {"color": "blue", "shape": "square"})) == "How many blue squares are there?"
    assert question_text(build_program("exists_cs", {"color": "red", "shape": "circle"})) == "Is there a red circle?"
    assert question_text(build_program("compare_shape", {"shape_a": "square", "shape_b": "triangle"})) == "Are there more squares than triangles?"
    assert (
        question_text(build_program("compare_cs", {"color": "blue", "shape_a": "square", "shape_b": "triangle"}))
        == "Are there more blue squares than triangles?"
    )


def test_pixel_detection_on_hand_scene():
    objs, stray = objects_from_pixels(render(SCENE))
    assert stray == 0
    key = lambda o: (o["color"], o["shape"])
    assert sorted(map(key, objs)) == sorted(map(key, SCENE["objects"]))


CFG = load_config(ROOT / "configs" / "generator.yaml")


def _gen(n=40):
    for i in range(n):
        scene = sample_scene(CFG, make_rng(CFG["seed"], "scene", i), f"s{i}")
        yield scene, make_questions(scene, CFG, make_rng(CFG["seed"], "questions", i))


def test_scene_constraints():
    for scene, _ in _gen():
        assert CFG["min_objects"] <= len(scene["objects"]) <= CFG["max_objects"]
        for a in scene["objects"]:
            for b in scene["objects"]:
                if a["id"] < b["id"]:
                    d2 = (a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2
                    assert d2 > (1.42 * (a["size"] + b["size"])) ** 2


def test_generation_is_deterministic():
    a = [q["answer"] for _, qs in _gen(10) for q in qs]
    b = [q["answer"] for _, qs in _gen(10) for q in qs]
    assert a == b


def test_label_audit_passes_on_generated_scenes():
    for scene, qs in _gen():
        assert audit_scene(render(scene), scene, qs) == []


def test_label_audit_catches_a_wrong_label():
    scene, qs = next(_gen(1))
    q = dict(qs[0])
    q["answer"] = "yes" if q["answer"] != "yes" else "no"
    assert audit_scene(render(scene), scene, [q]) != []


def test_label_audit_catches_scene_json_mismatch():
    scene, qs = next(_gen(1))
    bad = json.loads(json.dumps(scene))
    bad["objects"][0]["color"] = "green" if bad["objects"][0]["color"] != "green" else "red"
    assert audit_scene(render(scene), bad, []) != []
