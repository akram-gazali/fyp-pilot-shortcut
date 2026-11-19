from siftvqa import dsl
from siftvqa.config import ROOT
from siftvqa.executor import answer_to_str, execute
from siftvqa.shortcuts import apply_rule, load_shortcuts, score_question
from siftvqa.templates import build_program

S = dsl.Scene()
SCENE = {"objects": [
    {"id": 0, "shape": "square", "color": "blue", "x": 0, "y": 0, "size": 5},
    {"id": 1, "shape": "square", "color": "red", "x": 0, "y": 0, "size": 5},
    {"id": 2, "shape": "circle", "color": "blue", "x": 0, "y": 0, "size": 5},
]}


def q(template, params):
    p = build_program(template, params)
    return {"qid": "t", "template": template, "program": dsl.to_dict(p), "answer": answer_to_str(execute(p, SCENE))}


def test_shortcut_rules():
    p = build_program("count_cs", {"color": "blue", "shape": "square"})
    assert dsl.to_str(apply_rule(p, "drop_color")) == "count(filter_shape[square](scene))"
    assert dsl.to_str(apply_rule(p, "drop_shape")) == "count(filter_color[blue](scene))"


def test_scoring_count_and_exists():
    sc = load_shortcuts(ROOT / "configs" / "shortcuts.yaml")
    s = score_question(q("count_cs", {"color": "blue", "shape": "square"}), SCENE, sc)
    assert s["intended"] == "1"
    assert s["shortcuts"]["count_ignore_color"]["answer"] == "2"
    assert s["shortcuts"]["count_ignore_shape"]["answer"] == "2"
    assert s["d"] == 1
    s = score_question(q("exists_cs", {"color": "red", "shape": "circle"}), SCENE, sc)
    assert s["intended"] == "no" and s["shortcuts"]["exists_ignore_color"]["answer"] == "yes" and s["d"] == 1
    s = score_question(q("exists_cs", {"color": "blue", "shape": "circle"}), SCENE, sc)
    assert s["d"] == 0


def test_no_shortcut_for_comparisons():
    sc = load_shortcuts(ROOT / "configs" / "shortcuts.yaml")
    s = score_question(q("compare_shape", {"shape_a": "square", "shape_b": "circle"}), SCENE, sc)
    assert s["shortcuts"] == {} and s["d"] == 0


def test_only_three_frozen_shortcuts():
    assert len(load_shortcuts(ROOT / "configs" / "shortcuts.yaml")) == 3
