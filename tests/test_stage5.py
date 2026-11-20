import json
from pathlib import Path

import pytest

from siftvqa import formats

ROOT = Path(__file__).resolve().parents[1]




# ---------------------------------------------------------------- formats (pure python)
def test_prompt_per_template():
    assert formats.prompt_text("Q?", "count_cs").endswith("Answer with only a number.")
    for t in ("exists_cs", "compare_shape", "compare_cs"):
        assert formats.prompt_text("Q?", t).endswith("Answer with only yes or no.")
    assert formats.prompt_text("Q?", "exists_cs", "legacy").endswith("single word or number.")
    assert formats.candidates("count_cs") == [str(i) for i in range(9)]
    assert formats.candidates("compare_cs") == ["yes", "no"]


def test_strict_format():
    assert formats.strict_format_ok("Yes.", "exists_cs") and formats.strict_format_ok(" 3 ", "count_cs")
    assert not formats.strict_format_ok("Yes, 1", "exists_cs")      # the Stage 4 failure
    assert not formats.strict_format_ok("0", "exists_cs")            # a number for a yes/no question
    assert not formats.strict_format_ok("yes", "count_cs")


def test_pick_method_tie_rule():
    r = {"suffix": {"macro_acc": 80.0, "sec_per_item": 0.50}, "choice": {"macro_acc": 79.5, "sec_per_item": 0.20}}
    assert formats.pick_method(r, 1.0)["chosen"] == "choice"            # within 1 point: tie -> faster
    r["choice"]["macro_acc"] = 78.9
    assert formats.pick_method(r, 1.0)["chosen"] == "suffix"            # more than 1 point worse: accuracy wins
    r["choice"]["macro_acc"] = 81.0
    assert formats.pick_method(r, 1.0)["chosen"] == "choice"


def test_pick_resolution_two_point_rule():
    assert formats.pick_resolution({224: 80.0, 336: 82.0})["chosen"] == 224       # exactly 2 points: not MORE than 2
    assert formats.pick_resolution({224: 80.0, 336: 82.1})["chosen"] == 336


def test_best_candidate_logsumexp():
    cand = {"yes": [1, 2], "no": [3, 4]}
    best, sc = formats.best_candidate({1: -3.0, 2: -0.1, 3: -1.0, 4: -5.0}, cand)
    assert best == "yes" and sc["yes"] > sc["no"]


def test_pins_and_dtype():
    pins = formats.parse_pins("transformers==5.22.1  # x\n# torch==2\npeft==0.21.2\n")
    assert pins == {"transformers": "5.22.1", "peft": "0.21.2"}
    assert formats.version_mismatches(pins, {"transformers": "5.22.1", "peft": "0.20.0"}) == [("peft", "0.21.2", "0.20.0")]
    assert formats.version_mismatches({"torch": "2.14.1"}, {"torch": "9.9"}) == []     # torch never enforced
    assert formats.choose_dtype("auto", 7) == "fp16" and formats.choose_dtype("auto", 8) == "bf16"
    assert formats.choose_dtype("bf16", 7) == "bf16"


def test_atomic_write(tmp_path):
    formats.write_json_atomic(tmp_path / "a" / "x.json", {"a": 1})
    assert json.load(open(tmp_path / "a" / "x.json")) == {"a": 1} and not list((tmp_path / "a").glob("*.tmp"))


def test_requirements_gpu_has_pins():
    assert {"transformers", "peft"} <= set(formats.parse_pins((ROOT / "requirements-gpu.txt").read_text()))




def _script():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("s5script", Path(__file__).resolve().parents[1] / "colab" / "stage4_feasibility.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_joint_score_computable_for_a_trained_arm_eval():
    """The per-arm eval items include both edited images; a predictor yields the joint score (the primary metric)."""
    m = _script()
    rows = [{"qid": "q1", "scene_id": "s1", "template": "compare_cs", "question": "Q?", "answer": "yes"},
            {"qid": "q2", "scene_id": "s2", "template": "exists_cs", "question": "Q2?", "answer": "no"}]
    edits = [{"qid": "q1", "scene_id": "s1", "template": "compare_cs", "relevant_scene": "q1__rel", "irrelevant_scene": "q1__irr",
              "answers": {"original": "yes", "relevant": "no", "irrelevant": "yes"}},
             {"qid": "q2", "scene_id": "s2", "template": "exists_cs", "relevant_scene": "q2__rel", "irrelevant_scene": "q2__irr",
              "answers": {"original": "no", "relevant": "yes", "irrelevant": "no"}}]
    items = [m.eval_item(r["qid"], "p", r, r["answer"]) for r in rows] + m.edit_items(edits, {r["qid"]: r for r in rows})
    assert [i["key"] for i in items] == ["q1", "q2", "q1__rel", "q1__irr", "q2__rel", "q2__irr"]
    perfect = {i["key"]: i["truth"] for i in items}
    j = m.joint_report(edits, perfect, "compare_cs")
    assert j["held_out"]["joint_pct"] == 100.0 and j["seen"]["joint_pct"] == 100.0
    # a model that ignores the edit (answers the original answer on the relevant image) fails the joint score only
    blind = dict(perfect, q1__rel="yes")
    jb = m.joint_report(edits, blind, "compare_cs")
    assert jb["held_out"]["joint_pct"] == 0.0 and jb["held_out"]["original_pct"] == 100.0 and jb["held_out"]["relevant_edit_pct"] == 0.0
    assert jb["seen"]["joint_pct"] == 100.0
