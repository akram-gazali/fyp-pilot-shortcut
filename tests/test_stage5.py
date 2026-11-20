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


