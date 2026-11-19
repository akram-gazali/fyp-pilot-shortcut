import math

from siftvqa.cost import arm_hours, extrapolate
from siftvqa.power import max_sigma_seed, min_detectable_delta, n_paired, n_with_seed_noise, power_at
from siftvqa.vlm_eval import joint_report, normalize_answer, plain_accuracy_report


def test_power_roundtrip():
    n = n_paired(0.03, 0.2)
    assert power_at(n, 0.03, 0.2, 0.0001, 1000) >= 0.79      # ~no seed noise -> ~0.80
    n2 = n_with_seed_noise(0.03, 0.2, 0.01, 5)
    assert n2 > n and power_at(n2, 0.03, 0.2, 0.01, 5) >= 0.799
    assert power_at(n2 - 50, 0.03, 0.2, 0.01, 5) < 0.8


def test_seed_noise_ceiling():
    s = max_sigma_seed(0.03, 3)
    assert n_with_seed_noise(0.03, 0.2, s * 1.01, 3) is None
    assert n_with_seed_noise(0.03, 0.2, s * 0.9, 3) is not None


def test_min_detectable_delta_shrinks_with_n():
    a = min_detectable_delta(500, 0.2, 0.01, 5)
    b = min_detectable_delta(5000, 0.2, 0.01, 5)
    assert b < a


def test_normalize_answer():
    assert normalize_answer("Yes.") == "yes"
    assert normalize_answer("No, there are none") == "no"
    assert normalize_answer("3") == "3"
    assert normalize_answer("There are 4 blue squares") == "4"
    assert normalize_answer("two") == "2"
    assert normalize_answer("I cannot tell") == "i cannot tell"


def test_plain_and_joint_are_separate_endpoints():
    test = [{"qid": "a", "template": "compare_cs", "answer": "yes"}, {"qid": "b", "template": "count_cs", "answer": "2"}]
    rep = plain_accuracy_report(test, {"a": "yes", "b": "3"}, "compare_cs")
    assert rep["held_out_pct"] == 100.0 and rep["seen_pct"] == 0.0 and rep["all_pct"] == 50.0
    rec = {"qid": "a", "template": "compare_cs", "relevant_scene": "a__rel", "irrelevant_scene": "a__irr",
           "answers": {"original": "yes", "relevant": "no", "irrelevant": "yes"}}
    ok = {"a": "yes", "a__rel": "no", "a__irr": "yes"}
    assert joint_report([rec], ok, "compare_cs")["held_out"]["joint_pct"] == 100.0
    bad = {**ok, "a__irr": "no"}
    r = joint_report([rec], bad, "compare_cs")["held_out"]
    assert r["joint_pct"] == 0.0 and r["original_pct"] == 100.0 and r["irrelevant_edit_pct"] == 0.0
    assert joint_report([rec], {"a": "yes"}, "compare_cs")["held_out"]["joint_pct"] == 0.0   # missing = wrong


def test_cost_arithmetic():
    a = arm_hours(sec_per_step=2.0, batch_size=4, n_train=400, epochs=2, sec_per_eval_example=0.1, n_eval_examples=1000, load_overhead_sec=0)
    assert a["steps"] == 200 and math.isclose(a["train_hours"], 400 / 3600) and math.isclose(a["eval_hours"], 100 / 3600)
    out = extrapolate(2.0, 4, 0.1, 1000, [{"name": "x", "n_train": 400, "epochs": 2, "runs": 10}], budget_hours=20, load_overhead_sec=0)
    assert out[0]["fits_budget"] and math.isclose(out[0]["total_gpu_hours"], 10 * 500 / 3600, abs_tol=0.01)
