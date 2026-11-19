import sys
from collections import Counter

import pytest

from siftvqa import dsl
from siftvqa.config import ROOT, load_config
from siftvqa.edits import joint_correct, make_paired_edits
from siftvqa.executor import answer_to_str, execute
from siftvqa.leakage import duplicate_scenes, question_only_baseline, scene_group_overlap, signature_overlap
from siftvqa.scene import BOUND, render, sample_scene
from siftvqa.seeding import make_rng
from siftvqa.selectors import ARMS, bucket_of, make_plan, run_all_arms
from siftvqa.shortcuts import load_shortcuts, score_question
from siftvqa.splits import assign_scenes, signature, template_signature
from siftvqa.templates import TEMPLATES, build_program

sys.path.insert(0, str(ROOT / "scripts"))
from gen_pool import make_questions  # noqa: E402
from siftvqa.audit import audit_scene  # noqa: E402

GCFG = load_config(ROOT / "configs" / "generator.yaml")
SEL = load_config(ROOT / "configs" / "selection.yaml")
SC = load_shortcuts(ROOT / "configs" / "shortcuts.yaml")


def test_signatures_distinguish_the_held_out_combination():
    sigs = {t: template_signature(t) for t in TEMPLATES}
    assert len(set(sigs.values())) == 4
    p = build_program("compare_cs", {"color": "red", "shape_a": "circle", "shape_b": "square"})
    assert signature(p) == sigs["compare_cs"] == "gt(count(fc(fs(S))),count(fs(S)))"
    # attribute values do not change the signature
    q = build_program("compare_cs", {"color": "blue", "shape_a": "square", "shape_b": "triangle"})
    assert signature(q) == sigs["compare_cs"]


def test_split_is_scene_disjoint_and_sized():
    cfg = {"seed": 0, "n_train_scenes": 10, "n_pool_scenes": 30, "n_test_scenes": 10}
    ids = [f"s{i}" for i in range(60)]
    a = assign_scenes(ids, cfg)
    assert Counter(a.values()) == {"train": 10, "pool": 30, "test": 10}
    assert a == assign_scenes(ids, cfg)  # deterministic


@pytest.fixture(scope="module")
def pool():
    rows = []
    for i in range(500):
        scene = sample_scene(GCFG, make_rng(7, "scene", i), f"s{i}")
        for q in make_questions(scene, GCFG, make_rng(7, "q", i)):
            if q["template"] == "compare_cs":
                continue
            rows.append({**q, **score_question(q, scene, SC), "n_objects": len(scene["objects"]),
                         "n_distinct_colors": len({o["color"] for o in scene["objects"]}),
                         "n_distinct_shapes": len({o["shape"] for o in scene["objects"]}),
                         "signature": signature(q["program"])})
    return rows


def small_cfg(mode):
    return {**SEL, "budget": 100, "exists_mode": mode}


@pytest.mark.parametrize("mode", ["stratified_quota", "fixed_shared_base"])
def test_selectors_equal_size_matched_labels_and_shared_base(pool, mode):
    cfg = small_cfg(mode)
    by_id = {r["qid"]: r for r in pool}
    arms, quota, plan = run_all_arms(pool, cfg, seed=0)
    assert set(arms) == set(ARMS)
    sel_t = ("count_cs",) if mode == "fixed_shared_base" else ("count_cs", "exists_cs")
    lab = lambda ids: Counter((by_id[i]["template"], by_id[i]["answer"]) for i in ids if by_id[i]["template"] in sel_t)
    base = lab(arms["shortcut_aware"][0])
    for name, (ids, _) in arms.items():
        assert len(ids) == 100 and len(set(ids)) == 100
        assert Counter(by_id[i]["template"] for i in ids) == Counter(by_id[i]["template"] for i in arms["shortcut_aware"][0])
        assert lab(ids) == base, name  # same template and answer counts in every arm
    # matched random also matches the object-count bucket mix
    bk = lambda ids: Counter((by_id[i]["template"], by_id[i]["answer"], bucket_of(by_id[i]["n_objects"], SEL["n_objects_buckets"]))
                             for i in ids if by_id[i]["template"] in sel_t)
    assert bk(arms["matched_random"][0]) == bk(arms["shortcut_aware"][0])
    # all count examples in the shortcut arm are d=1
    assert all(by_id[i]["d"] == 1 for i in arms["shortcut_aware"][0] if by_id[i]["template"] == "count_cs")
    # compare_shape (and exists in fixed mode) are the same set in every arm
    shared = [set(i for i in ids if by_id[i]["template"] == "compare_shape") for ids, _ in arms.values()]
    assert all(s == shared[0] for s in shared)
    if mode == "fixed_shared_base":
        ex = [set(i for i in ids if by_id[i]["template"] == "exists_cs") for ids, _ in arms.values()]
        assert all(s == ex[0] for s in ex)
    else:
        ex_no_d1 = [by_id[i] for i in arms["shortcut_aware"][0] if by_id[i]["template"] == "exists_cs" and by_id[i]["answer"] == "no"]
        assert all(r["d"] == 1 for r in ex_no_d1)
        yes = [by_id[i] for i in arms["shortcut_aware"][0] if by_id[i]["template"] == "exists_cs" and by_id[i]["answer"] == "yes"]
        assert len(yes) == plan["exists_yes"]


def test_selectors_are_deterministic(pool):
    a, _, _ = run_all_arms(pool, small_cfg("stratified_quota"), seed=3)
    b, _, _ = run_all_arms(pool, small_cfg("stratified_quota"), seed=3)
    c, _, _ = run_all_arms(pool, small_cfg("stratified_quota"), seed=4)
    assert all(a[k][0] == b[k][0] for k in a)
    assert a["matched_random"][0] != c["matched_random"][0]


def test_plan_sums_to_budget():
    assert sum(v for k, v in make_plan(401, {"count_cs": .4, "exists_cs": .4, "compare_shape": .2}, .5).items() if k != "exists_yes") == 401


def test_paired_edits():
    made = 0
    for i in range(60):
        scene = sample_scene(GCFG, make_rng(11, "scene", i), f"e{i}")
        qs = make_questions(scene, GCFG, make_rng(11, "q", i))
        for q in qs:
            prog = dsl.from_dict(q["program"])
            pair = make_paired_edits(scene, prog, GCFG, make_rng(11, "edit", q["qid"]))
            if pair is None:
                continue
            made += 1
            a = pair["answers"]
            assert a["relevant"] != a["original"] and a["irrelevant"] == a["original"]
            for which in ("relevant", "irrelevant"):
                sc = pair[which]
                assert answer_to_str(execute(prog, sc)) == a[which]       # label comes from the executor
                assert GCFG["min_objects"] <= len(sc["objects"]) <= GCFG["max_objects"]
                for x in sc["objects"]:
                    for y in sc["objects"]:
                        if x["id"] < y["id"]:
                            assert (x["x"] - y["x"]) ** 2 + (x["y"] - y["y"]) ** 2 > (BOUND * (x["size"] + y["size"])) ** 2
                aq = {"qid": "t", "template": q["template"], "params": q["params"], "answer": a[which]}
                assert audit_scene(render(sc), sc, [aq]) == []            # pixels agree too
    assert made > 40


def test_joint_correctness_needs_all_three():
    ans = {"original": "2", "relevant": "3", "irrelevant": "2"}
    assert joint_correct("2", "3", "2", ans) == 1
    assert joint_correct("2", "3", "3", ans) == 0
    assert joint_correct("2", "2", "2", ans) == 0
    assert joint_correct("1", "3", "2", ans) == 0


def test_leakage_checks_detect_planted_leaks():
    sig = "gt(count(fc(fs(S))),count(fs(S)))"
    rows = {"train": [{"signature": sig, "scene_id": "a"}], "pool": [{"signature": "x", "scene_id": "b"}],
            "test": [{"signature": sig, "scene_id": "a"}]}
    assert signature_overlap(rows, sig)["held_out_in_train"] == 1
    assert scene_group_overlap(rows)["train&test"] == 1
    sc = {"objects": [{"shape": "square", "color": "red", "x": 5, "y": 5, "size": 10}]}
    d = duplicate_scenes({"train": {"a": sc}, "test": {"b": dict(sc)}})
    assert d["exact_duplicate_scenes_across_splits"] == 1


def test_question_only_baseline_majority():
    mk = lambda t, a: {"template": t, "answer": a, "params": {"shape": "square"}}
    train = [mk("exists_cs", "yes")] * 3 + [mk("exists_cs", "no")]
    test = [mk("exists_cs", "yes"), mk("exists_cs", "no")]
    out = question_only_baseline(train, test, "compare_cs")
    assert out["per_template_majority"]["seen_acc_pct"] == 50.0
