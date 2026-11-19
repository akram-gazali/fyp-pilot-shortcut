"""Leakage checks and model-free baselines. Everything is computed from the split files."""
import hashlib
import json
from collections import Counter, defaultdict

from .executor import answer_to_str, execute
from . import dsl
from .seeding import make_rng


def scene_hash(scene):
    """Content hash of a scene (what is drawn, where)."""
    key = sorted((o["shape"], o["color"], o["x"], o["y"], o["size"]) for o in scene["objects"])
    return hashlib.sha256(json.dumps(key).encode()).hexdigest()


def attribute_multiset(scene):
    return tuple(sorted((o["shape"], o["color"]) for o in scene["objects"]))


def signature_overlap(rows_by_split, held_out_sig):
    sigs = {s: dict(Counter(r["signature"] for r in rows)) for s, rows in rows_by_split.items()}
    return {
        "signature_counts_by_split": sigs,
        "held_out_signature": held_out_sig,
        "held_out_in_train": sigs["train"].get(held_out_sig, 0),
        "held_out_in_pool": sigs["pool"].get(held_out_sig, 0),
        "held_out_in_test": sigs["test"].get(held_out_sig, 0),
        "seen_signatures_in_test_missing_from_train": sorted(
            s for s in sigs["test"] if s != held_out_sig and s not in sigs["train"]
        ),
    }


def scene_group_overlap(rows_by_split):
    sets = {s: {r["scene_id"] for r in rows} for s, rows in rows_by_split.items()}
    names = list(sets)
    return {f"{a}&{b}": len(sets[a] & sets[b]) for i, a in enumerate(names) for b in names[i + 1:]}


def duplicate_scenes(scenes_by_split):
    """scenes_by_split: split -> {scene_id: scene}. Exact content duplicates, and (informational)
    scenes with the same multiset of colour+shape attributes across splits."""
    seen, dup_within, dup_across = {}, 0, 0
    ms = defaultdict(set)
    for split, scenes in scenes_by_split.items():
        for sid, sc in scenes.items():
            h = scene_hash(sc)
            if h in seen:
                if seen[h] == split:
                    dup_within += 1
                else:
                    dup_across += 1
            else:
                seen[h] = split
            ms[attribute_multiset(sc)].add(split)
    return {
        "exact_duplicate_scenes_within_split": dup_within,
        "exact_duplicate_scenes_across_splits": dup_across,
        "attribute_multisets_shared_across_splits_(info_only)": sum(1 for v in ms.values() if len(v) > 1),
        "attribute_multisets_total": len(ms),
    }


def duplicate_questions_across_splits(rows_by_split, scenes_by_split):
    seen, dup = {}, 0
    for split, rows in rows_by_split.items():
        for r in rows:
            key = (scene_hash(scenes_by_split[split][r["scene_id"]]), r["question"])
            if key in seen and seen[key] != split:
                dup += 1
            seen.setdefault(key, split)
    return dup


def _param_key(r):
    return (r["template"], tuple(sorted(r["params"].items())))


def question_only_baseline(train, test, held_out_template):
    """Answer from the question text alone (no image). Fit on train, score on test.
    A) per-template majority; unseen template -> global yes/no majority.
    B) per-(template, parameters) majority; fallback chain
       (template, params) -> template -> for the held-out template, the same shape pair in
       compare_shape -> global yes/no majority."""
    by_t, by_p = defaultdict(Counter), defaultdict(Counter)
    yn = Counter()
    pair = defaultdict(Counter)
    for r in train:
        by_t[r["template"]][r["answer"]] += 1
        by_p[_param_key(r)][r["answer"]] += 1
        if r["answer"] in ("yes", "no"):
            yn[r["answer"]] += 1
        if r["template"] == "compare_shape":
            pair[(r["params"]["shape_a"], r["params"]["shape_b"])][r["answer"]] += 1
    glob = yn.most_common(1)[0][0]
    top = lambda c: c.most_common(1)[0][0]

    def pred_a(r):
        return top(by_t[r["template"]]) if by_t.get(r["template"]) else glob

    def pred_b(r):
        if by_p.get(_param_key(r)):
            return top(by_p[_param_key(r)])
        if by_t.get(r["template"]):
            return top(by_t[r["template"]])
        k = (r["params"].get("shape_a"), r["params"].get("shape_b"))
        return top(pair[k]) if pair.get(k) else glob

    out = {}
    for name, f in (("per_template_majority", pred_a), ("per_template_and_params_majority", pred_b)):
        seen = [r for r in test if r["template"] != held_out_template]
        held = [r for r in test if r["template"] == held_out_template]
        acc = lambda rs: round(100 * sum(f(r) == r["answer"] for r in rs) / len(rs), 1) if rs else None
        out[name] = {"seen_acc_pct": acc(seen), "held_out_acc_pct": acc(held), "n_seen": len(seen), "n_held_out": len(held)}
    return out


def shuffled_scene_accuracy(test, scenes_by_id, held_out_template, seed, reps=20):
    rng = make_rng(seed, "shuffle")
    res = {"seen": [0, 0], "held_out": [0, 0]}
    for _ in range(reps):
        for r in test:
            other = test[rng.randrange(len(test))]
            while other["scene_id"] == r["scene_id"]:
                other = test[rng.randrange(len(test))]
            pred = answer_to_str(execute(dsl.from_dict(r["program"]), scenes_by_id[other["scene_id"]]))
            k = "held_out" if r["template"] == held_out_template else "seen"
            res[k][0] += pred == r["answer"]
            res[k][1] += 1
    return {k: round(100 * c / n, 1) if n else None for k, (c, n) in res.items()}
