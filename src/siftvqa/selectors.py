"""Example selectors: pure functions over a fixed pool.

Pool rows are dicts with: qid, template, answer, d, d_per_shortcut, n_objects, params,
n_distinct_colors, n_distinct_shapes. Only the CHOICE of examples differs between arms:
same pool, same size, same template mix, same shared base.

Design of the quotas
 * compare_shape (and exists_cs when exists_mode == fixed_shared_base) form a SHARED BASE:
   one seeded random sample, identical in every arm of a given seed.
 * The shortcut-aware arm picks d=1 examples (count_cs; exists_cs "no" part). Its answer-value
   counts per template are whatever that pick gives.
 * matched random copies those answer counts (and the object-count bucket mix) exactly.
 * difficulty and diversity copy the template and answer counts (they do not copy buckets,
   because choosing hard / spread-out examples is their point).
"""
from collections import Counter, defaultdict

import numpy as np

from .seeding import make_rng

SELECTABLE = ("count_cs", "exists_cs")
ARMS = ("shortcut_aware", "matched_random", "difficulty", "diversity")


def bucket_of(n, buckets):
    for i, (lo, hi) in enumerate(buckets):
        if lo <= n <= hi:
            return i
    raise ValueError(n)


def make_plan(budget, template_shares, exists_yes_share):
    """Examples per template, with largest-remainder rounding so the total is exactly `budget`."""
    raw = {t: budget * s for t, s in template_shares.items()}
    plan = {t: int(v) for t, v in raw.items()}
    for t in sorted(raw, key=lambda t: raw[t] - plan[t], reverse=True)[: budget - sum(plan.values())]:
        plan[t] += 1
    plan["exists_yes"] = round(plan["exists_cs"] * exists_yes_share)
    return plan


def _rows(pool, template, **cond):
    return [r for r in pool if r["template"] == template and all(r[k] == v for k, v in cond.items())]


def _take(rng, rows, k, exclude=()):
    rows = [r for r in rows if r["qid"] not in exclude]
    k = min(k, len(rows))
    return rng.sample(rows, k)


def shared_base(pool, plan, seed, exists_mode):
    """Examples every arm receives (same set across arms for a given seed)."""
    rng = make_rng(seed, "shared")
    ids = [r["qid"] for r in _take(rng, _rows(pool, "compare_shape"), plan["compare_shape"])]
    if exists_mode == "fixed_shared_base":
        ids += [r["qid"] for r in _take(rng, _rows(pool, "exists_cs"), plan["exists_cs"])]
    return ids


def select_shortcut_aware(pool, plan, seed, exists_mode):
    """d=1 examples for count_cs; for exists_cs (stratified_quota) the "no" part from d=1
    examples and the "yes" part from d=0 examples (every "yes" is d=0 by construction)."""
    rng = make_rng(seed, "shortcut")
    chosen = shared_base(pool, plan, seed, exists_mode)
    info = {"shortfalls": {}}
    c = _take(rng, _rows(pool, "count_cs", d=1), plan["count_cs"])
    chosen += [r["qid"] for r in c]
    if len(c) < plan["count_cs"]:
        info["shortfalls"]["count_cs_d1"] = plan["count_cs"] - len(c)
    if exists_mode == "stratified_quota":
        n_yes = plan["exists_yes"]
        n_no = plan["exists_cs"] - n_yes
        no = _take(rng, _rows(pool, "exists_cs", answer="no", d=1), n_no)
        if len(no) < n_no:
            info["shortfalls"]["exists_no_d1"] = n_no - len(no)
        yes = _take(rng, _rows(pool, "exists_cs", answer="yes"), n_yes)
        chosen += [r["qid"] for r in no] + [r["qid"] for r in yes]
    return chosen, info


def quota_of(pool_by_id, qids, templates, buckets):
    """(template, answer, bucket) -> count for the non-shared templates among `qids`."""
    q = Counter()
    for i in qids:
        r = pool_by_id[i]
        if r["template"] in templates:
            q[(r["template"], r["answer"], bucket_of(r["n_objects"], buckets))] += 1
    return q


def _selectable_templates(exists_mode):
    return ("count_cs",) if exists_mode == "fixed_shared_base" else SELECTABLE


def select_matched_random(pool, quota, plan, seed, exists_mode, match_buckets, buckets):
    """Random examples with the same (template, answer[, object-count bucket]) counts as the
    shortcut-aware arm. If a bucket cell is short, borrow from the same (template, answer)."""
    rng = make_rng(seed, "random")
    chosen = shared_base(pool, plan, seed, exists_mode)
    used = set(chosen)
    borrowed = 0
    if match_buckets:
        for (t, a, b), k in sorted(quota.items()):
            cell = [r for r in _rows(pool, t, answer=a) if bucket_of(r["n_objects"], buckets) == b]
            got = _take(rng, cell, k, used)
            used |= {r["qid"] for r in got}
            chosen += [r["qid"] for r in got]
            if len(got) < k:
                borrowed += k - len(got)
                more = _take(rng, _rows(pool, t, answer=a), k - len(got), used)
                used |= {r["qid"] for r in more}
                chosen += [r["qid"] for r in more]
    else:
        per = Counter()
        for (t, a, _b), k in quota.items():
            per[(t, a)] += k
        for (t, a), k in sorted(per.items()):
            got = _take(rng, _rows(pool, t, answer=a), k, used)
            used |= {r["qid"] for r in got}
            chosen += [r["qid"] for r in got]
    return chosen, {"borrowed_across_buckets": borrowed}


def _label_quota(quota):
    per = Counter()
    for (t, a, _b), k in quota.items():
        per[(t, a)] += k
    return per


def select_difficulty(pool, quota, plan, seed, exists_mode):
    """Within each (template, answer) cell take the hardest examples. Difficulty is a model-free
    proxy: number of objects + program length (ties broken by the seeded RNG)."""
    rng = make_rng(seed, "difficulty")
    chosen = shared_base(pool, plan, seed, exists_mode)
    for (t, a), k in sorted(_label_quota(quota).items()):
        cell = _rows(pool, t, answer=a)
        rng.shuffle(cell)
        cell.sort(key=lambda r: -(r["n_objects"] + r["program_length"]))
        chosen += [r["qid"] for r in cell[:k]]
    return chosen, {}


_COLORS = ["red", "blue", "green", "yellow", "purple", "orange"]
_SHAPES = ["square", "circle", "triangle"]


def features(r):
    p = r["params"]
    f = [1.0 * (p.get("color") == c) for c in _COLORS]
    f += [1.0 * (p.get("shape", p.get("shape_a")) == s) for s in _SHAPES]
    f += [1.0 * (p.get("shape_b") == s) for s in _SHAPES]
    f += [r["n_objects"] / 8, r["n_distinct_colors"] / 6, r["n_distinct_shapes"] / 3]
    return f


def select_diversity(pool, quota, plan, seed, exists_mode):
    """Within each (template, answer) cell, greedy farthest-first selection over question
    parameters and scene statistics (coverage of the feature space)."""
    rng = make_rng(seed, "diversity")
    chosen = shared_base(pool, plan, seed, exists_mode)
    for (t, a), k in sorted(_label_quota(quota).items()):
        cell = _rows(pool, t, answer=a)
        if k >= len(cell):
            chosen += [r["qid"] for r in cell]
            continue
        X = np.array([features(r) for r in cell])
        first = rng.randrange(len(cell))
        picked = [first]
        dist = np.linalg.norm(X - X[first], axis=1)
        while len(picked) < k:
            nxt = int(np.argmax(dist))
            picked.append(nxt)
            dist = np.minimum(dist, np.linalg.norm(X - X[nxt], axis=1))
        chosen += [cell[i]["qid"] for i in picked]
    return chosen, {}


def run_all_arms(pool, cfg, seed):
    """Run the four selectors for one seed; returns {arm: (qids, info)}, quota."""
    plan = make_plan(cfg["budget"], cfg["template_shares"], cfg["exists_yes_share"])
    mode = cfg["exists_mode"]
    buckets = cfg["n_objects_buckets"]
    by_id = {r["qid"]: r for r in pool}
    sc_ids, sc_info = select_shortcut_aware(pool, plan, seed, mode)
    quota = quota_of(by_id, sc_ids, _selectable_templates(mode), buckets)
    arms = {
        "shortcut_aware": (sc_ids, sc_info),
        "matched_random": select_matched_random(pool, quota, plan, seed, mode, cfg["match_n_objects"], buckets),
        "difficulty": select_difficulty(pool, quota, plan, seed, mode),
        "diversity": select_diversity(pool, quota, plan, seed, mode),
    }
    for name, (ids, _) in arms.items():
        assert len(ids) == cfg["budget"] and len(set(ids)) == len(ids), f"{name}: size {len(ids)}"
    return arms, quota, plan
