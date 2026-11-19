"""Cost extrapolation from measured seconds per step / per example. Pure arithmetic."""
from math import ceil


def arm_hours(sec_per_step, batch_size, n_train, epochs, sec_per_eval_example, n_eval_examples, load_overhead_sec=120):
    steps = ceil(n_train / batch_size) * epochs
    train = steps * sec_per_step
    evalt = n_eval_examples * sec_per_eval_example
    return {"steps": steps, "train_hours": train / 3600, "eval_hours": evalt / 3600,
            "total_hours": (train + evalt + load_overhead_sec) / 3600}


def extrapolate(sec_per_step, batch_size, sec_per_eval_example, n_eval_examples, scenarios, budget_hours=20.0,
                load_overhead_sec=120):
    """scenarios: list of {name, n_train, epochs, runs} where runs = total arm-training runs
    (arms x seeds). Returns per-scenario hours and the share of the GPU-hour budget."""
    out = []
    for s in scenarios:
        a = arm_hours(sec_per_step, batch_size, s["n_train"], s["epochs"], sec_per_eval_example, n_eval_examples, load_overhead_sec)
        total = a["total_hours"] * s["runs"]
        out.append({**s, **{k: round(v, 3) if isinstance(v, float) else v for k, v in a.items()},
                    "total_gpu_hours": round(total, 2), "share_of_budget_pct": round(100 * total / budget_hours, 1),
                    "fits_budget": total <= budget_hours})
    return out
