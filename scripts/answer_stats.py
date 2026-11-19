"""Answer distribution and majority-answer baseline per template, from a questions.jsonl.

Usage: python scripts/answer_stats.py data/pool/questions.jsonl reports/answer_stats_after.json
"""
import json
import sys
from collections import Counter


def stats(path):
    qs = [json.loads(l) for l in open(path)]
    out = {}
    for t in sorted({q["template"] for q in qs}):
        c = Counter(q["answer"] for q in qs if q["template"] == t)
        n = sum(c.values())
        ans, k = c.most_common(1)[0]
        out[t] = {"n": n, "dist": dict(sorted(c.items())), "majority_answer": ans, "majority_baseline": round(k / n, 4)}
    return out


if __name__ == "__main__":
    s = stats(sys.argv[1])
    json.dump(s, open(sys.argv[2], "w"), indent=2)
    for t, v in s.items():
        print(t, v["n"], v["dist"], "majority", v["majority_answer"], v["majority_baseline"])
