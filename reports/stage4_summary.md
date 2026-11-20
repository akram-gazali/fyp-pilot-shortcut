# Stage 4 summary (Colab feasibility)

All numbers are PILOT numbers from one run, generated from `reports/stage4/*.json`.

## Data hashes
Colab rebuild matches committed manifests: `{'questions': True, 'train': True, 'pool': True, 'test': True, 'test_edits': True}`

Environment: `{'python': '3.13.16', 'torch': '2.14.1+cu130', 'cuda': True, 'gpu': 'Tesla T4', 'gpu_memory_gb': 14.6, 'bf16_supported': True, 'transformers': '5.22.1', 'peft': '0.21.2'}`

## Zero-shot reference (Qwen/Qwen2.5-VL-3B-Instruct, 448 px, greedy)

| Plain accuracy, all questions | % |
|---|---|
| held-out (compare_cs), n=500 | 47.6 |
| seen templates, n=1500 | 70.67 |
| all, n=2000 | 64.9 |
| compare_cs | 47.6 |
| compare_shape | 71.2 |
| count_cs | 93.0 |
| exists_cs | 47.8 |

| Joint score, paired subset | n | joint % | original % | relevant edit % | irrelevant edit % |
|---|---|---|---|---|---|
| held_out | 355 | 2.82 | 41.69 | 13.24 | 38.87 |
| seen | 1278 | 43.74 | 67.06 | 57.82 | 66.2 |
| all_paired | 1633 | 34.84 | 61.54 | 48.13 | 60.26 |

Shuffled-image control: all 24.85%, seen 25.73%, held-out 22.2%.
Question-only majority baselines (Stage 3): `{'per_template_majority': {'seen_acc_pct': 44.3, 'held_out_acc_pct': 50.8, 'n_seen': 1500, 'n_held_out': 500}, 'per_template_and_params_majority': {'seen_acc_pct': 42.8, 'held_out_acc_pct': 52.0, 'n_seen': 1500, 'n_held_out': 500}}`.
Speed: 1.7746 s/example at batch 16 (7266 items). Unparsed outputs: 0.

Caveat: the normaliser takes the first yes/no/number in the output, so replies such as `0` or `Yes, 1` to a yes/no question are scored wrong. Low exists/compare accuracy may be answer-format failure, not ability. See `raw_output_examples` in zero_shot.json.

## Memory/speed grid
NOT RUN yet

## Cost extrapolation
NOT RUN yet

## Tiny fine-tune
NOT RUN yet
