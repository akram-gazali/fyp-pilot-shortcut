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
```json
{
 "PILOT_NOTE": "measured on this GPU only; do not transfer to other GPUs",
 "TRANSCRIBED": "rows copied by hand from the printed Colab log of the first grid run (the original grid.json was lost when the runtime restarted); skipped rows follow the script's skip rule (not printed in the log)",
 "env": {
  "python": "3.13.16",
  "torch": "2.14.1+cu130",
  "cuda": true,
  "gpu": "Tesla T4",
  "gpu_memory_gb": 14.6,
  "bf16_supported": true,
  "transformers": "5.22.1",
  "peft": "0.21.2"
 },
 "dtype": "torch.float16 (assumed: T4 path)",
 "lora_r": 16,
 "gradient_checkpointing": true,
 "rows": [
  {
   "resolution": 224,
   "batch": 1,
   "status": "ok",
   "peak_mem_gb": 8.62,
   "sec_per_step_incl_preprocess": 1.163,
   "preprocess_sec_per_batch": 0.075,
   "tokens_per_sample": 102
  },
  {
   "resolution": 224,
   "batch": 2,
   "status": "ok",
   "peak_mem_gb": 9.84,
   "sec_per_step_incl_preprocess": 2.015,
   "preprocess_sec_per_batch": 0.017,
   "tokens_per_sample": 102
  },
  {
   "resolution": 224,
   "batch": 4,
   "status": "ok",
   "peak_mem_gb": 12.29,
   "sec_per_step_incl_preprocess": 3.793,
   "preprocess_sec_per_batch": 0.026,
   "tokens_per_sample": 101
  },
  {
   "resolution": 224,
   "batch": 8,
   "status": "OOM"
  },
  {
   "resolution": 336,
   "batch": 1,
   "status": "ok",
   "peak_mem_gb": 9.89,
   "sec_per_step_incl_preprocess": 2.005,
   "preprocess_sec_per_batch": 0.029,
   "tokens_per_sample": 182
  },
  {
   "resolution": 336,
   "batch": 2,
   "status": "ok",
   "peak_mem_gb": 12.32,
   "sec_per_step_incl_preprocess": 4.037,
   "preprocess_sec_per_batch": 0.04,
   "tokens_per_sample": 182
  },
  {
   "resolution": 336,
   "batch": 4,
   "status": "OOM"
  },
  {
   "resolution": 336,
   "batch": 8,
   "status": "skipped (smaller batch already OOM)"
  },
  {
   "resolution": 448,
   "batch": 1,
   "status": "ok",
   "peak_mem_gb": 11.69,
   "sec_per_step_incl_preprocess": 3.636,
   "preprocess_sec_per_batch": 0.03,
   "tokens_per_sample": 294
  },
  {
   "resolution": 448,
   "batch": 2,
   "status": "OOM"
  },
  {
   "resolution": 448,
   "batch": 4,
   "status": "skipped (smaller batch already OOM)"
  },
  {
   "resolution": 448,
   "batch": 8,
   "status": "skipped (smaller batch already OOM)"
  }
 ]
}
```

## Cost extrapolation
```json
{
 "PILOT_NOTE": "extrapolation from measured step time; ASSUMED parts are labelled",
 "based_on": {
  "resolution": 448,
  "batch": 1,
  "sec_per_step": 3.636,
  "peak_mem_gb": 11.69,
  "sec_per_eval_example": 1.7746,
  "sec_per_eval_example_source": "measured in zero-shot",
  "eval_items_per_run": 7266,
  "gpu": "Tesla T4"
 },
 "budget_hours": 20.0,
 "scenarios": [
  {
   "name": "pilot tiny fine-tune (400 ex, 1 epoch, 4 arms x 1 seed)",
   "n_train": 400,
   "epochs": 1,
   "runs": 4,
   "steps": 400,
   "train_hours": 0.404,
   "eval_hours": 3.582,
   "total_hours": 4.019,
   "total_gpu_hours": 16.08,
   "share_of_budget_pct": 80.4,
   "fits_budget": true
  },
  {
   "name": "400 ex, 3 epochs, 4 arms x 1 seed",
   "n_train": 400,
   "epochs": 3,
   "runs": 4,
   "steps": 1200,
   "train_hours": 1.212,
   "eval_hours": 3.582,
   "total_hours": 4.827,
   "total_gpu_hours": 19.31,
   "share_of_budget_pct": 96.5,
   "fits_budget": true
  },
  {
   "name": "400 ex, 3 epochs, shortcut+random x3 seeds, difficulty+diversity x1",
   "n_train": 400,
   "epochs": 3,
   "runs": 8,
   "steps": 1200,
   "train_hours": 1.212,
   "eval_hours": 3.582,
   "total_hours": 4.827,
   "total_gpu_hours": 38.62,
   "share_of_budget_pct": 193.1,
   "fits_budget": false
  },
  {
   "name": "400 ex, 3 epochs, 4 arms x 3 seeds",
   "n_train": 400,
   "epochs": 3,
   "runs": 12,
   "steps": 1200,
   "train_hours": 1.212,
   "eval_hours": 3.582,
   "total_hours": 4.827,
   "total_gpu_hours": 57.92,
   "share_of_budget_pct": 289.6,
   "fits_budget": false
  },
  {
   "name": "1000 ex, 3 epochs, 4 arms x 3 seeds",
   "n_train": 1000,
   "epochs": 3,
   "runs": 12,
   "steps": 3000,
   "train_hours": 3.03,
   "eval_hours": 3.582,
   "total_hours": 6.645,
   "total_gpu_hours": 79.74,
   "share_of_budget_pct": 398.7,
   "fits_budget": false
  }
 ],
 "caveat": "numbers are for this GPU; an A100 is several times faster than a T4, and model load (~2 min) is included per run"
}
```

## Tiny fine-tune
NOT RUN yet
