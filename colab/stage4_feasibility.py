#!/usr/bin/env python
"""Stage 4: model feasibility check for SIFT-VQA (Qwen2.5-VL-3B-Instruct + LoRA).

WRITTEN BUT NOT YET RUN ON A GPU. Expect small fixes on the first Colab run (library versions).
This is a feasibility check, NOT evidence for the hypothesis.

Subcommands (run from the repo root, in this order):
  prepare        regenerate data/ from the committed configs (CPU) and compare hashes to the manifests
  zero-shot      frozen reference: plain accuracy on ALL test questions (seen / held-out, by template),
                 joint score on the PAIRED subset (separately), and a shuffled-image baseline
  grid           peak GPU memory and seconds per LoRA training step over resolution x batch size
  cost           extrapolate GPU hours per arm / per study scenario from the measured numbers
  tiny-finetune  OPTIONAL: one tiny fine-tune per arm (seed 0) to confirm the code path runs end to end
  all            prepare + zero-shot + grid + cost

  --dry-run      no model, no GPU: checks data loading and the scoring code with a stand-in predictor
                 that returns the true answers (so every endpoint should read 100%).

Needs: GPU runtime; HF token as Colab secret `HF_TOKEN` (or env var HF_TOKEN); see colab/COLAB_STEPS.md.
The token is read from the environment only and is never written to any output file.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from siftvqa.config import sha256_file  # noqa: E402
from siftvqa.cost import extrapolate  # noqa: E402
from siftvqa.runmeta import write_run_meta  # noqa: E402
from siftvqa.vlm_eval import joint_report, normalize_answer, plain_accuracy_report  # noqa: E402

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
SUFFIX = " Answer with a single word or number."
HELD_OUT = "compare_cs"
LORA_TARGETS = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"  # language model only
SPL = ROOT / "data" / "splits"


# ----------------------------------------------------------------------------- environment
def login_hf():
    """Use the token only for hub authentication; never print or save it."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
        except Exception:
            token = None
    if not token:
        print("WARNING: no HF_TOKEN found (Colab secret or env var). The model is public, so this may still work, "
              "but downloads can be rate-limited.")
        return False
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)
    return True


def env_info():
    info = {"python": sys.version.split()[0]}
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.cuda.is_available())
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update(gpu=p.name, gpu_memory_gb=round(p.total_memory / 2 ** 30, 1), bf16_supported=torch.cuda.is_bf16_supported())
        import transformers
        info["transformers"] = transformers.__version__
        import peft
        info["peft"] = peft.__version__
    except ImportError:
        pass
    return info


def pick_dtype(args):
    import torch
    if args.dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16   # T4: fp16
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]


# ----------------------------------------------------------------------------- data
def read_jsonl(p):
    return [json.loads(l) for l in open(p)]


def img_path(scene_id, edited=False):
    if edited:
        return ROOT / "data" / "test_edits" / f"{scene_id}.png"
    return ROOT / "data" / "pool" / "scenes" / f"{scene_id}.png"


def load_image(path, res):
    from PIL import Image
    return Image.open(path).convert("RGB").resize((res, res), Image.BICUBIC)   # res multiple of 28 for Qwen2.5-VL


def cmd_prepare(args):
    """Regenerate data deterministically (it is git-ignored) and compare to the committed manifests."""
    mdir = ROOT / "data" / "manifests"
    expect = {"questions": json.load(open(mdir / "stage1_manifest.json"))["questions_sha256"]}
    sm = json.load(open(mdir / "stage3_splits_manifest.json"))
    expect.update({k: sm[f"{k}_sha256"] for k in ("train", "pool", "test")})
    expect["test_edits"] = json.load(open(mdir / "stage3_edits_manifest.json"))["test_edits_sha256"]
    steps = [["scripts/gen_pool.py"], ["scripts/score_pool.py"], ["scripts/build_splits.py"]]
    if not args.skip_edits:
        steps.append(["scripts/build_edits.py"])       # ~6-10 min on CPU; needed for the paired-subset joint score
    for s in steps:
        print(">>", " ".join(s), flush=True)
        subprocess.run([sys.executable, *s], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    got = {"questions": sha256_file(ROOT / "data" / "pool" / "questions.jsonl"),
           **{k: sha256_file(SPL / f"{k}.jsonl") for k in ("train", "pool", "test")}}
    if not args.skip_edits:
        got["test_edits"] = sha256_file(SPL / "test_edits.jsonl")
    rep = {k: got[k] == expect[k] for k in got}
    print("hash match with committed manifests:", rep)
    if not all(rep.values()):
        print("WARNING: regenerated data differs from the committed manifests (different Python/library version?). "
              "Results are then not comparable with the local Stage 1-3 numbers.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prepare_hash_check.json").write_text(json.dumps({"match": rep, "env": env_info()}, indent=2))


# ----------------------------------------------------------------------------- model helpers
def load_model(args, for_training=False):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    dtype = pick_dtype(args)
    kw = {"torch_dtype": dtype, "attn_implementation": "sdpa", "device_map": {"": 0}}   # sdpa: no flash-attn needed (T4)
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, **kw)
    # images are resized by us to RxR (multiple of 28); keep the processor from resizing again
    proc = AutoProcessor.from_pretrained(MODEL, min_pixels=28 * 28 * 4, max_pixels=1024 * 28 * 28)
    if for_training:
        model = add_lora(model, args, dtype)
    return model, proc, dtype


def add_lora(model, args, dtype):
    from peft import LoraConfig, get_peft_model
    model.config.use_cache = False
    if args.load_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)       # adapters are kept in fp32 by peft even if the base is fp16
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA r={args.lora_r}: trainable params {trainable:,}")
    return model


def chat_text(proc, question, add_generation_prompt=True):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question + SUFFIX}]}]
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)


# ----------------------------------------------------------------------------- inference
def run_inference(model, proc, items, res, bs, max_new_tokens=6):
    """items: list of dicts {key, path, question}. Returns {key: raw_text}, seconds per example."""
    import torch
    proc.tokenizer.padding_side = "left"
    model.eval()
    out, t0 = {}, time.time()
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        images = [load_image(c["path"], res) for c in chunk]
        texts = [chat_text(proc, c["question"]) for c in chunk]
        enc = proc(text=texts, images=images, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        dec = proc.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for c, d in zip(chunk, dec):
            out[c["key"]] = d
        if (i // bs) % 20 == 0:
            print(f"  inference {i + len(chunk)}/{len(items)}", flush=True)
    torch.cuda.synchronize()
    return out, (time.time() - t0) / max(1, len(items))


def dry_inference(items):
    """--dry-run stand-in: returns the true answer for every item (so scores must be 100%)."""
    for it in items:
        assert Path(it["path"]).exists(), f"missing image {it['path']}"
    return {it["key"]: it["truth"] for it in items}, 0.0


def cmd_zero_shot(args):
    test = read_jsonl(SPL / "test.jsonl")
    edits = read_jsonl(SPL / "test_edits.jsonl") if (SPL / "test_edits.jsonl").exists() else []
    if args.held_out_only:
        test = [r for r in test if r["template"] == HELD_OUT]
    if args.max_questions:
        rng = random.Random(0)
        test = rng.sample(test, min(args.max_questions, len(test)))
    keep = {r["qid"] for r in test}
    edits = [e for e in edits if e["qid"] in keep]
    qtext = {r["qid"]: r["question"] for r in test}

    items = [{"key": r["qid"], "path": img_path(r["scene_id"]), "question": r["question"], "truth": r["answer"]} for r in test]
    for e in edits:
        items.append({"key": e["relevant_scene"], "path": img_path(e["relevant_scene"], True), "question": qtext[e["qid"]], "truth": e["answers"]["relevant"]})
        items.append({"key": e["irrelevant_scene"], "path": img_path(e["irrelevant_scene"], True), "question": qtext[e["qid"]], "truth": e["answers"]["irrelevant"]})
    shuf = []
    if not args.skip_shuffle:       # model-based shuffled-image baseline: right question, a DIFFERENT scene's image
        rng = random.Random(1)
        for r in test:
            o = rng.choice(test)
            while o["scene_id"] == r["scene_id"]:
                o = rng.choice(test)
            shuf.append({"key": r["qid"] + "__shuf", "path": img_path(o["scene_id"]), "question": r["question"], "truth": r["answer"]})
        items += shuf
    print(f"zero-shot items: {len(test)} originals, {2 * len(edits)} edited images, {len(shuf)} shuffled")

    if args.dry_run:
        raw, spe = dry_inference(items)
    else:
        login_hf()
        model, proc, _ = load_model(args)
        raw, spe = run_inference(model, proc, items, args.resolution, args.batch_size)
    norm = {k: normalize_answer(v) for k, v in raw.items()}

    rep = {"PILOT_NOTE": "frozen-model reference (headroom). Pilot number, one run, greedy decoding.",
           "model": MODEL, "resolution": args.resolution, "dry_run": args.dry_run, "env": env_info(),
           "plain_accuracy_all_questions": plain_accuracy_report(test, norm, HELD_OUT),
           "joint_score_paired_subset": joint_report(edits, norm, HELD_OUT),
           "shuffled_image_baseline": plain_accuracy_report(
               [{**r, "qid": r["qid"] + "__shuf"} for r in test], norm, HELD_OUT) if shuf else None,
           "majority_baselines_from_stage3_for_comparison": "see reports/stage3_leakage.json",
           "timing": {"sec_per_example": round(spe, 4), "n_items": len(items), "batch_size": args.batch_size},
           "raw_output_examples": dict(list(raw.items())[:15]),
           "unparsed_outputs": sum(1 for v in norm.values() if not (v in ("yes", "no") or v.isdigit()))}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = "zero_shot_dry_run.json" if args.dry_run else "zero_shot.json"
    (out / name).write_text(json.dumps(rep, indent=2))
    write_run_meta(out, vars(args), stage="4-zero-shot")
    print(json.dumps({k: rep[k] for k in ("plain_accuracy_all_questions", "joint_score_paired_subset", "shuffled_image_baseline", "timing")}, indent=2))


# ----------------------------------------------------------------------------- training
def make_train_batch(proc, rows, res, device):
    """Chat prompt + answer; loss only on the answer tokens (prompt, image and padding masked)."""
    proc.tokenizer.padding_side = "right"
    images = [load_image(img_path(r["scene_id"]), res) for r in rows]
    prompts = [chat_text(proc, r["question"]) for r in rows]
    full = [p + r["answer"] + "<|im_end|>" for p, r in zip(prompts, rows)]
    enc_p = proc(text=prompts, images=images, return_tensors="pt", padding=True)
    enc = proc(text=full, images=images, return_tensors="pt", padding=True)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    for i, n in enumerate(enc_p["attention_mask"].sum(1).tolist()):
        labels[i, :n] = -100
    enc["labels"] = labels
    return {k: v.to(device) for k, v in enc.items()}


def train_on_batch(model, dtype, batch, opt, scaler):
    """One optimisation step. fp16 (T4) uses loss scaling; bf16 does not need it."""
    import torch
    with torch.autocast("cuda", dtype=dtype):
        loss = model(**batch).loss
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    return float(loss)


def make_opt(model, dtype, lr):
    import torch
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    return opt, torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))


def train_steps(model, proc, dtype, rows, res, bs, epochs, lr, seed, max_steps=None):
    opt, scaler = make_opt(model, dtype, lr)
    rng, losses, step, t0 = random.Random(seed), [], 0, time.time()
    model.train()
    for _ep in range(epochs):
        order = rows[:]
        rng.shuffle(order)
        for i in range(0, len(order), bs):
            batch = make_train_batch(proc, order[i:i + bs], res, model.device)
            losses.append(train_on_batch(model, dtype, batch, opt, scaler))
            step += 1
            if step % 10 == 0:
                print(f"  step {step} loss {sum(losses[-10:]) / 10:.4f}", flush=True)
            if max_steps and step >= max_steps:
                return {"steps": step, "losses": losses, "seconds": time.time() - t0}
    return {"steps": step, "losses": losses, "seconds": time.time() - t0}


def cmd_grid(args):
    """Peak memory and seconds/step for one LoRA training step over resolution x batch size."""
    plan = [(r, b) for r in args.grid_res for b in args.grid_bs]
    print("grid plan (resolution, batch):", plan)
    if args.dry_run:
        return
    import torch
    login_hf()
    model, proc, dtype = load_model(args, for_training=True)
    pool = read_jsonl(SPL / "pool.jsonl")[: max(args.grid_bs) * 2]
    rows, oom_at = [], {}
    for res, bs in plan:
        if res in oom_at and bs >= oom_at[res]:
            rows.append({"resolution": res, "batch": bs, "status": "skipped (smaller batch already OOM)"})
            continue
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t_pre = time.time()
            batch = make_train_batch(proc, pool[:bs], res, model.device)
            pre = time.time() - t_pre
            ntok = int(batch["attention_mask"].sum(1).float().mean())
            opt, scaler = make_opt(model, dtype, 1e-5)
            for _ in range(2):                                   # warm-up (kernel selection, allocator)
                train_on_batch(model, dtype, batch, opt, scaler)
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(args.grid_steps):                     # same batch re-used: pure compute time
                train_on_batch(model, dtype, batch, opt, scaler)
            torch.cuda.synchronize()
            sec = (time.time() - t) / args.grid_steps + pre     # add preprocessing (image load + tokenise) once per step
            rows.append({"resolution": res, "batch": bs, "status": "ok", "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                         "sec_per_step_incl_preprocess": round(sec, 3), "preprocess_sec_per_batch": round(pre, 3), "tokens_per_sample": ntok})
        except torch.cuda.OutOfMemoryError:
            oom_at[res] = min(oom_at.get(res, 10 ** 9), bs)
            rows.append({"resolution": res, "batch": bs, "status": "OOM"})
            torch.cuda.empty_cache()
        print(rows[-1], flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "grid.json").write_text(json.dumps({"PILOT_NOTE": "measured on this GPU only; do not transfer to other GPUs",
                                               "env": env_info(), "dtype": str(dtype), "lora_r": args.lora_r,
                                               "gradient_checkpointing": True, "rows": rows}, indent=2))
    write_run_meta(out, vars(args), stage="4-grid")


def cmd_cost(args):
    out = Path(args.out)
    grid = json.load(open(out / "grid.json"))
    ok = [r for r in grid["rows"] if r["status"] == "ok"]
    if not ok:
        sys.exit("no successful grid rows; run `grid` first")
    cfgs = [r for r in ok if r["resolution"] == args.cost_res and r["batch"] == args.cost_bs] or \
           sorted(ok, key=lambda r: (-r["resolution"], -r["batch"]))[:1]
    c = cfgs[0]
    zs = out / "zero_shot.json"
    if zs.exists():
        spe, src = json.load(open(zs))["timing"]["sec_per_example"], "measured in zero-shot"
    else:
        spe, src = args.sec_per_eval, "ASSUMED (zero_shot.json not found)"
    n_test = len(read_jsonl(SPL / "test.jsonl"))
    n_pairs = len(read_jsonl(SPL / "test_edits.jsonl")) if (SPL / "test_edits.jsonl").exists() else 0
    n_eval = n_test + 2 * n_pairs + n_test       # originals + edits + shuffled-image pass
    sps = c["sec_per_step_incl_preprocess"]
    scen = [{"name": "pilot tiny fine-tune (400 ex, 1 epoch, 4 arms x 1 seed)", "n_train": 400, "epochs": 1, "runs": 4},
            {"name": "400 ex, 3 epochs, 4 arms x 1 seed", "n_train": 400, "epochs": 3, "runs": 4},
            {"name": "400 ex, 3 epochs, shortcut+random x3 seeds, difficulty+diversity x1", "n_train": 400, "epochs": 3, "runs": 8},
            {"name": "400 ex, 3 epochs, 4 arms x 3 seeds", "n_train": 400, "epochs": 3, "runs": 12},
            {"name": "1000 ex, 3 epochs, 4 arms x 3 seeds", "n_train": 1000, "epochs": 3, "runs": 12}]
    res = extrapolate(sps, c["batch"], spe, n_eval, scen, budget_hours=args.budget_hours)
    rep = {"PILOT_NOTE": "extrapolation from measured step time; ASSUMED parts are labelled",
           "based_on": {"resolution": c["resolution"], "batch": c["batch"], "sec_per_step": sps, "peak_mem_gb": c["peak_mem_gb"],
                        "sec_per_eval_example": spe, "sec_per_eval_example_source": src, "eval_items_per_run": n_eval,
                        "gpu": grid["env"].get("gpu")},
           "budget_hours": args.budget_hours, "scenarios": res,
           "caveat": "numbers are for this GPU; an A100 is several times faster than a T4, and model load (~2 min) is included per run"}
    (out / "cost_extrapolation.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


def cmd_tiny(args):
    """OPTIONAL. One tiny fine-tune per arm, seed 0. Proves the code path runs; proves nothing else."""
    mode = args.exists_mode
    man = json.load(open(ROOT / "data" / "manifests" / f"selection_{mode}_seed0.json"))
    pool = {r["qid"]: r for r in read_jsonl(SPL / "pool.jsonl")}
    test = read_jsonl(SPL / "test.jsonl")
    rng = random.Random(0)
    held = rng.sample([r for r in test if r["template"] == HELD_OUT], args.tiny_eval_n)
    seen = rng.sample([r for r in test if r["template"] != HELD_OUT], args.tiny_eval_n)
    evalrows = held + seen
    items = [{"key": r["qid"], "path": img_path(r["scene_id"]), "question": r["question"], "truth": r["answer"]} for r in evalrows]
    results = {"BANNER": "PIPELINE CHECK ONLY: one seed, tiny budget. This shows the code path runs end to end. "
                         "It is NOT evidence for or against the hypothesis; differences between arms are within noise.",
               "exists_mode": mode, "env": env_info(), "arms": {}}
    if args.dry_run:
        print("dry-run: arms", {a: len(v) for a, v in man["arms"].items()}, "eval items", len(items))
        return
    login_hf()
    import torch
    for arm, qids in man["arms"].items():
        print(f"=== arm {arm}")
        model, proc, dtype = load_model(args, for_training=True)
        train_rows = [pool[q] for q in qids][: args.tiny_n]
        with model.disable_adapter():                      # same eval subset before training, for reference
            raw0, _ = run_inference(model, proc, items, args.resolution, args.batch_size)
        tr = train_steps(model, proc, dtype, train_rows, args.resolution, args.tiny_bs, 1, args.lr, seed=0)
        raw1, spe = run_inference(model, proc, items, args.resolution, args.batch_size)
        acc = lambda raw: plain_accuracy_report(evalrows, {k: normalize_answer(v) for k, v in raw.items()}, HELD_OUT)
        results["arms"][arm] = {"n_train": len(train_rows), "steps": tr["steps"], "train_seconds": round(tr["seconds"], 1),
                                "loss_first10": round(sum(tr["losses"][:10]) / min(10, len(tr["losses"])), 4),
                                "loss_last10": round(sum(tr["losses"][-10:]) / min(10, len(tr["losses"])), 4),
                                "before": acc(raw0), "after": acc(raw1)}
        del model
        torch.cuda.empty_cache()
        print(results["arms"][arm])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tiny_finetune.json").write_text(json.dumps(results, indent=2))
    write_run_meta(out, vars(args), stage="4-tiny-finetune")


# ----------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "zero-shot", "grid", "cost", "tiny-finetune", "all"])
    ap.add_argument("--out", default=str(ROOT / "reports" / "stage4"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16"])
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA-style 4-bit base (needs bitsandbytes); use if fp16 OOMs")
    ap.add_argument("--resolution", type=int, default=448, help="image side in pixels, multiple of 28 (224/336/448)")
    ap.add_argument("--batch-size", type=int, default=16, help="inference batch size")
    ap.add_argument("--skip-edits", action="store_true", help="prepare: skip the slow edit build (no joint score then)")
    ap.add_argument("--held-out-only", action="store_true")
    ap.add_argument("--max-questions", type=int, default=None, help="zero-shot smoke test on a random subset")
    ap.add_argument("--skip-shuffle", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--grid-res", type=int, nargs="+", default=[224, 336, 448])
    ap.add_argument("--grid-bs", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--grid-steps", type=int, default=5)
    ap.add_argument("--cost-res", type=int, default=448)
    ap.add_argument("--cost-bs", type=int, default=4)
    ap.add_argument("--sec-per-eval", type=float, default=0.25)
    ap.add_argument("--budget-hours", type=float, default=20.0)
    ap.add_argument("--exists-mode", default="stratified_quota", choices=["stratified_quota", "fixed_shared_base"])
    ap.add_argument("--tiny-n", type=int, default=400)
    ap.add_argument("--tiny-bs", type=int, default=4)
    ap.add_argument("--tiny-eval-n", type=int, default=100, help="held-out AND seen eval questions each")
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    if args.cmd in ("prepare", "all"):
        cmd_prepare(args)
    if args.cmd in ("zero-shot", "all"):
        cmd_zero_shot(args)
    if args.cmd in ("grid", "all"):
        cmd_grid(args)
    if args.cmd in ("cost", "all") and not args.dry_run:
        cmd_cost(args)
    if args.cmd == "tiny-finetune":
        cmd_tiny(args)


if __name__ == "__main__":
    main()
