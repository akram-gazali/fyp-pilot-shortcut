#!/usr/bin/env python
"""Stages 4/5: model feasibility and evaluation-format study for SIFT-VQA (Qwen2.5-VL-3B-Instruct + LoRA).

GPU parts have NOT been run since the Stage 5 rewrite. This is a feasibility check, NOT evidence for the hypothesis.

Subcommands (run from the repo root, in this order):
  check-env      print library versions, fail if a pinned non-torch package differs from requirements-gpu.txt
  prepare        regenerate data/ from the committed configs (CPU) and compare hashes to the manifests
  dev-format     choose the answer-format method (suffix vs choice scoring) and image resolution on the DEV set
                 (the train split). The test set is never used to choose either.
  zero-shot      frozen reference on the test set with the chosen method: plain accuracy on ALL questions,
                 joint score on the PAIRED subset (separately), shuffled-image baseline
  grid           peak GPU memory and seconds per LoRA training step; writes grid.json row by row
  cost           extrapolate GPU hours per arm / per study scenario from the measured numbers
  tiny-finetune  OPTIONAL: one tiny fine-tune per arm (seed 0) to confirm the code path runs end to end

  --dry-run      no model, no GPU: checks data loading and the scoring code with a stand-in predictor.

Platform: Kaggle (preferred) or Colab. On Kaggle one GPU is used (CUDA_VISIBLE_DEVICES=0), the Hugging Face cache goes
to /kaggle/temp and results to /kaggle/working. No Colab-only calls are needed. The HF token is optional (public model);
if present it is read from env HF_TOKEN or the Kaggle/Colab secret store and never printed or saved.
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

ON_KAGGLE = Path("/kaggle").exists()
if ON_KAGGLE:                                   # must happen before torch is imported
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("HF_HOME", "/kaggle/temp/hf")

from siftvqa import formats  # noqa: E402
from siftvqa.config import sha256_file  # noqa: E402
from siftvqa.cost import extrapolate  # noqa: E402
from siftvqa.runmeta import write_run_meta  # noqa: E402
from siftvqa.train_utils import build_train_batch, lora_report, seed_all  # noqa: E402
from siftvqa.vlm_eval import accuracy, joint_report, normalize_answer, plain_accuracy_report  # noqa: E402

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
HELD_OUT = "compare_cs"
LORA_TARGETS = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"  # language model only
SPL = ROOT / "data" / "splits"
DEFAULT_OUT = "/kaggle/working/stage5_results" if ON_KAGGLE else str(ROOT / "reports" / "stage5")
PINS = ROOT / "requirements-gpu.txt"


def jdump(path, obj):
    formats.write_json_atomic(path, obj)


# ----------------------------------------------------------------------------- environment
def login_hf():
    """Optional (the model is public). Token only for hub authentication; never printed or saved."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        for mod, getter in (("kaggle_secrets", lambda m: m.UserSecretsClient().get_secret("HF_TOKEN")),
                            ("google.colab.userdata", lambda m: m.get("HF_TOKEN"))):
            try:
                import importlib
                token = getter(importlib.import_module(mod))
                if token:
                    break
            except Exception:
                token = None
    if not token:
        print("note: no HF_TOKEN found; the model is public, so this is fine (downloads may be rate-limited).")
        return False
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)
    return True


def env_info():
    info = {"python": sys.version.split()[0], "platform": "kaggle" if ON_KAGGLE else "other"}
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.cuda.is_available())
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update(gpu=p.name, gpu_memory_gb=round(p.total_memory / 2 ** 30, 1), compute_capability=f"{p.major}.{p.minor}",
                        bf16_supported=torch.cuda.is_bf16_supported())
        import transformers
        info["transformers"] = transformers.__version__
        import peft
        info["peft"] = peft.__version__
    except ImportError:
        pass
    return info


def pick_dtype(args):
    """T4 (compute capability 7.5) -> fp16 even though torch reports bf16 'supported' (emulated, slow)."""
    import torch
    major = torch.cuda.get_device_properties(0).major if torch.cuda.is_available() else 8
    name = formats.choose_dtype(args.dtype, major)
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[name]


def dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def preflight(args):
    if args.dry_run or args.skip_version_check:
        return {}
    return formats.check_pins(PINS)


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
    jdump(Path(args.out) / "prepare_hash_check.json", {"match": rep, "env": env_info()})


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
    cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, target_modules=LORA_TARGETS,
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg, autocast_adapter_dtype=True)   # adapters fp32 even on an fp16 base
    rep = lora_report(model)                  # raises if adapters are not fp32 or touch the vision tower
    print(f"LoRA r={args.lora_r}: {rep}")
    return model


def chat_text(proc, question, template, method, add_generation_prompt=True):
    """The ONE prompt builder, used by evaluation and training alike."""
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": formats.prompt_text(question, template, method)}]}]
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)


# ----------------------------------------------------------------------------- inference
def run_generate(model, proc, items, res, bs, method):
    """Greedy free generation. Returns {key: raw_text}, seconds per item."""
    import torch
    proc.tokenizer.padding_side = "left"
    model.eval()
    out, t0 = {}, time.time()
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        images = [load_image(c["path"], res) for c in chunk]
        texts = [chat_text(proc, c["question"], c["template"], method) for c in chunk]
        enc = proc(text=texts, images=images, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=6 if method == "legacy" else 4, do_sample=False, use_cache=True)
        dec = proc.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for c, d in zip(chunk, dec):
            out[c["key"]] = d
        if (i // bs) % 20 == 0:
            print(f"  generate {i + len(chunk)}/{len(items)}", flush=True)
    torch.cuda.synchronize()
    return out, (time.time() - t0) / max(1, len(items))


def candidate_token_ids(proc, templates):
    """{template: {answer: [token ids of its surface forms]}}; every surface form must be ONE token, else stop."""
    tok, table = proc.tokenizer, {}
    for t in set(templates):
        table[t] = {}
        for ans in formats.candidates(t):
            ids = []
            for s in formats.surface_forms(ans):
                enc = tok.encode(s, add_special_tokens=False)
                if len(enc) != 1:
                    raise SystemExit(f"STOP: candidate {s!r} is {len(enc)} tokens; choice scoring needs single-token answers.")
                ids.append(enc[0])
            table[t][ans] = ids
    return table


def run_choice(model, proc, items, res, bs):
    """Choice scoring: one forward pass per batch; the answer is the candidate with the highest log-probability at
    the first answer position. No generation. Returns {key: answer}, seconds per item."""
    import torch
    proc.tokenizer.padding_side = "left"
    model.eval()
    table = candidate_token_ids(proc, [it["template"] for it in items])
    out, t0 = {}, time.time()
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        images = [load_image(c["path"], res) for c in chunk]
        texts = [chat_text(proc, c["question"], c["template"], "choice") for c in chunk]
        enc = proc(text=texts, images=images, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            try:
                logits = model(**enc, logits_to_keep=1).logits[:, -1]          # last position only (saves memory)
            except TypeError:
                logits = model(**enc).logits[:, -1]
            logp = torch.log_softmax(logits.float(), dim=-1).cpu()
        for row, c in enumerate(chunk):
            need = {tid for ids in table[c["template"]].values() for tid in ids}
            best, _ = formats.best_candidate({t: float(logp[row, t]) for t in need}, table[c["template"]])
            out[c["key"]] = best
        if (i // bs) % 20 == 0:
            print(f"  choice {i + len(chunk)}/{len(items)}", flush=True)
    torch.cuda.synchronize()
    return out, (time.time() - t0) / max(1, len(items))


def predict(model, proc, items, res, bs, method):
    """-> (raw, normalised predictions, sec/item). Same call for every arm and every evaluation."""
    raw, spe = run_choice(model, proc, items, res, bs) if method == "choice" else run_generate(model, proc, items, res, bs, method)
    norm = dict(raw) if method == "choice" else {k: normalize_answer(v) for k, v in raw.items()}
    return raw, norm, spe


def dry_predict(items):
    """--dry-run stand-in: returns the true answer for every item (so scores must be 100%)."""
    for it in items:
        assert Path(it["path"]).exists(), f"missing image {it['path']}"
    raw = {it["key"]: it["truth"] for it in items}
    return raw, dict(raw), 0.0


def wrong_format_pct(raw, items, method):
    if method == "choice":
        return 0.0
    n_bad = sum(not formats.strict_format_ok(raw[it["key"]], it["template"]) for it in items)
    return round(100 * n_bad / len(items), 2)


def load_choice(out, args):
    """method/resolution: explicit flags win; otherwise the dev-format study's choice file."""
    f = Path(out) / "eval_choice.json"
    ch = json.load(open(f)) if f.exists() else {}
    method = args.method or ch.get("method")
    res = args.resolution or ch.get("resolution")
    if args.cmd != "cost" and (not method or not res) and not args.dry_run:
        sys.exit("no chosen method/resolution: run `dev-format` first (it writes eval_choice.json) or pass --method and --resolution.")
    return method or "suffix", res or 224


# ----------------------------------------------------------------------------- dev-set format study (Task A/B)
def dev_items(args):
    rows = read_jsonl(SPL / "train.jsonl")                         # DEV set = train split; never the test set
    if args.max_questions:
        rows = random.Random(0).sample(rows, min(args.max_questions, len(rows)))
    items = [{"key": r["qid"], "path": img_path(r["scene_id"]), "question": r["question"], "template": r["template"],
              "truth": r["answer"]} for r in rows]
    return rows, items


def score_run(rows, items, raw, norm, spe, method, res, dtype):
    per_t = {}
    for t in sorted({r["template"] for r in rows}):
        per_t[t] = accuracy([r for r in rows if r["template"] == t], norm)
    return {"method": method, "resolution": res, "dtype": dtype, "n": len(rows), "acc_by_template_pct": per_t,
            "macro_acc_pct": formats.macro(per_t), "overall_acc_pct": accuracy(rows, norm),
            "wrong_format_pct": wrong_format_pct(raw, items, method), "sec_per_item": round(spe, 4),
            "batch_size": None, "raw_examples": dict(list(raw.items())[:10])}


def cmd_dev_format(args):
    preflight(args)
    rows, items = dev_items(args)
    print(f"dev set: {len(rows)} questions from the train split, templates {sorted({r['template'] for r in rows})} "
          "(no compare_cs: its questions were removed from train; ASSUMED to behave like the other yes/no templates)")
    out = Path(args.out)
    if args.dry_run:
        raw, norm, spe = dry_predict(items)
        rep = score_run(rows, items, raw, norm, spe, "dry-run", 0, "none")
        print(json.dumps({k: rep[k] for k in ("acc_by_template_pct", "macro_acc_pct", "wrong_format_pct")}))
        return
    login_hf()
    model, proc, dtype = load_model(args)
    dn = dtype_name(dtype)
    doc = {"PILOT_NOTE": "dev-set study on the TRAIN split; the test set is not used. Pilot numbers, one run each.",
           "env": env_info(), "dtype": dn, "dev_set": {"split": "train", "n": len(rows)}, "runs": {}}

    def run(method, res):
        raw, norm, spe = predict(model, proc, items, res, args.batch_size, method)
        rep = score_run(rows, items, raw, norm, spe, method, res, dn)
        rep["batch_size"] = args.batch_size
        doc["runs"][f"{method}@{res}"] = rep
        jdump(out / "dev_format.json", doc)                                  # written as soon as it exists
        print(f"{method}@{res}: macro {rep['macro_acc_pct']}  wrong-format {rep['wrong_format_pct']}%  {rep['sec_per_item']} s/item", flush=True)
        return rep

    base = args.base_resolution
    res_main = {m: run(m, base) for m in ("suffix", "choice")}
    run("legacy", base)                                                      # diagnostic only, never selectable
    pm = formats.pick_method({m: {"macro_acc": r["macro_acc_pct"], "sec_per_item": r["sec_per_item"]} for m, r in res_main.items()}, tie_pts=1.0)
    hi = run(pm["chosen"], args.compare_resolution)
    pr = formats.pick_resolution({base: res_main[pm["chosen"]]["macro_acc_pct"], args.compare_resolution: hi["macro_acc_pct"]},
                                 base=base, margin_pts=2.0)
    doc["selection"] = {"method": pm, "resolution": pr,
                        "rule": "method: highest dev macro accuracy, within 1 point is a tie -> faster; resolution: native unless the "
                                "larger one is better by MORE than 2 macro points"}
    jdump(out / "dev_format.json", doc)
    jdump(out / "eval_choice.json", {"method": pm["chosen"], "resolution": pr["chosen"], "dtype": dn, "decided_on": "dev set (train split)",
                                     "details": doc["selection"]})
    print("CHOSEN:", pm["chosen"], pr["chosen"])


# ----------------------------------------------------------------------------- zero-shot on the test set
def eval_item(key, path, q, truth):
    return {"key": key, "path": path, "question": q["question"], "template": q["template"], "truth": truth}


def edit_items(edits, qrow):
    """Items for the paired-edit (joint) score: the relevant and the irrelevant edited image of each edit record.
    The original question is scored from the ordinary item for the same qid."""
    items = []
    for e in edits:
        q = qrow[e["qid"]]
        items.append(eval_item(e["relevant_scene"], img_path(e["relevant_scene"], True), q, e["answers"]["relevant"]))
        items.append(eval_item(e["irrelevant_scene"], img_path(e["irrelevant_scene"], True), q, e["answers"]["irrelevant"]))
    return items


def cmd_zero_shot(args):
    preflight(args)
    out = Path(args.out)
    method, res = load_choice(out, args)
    test = read_jsonl(SPL / "test.jsonl")
    edits = read_jsonl(SPL / "test_edits.jsonl") if (SPL / "test_edits.jsonl").exists() else []
    if args.held_out_only:
        test = [r for r in test if r["template"] == HELD_OUT]
    if args.max_questions:
        test = random.Random(0).sample(test, min(args.max_questions, len(test)))
    keep = {r["qid"] for r in test}
    edits = [e for e in edits if e["qid"] in keep]
    qrow = {r["qid"]: r for r in test}

    item = eval_item
    items = [item(r["qid"], img_path(r["scene_id"]), r, r["answer"]) for r in test]
    items += edit_items(edits, qrow)
    shuf = []
    if not args.skip_shuffle:       # model-based shuffled-image baseline: right question, a DIFFERENT scene's image
        rng = random.Random(1)
        for r in test:
            o = rng.choice(test)
            while o["scene_id"] == r["scene_id"]:
                o = rng.choice(test)
            shuf.append(item(r["qid"] + "__shuf", img_path(o["scene_id"]), r, r["answer"]))
        items += shuf
    print(f"zero-shot items: {len(test)} originals, {2 * len(edits)} edited images, {len(shuf)} shuffled; method={method} resolution={res}")

    dn = "none"
    if args.dry_run:
        raw, norm, spe = dry_predict(items)
    else:
        login_hf()
        model, proc, dtype = load_model(args)
        dn = dtype_name(dtype)
        raw, norm, spe = predict(model, proc, items, res, args.batch_size, method)
    rep = {"PILOT_NOTE": "frozen-model reference (headroom). Pilot number, one run; method and resolution were chosen on the dev set.",
           "model": MODEL, "method": method, "resolution": res, "dtype": dn, "dry_run": args.dry_run, "env": env_info(),
           "plain_accuracy_all_questions": plain_accuracy_report(test, norm, HELD_OUT),
           "joint_score_paired_subset": joint_report(edits, norm, HELD_OUT),
           "shuffled_image_baseline": plain_accuracy_report([{**r, "qid": r["qid"] + "__shuf"} for r in test], norm, HELD_OUT) if shuf else None,
           "majority_baselines_from_stage3_for_comparison": "see reports/stage3_leakage.json",
           "timing": {"sec_per_example": round(spe, 4), "n_items": len(items), "batch_size": args.batch_size},
           "wrong_format_pct": wrong_format_pct(raw, items, method),
           "raw_output_examples": dict(list(raw.items())[:15])}
    name = "zero_shot_dry_run.json" if args.dry_run else "zero_shot.json"
    jdump(out / name, rep)
    write_run_meta(out, vars(args), stage="5-zero-shot")
    print(json.dumps({k: rep[k] for k in ("plain_accuracy_all_questions", "joint_score_paired_subset", "shuffled_image_baseline", "timing")}, indent=2))


# ----------------------------------------------------------------------------- training
def make_train_batch(proc, rows, res, device, method):
    images = [load_image(img_path(r["scene_id"]), res) for r in rows]
    prompts = [chat_text(proc, r["question"], r["template"], method) for r in rows]     # same builder as evaluation
    enc = build_train_batch(proc, prompts, [r["answer"] for r in rows], images)
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
    return loss.item()


def make_opt(model, dtype, lr):
    import torch
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    return opt, torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))


def train_steps(model, proc, dtype, rows, res, bs, epochs, lr, seed, method, max_steps=None):
    opt, scaler = make_opt(model, dtype, lr)
    rng, losses, step, t0 = random.Random(seed), [], 0, time.time()
    model.train()
    for _ep in range(epochs):
        order = rows[:]
        rng.shuffle(order)
        for i in range(0, len(order), bs):
            batch = make_train_batch(proc, order[i:i + bs], res, model.device, method)
            losses.append(train_on_batch(model, dtype, batch, opt, scaler))
            step += 1
            if step % 10 == 0:
                print(f"  step {step} loss {sum(losses[-10:]) / 10:.4f}", flush=True)
            if max_steps and step >= max_steps:
                return {"steps": step, "losses": losses, "seconds": time.time() - t0}
    return {"steps": step, "losses": losses, "seconds": time.time() - t0}


def cmd_grid(args):
    """Peak memory and seconds/step for one LoRA training step over resolution x batch size. grid.json is written by this
    function after EVERY row, so a killed session keeps what was measured; no transcription from logs."""
    plan = [(r, b) for r in args.grid_res for b in args.grid_bs]
    print("grid plan (resolution, batch):", plan)
    if args.dry_run:
        return
    preflight(args)
    import torch
    login_hf()
    method, _ = load_choice(Path(args.out), args)
    seed_all(0)
    model, proc, dtype = load_model(args, for_training=True)
    pool = read_jsonl(SPL / "pool.jsonl")[: max(args.grid_bs) * 2]
    rows, oom_at = [], {}
    doc = {"PILOT_NOTE": "measured on this GPU only; do not transfer to other GPUs", "complete": False, "env": env_info(),
           "dtype": dtype_name(dtype), "lora_r": args.lora_r, "gradient_checkpointing": True, "prompt_method": method, "rows": rows}
    gpath = Path(args.out) / "grid.json"
    for res, bs in plan:
        if res in oom_at and bs >= oom_at[res]:
            rows.append({"resolution": res, "batch": bs, "status": "skipped (smaller batch already OOM)"})
            jdump(gpath, doc)
            continue
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t_pre = time.time()
            batch = make_train_batch(proc, pool[:bs], res, model.device, method)
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
            sec = (time.time() - t) / args.grid_steps + pre
            rows.append({"resolution": res, "batch": bs, "status": "ok", "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                         "sec_per_step_incl_preprocess": round(sec, 3), "preprocess_sec_per_batch": round(pre, 3), "tokens_per_sample": ntok})
        except torch.cuda.OutOfMemoryError:
            oom_at[res] = min(oom_at.get(res, 10 ** 9), bs)
            rows.append({"resolution": res, "batch": bs, "status": "OOM"})
            torch.cuda.empty_cache()
        jdump(gpath, doc)
        print(rows[-1], flush=True)
    doc["complete"] = True
    jdump(gpath, doc)
    write_run_meta(Path(args.out), vars(args), stage="5-grid")


def cmd_cost(args):
    out = Path(args.out)
    grid = json.load(open(out / "grid.json"))
    if not grid.get("complete", True) or "TRANSCRIBED" in grid:
        print("WARNING: grid.json is incomplete or transcribed; rerun `grid`.")
    ok = [r for r in grid["rows"] if r["status"] == "ok"]
    if not ok:
        sys.exit("no successful grid rows; run `grid` first")
    _, res = load_choice(out, args)
    cres, cbs = args.cost_res or res, args.cost_bs
    cands = [r for r in ok if r["resolution"] == cres]
    c = ([r for r in cands if r["batch"] == cbs] or sorted(cands, key=lambda r: -r["batch"]) or sorted(ok, key=lambda r: (-r["resolution"], -r["batch"])))[0]
    zs = out / "zero_shot.json"
    if zs.exists():
        spe, src = json.load(open(zs))["timing"]["sec_per_example"], "measured in zero-shot"
    else:
        spe, src = args.sec_per_eval, "ASSUMED (zero_shot.json not found)"
    n_test = len(read_jsonl(SPL / "test.jsonl"))
    n_pairs = len(read_jsonl(SPL / "test_edits.jsonl")) if (SPL / "test_edits.jsonl").exists() else 0
    n_eval = n_test + 2 * n_pairs + n_test       # originals + edits + shuffled-image pass
    sps, sps_src = c["sec_per_step_incl_preprocess"], f"grid ({c['resolution']} px, batch {c['batch']}, 5 timed steps)"
    tiny = out / "tiny_finetune.json"
    if tiny.exists():                      # a real 100-step training run is a better step time than 5 grid steps
        tj = json.load(open(tiny))
        arms = [a for a in tj["arms"].values() if a.get("steps")]
        if arms and tj.get("resolution") == c["resolution"] and args.tiny_bs == c["batch"]:
            sps = round(sum(a["train_seconds"] for a in arms) / sum(a["steps"] for a in arms), 4)
            sps_src = f"measured in tiny-finetune ({len(arms)} arms, {sum(a['steps'] for a in arms)} steps, {c['resolution']} px, batch {c['batch']})"
    scen = [{"name": "pilot tiny fine-tune (400 ex, 1 epoch, 4 arms x 1 seed)", "n_train": 400, "epochs": 1, "runs": 4},
            {"name": "400 ex, 3 epochs, 4 arms x 1 seed", "n_train": 400, "epochs": 3, "runs": 4},
            {"name": "400 ex, 3 epochs, shortcut+random x3 seeds, difficulty+diversity x1", "n_train": 400, "epochs": 3, "runs": 8},
            {"name": "400 ex, 3 epochs, 4 arms x 3 seeds", "n_train": 400, "epochs": 3, "runs": 12},
            {"name": "1000 ex, 3 epochs, 4 arms x 3 seeds", "n_train": 1000, "epochs": 3, "runs": 12}]
    for k in (3, 5):          # larger test sets (power proposal): evaluation items scale with the number of test scenes
        for n_tr in (400, 1000):
            scen.append({"name": f"{n_tr} ex, 3 epochs, 4 arms x 3 seeds, test set {k}x", "n_train": n_tr, "epochs": 3, "runs": 12, "test_scale": k})
    res_rows = []
    for sc in scen:
        k = sc.get("test_scale", 1)
        r = extrapolate(sps, c["batch"], spe, n_eval * k, [sc], budget_hours=args.budget_hours)[0]
        r["eval_items_per_run"] = n_eval * k
        res_rows.append(r)
    prev = ROOT / "reports" / "stage4" / "cost_extrapolation.json"
    rep = {"PILOT_NOTE": "extrapolation from measured step time; ASSUMED parts are labelled",
           "based_on": {"resolution": c["resolution"], "batch": c["batch"], "sec_per_step": sps, "sec_per_step_source": sps_src,
                        "grid_sec_per_step_for_comparison": c["sec_per_step_incl_preprocess"], "peak_mem_gb": c["peak_mem_gb"],
                        "dtype": grid.get("dtype"), "sec_per_eval_example": spe, "sec_per_eval_example_source": src,
                        "eval_items_per_run": n_eval, "gpu": grid["env"].get("gpu")},
           "budget_hours": args.budget_hours, "scenarios": res_rows,
           "previous_stage4_scenarios_for_comparison": json.load(open(prev))["scenarios"] if prev.exists() else None,
           "previous_stage4_based_on": json.load(open(prev))["based_on"] if prev.exists() else None,
           "caveat": "numbers are for this GPU; an A100 is several times faster than a T4, and model load (~2 min) is included per run"}
    jdump(out / "cost_extrapolation.json", rep)
    print(json.dumps(rep, indent=2))


def cmd_tiny(args):
    """OPTIONAL. One tiny fine-tune per arm, seed 0. Proves the code path runs; proves nothing else."""
    mode = args.exists_mode
    mpath = ROOT / "data" / "manifests" / f"selection_{mode}_seed0.json"
    man = json.load(open(mpath))
    pool = {r["qid"]: r for r in read_jsonl(SPL / "pool.jsonl")}
    test = read_jsonl(SPL / "test.jsonl")
    test_scenes = {r["scene_id"] for r in test}
    method, res = load_choice(Path(args.out), args)
    rng = random.Random(0)
    held = rng.sample([r for r in test if r["template"] == HELD_OUT], args.tiny_eval_n)
    seen = rng.sample([r for r in test if r["template"] != HELD_OUT], args.tiny_eval_n)
    evalrows = held + seen
    items = [eval_item(r["qid"], img_path(r["scene_id"]), r, r["answer"]) for r in evalrows]
    # paired edits of the sampled questions, so the PRIMARY metric (joint score) is computed for every trained arm
    all_edits = read_jsonl(SPL / "test_edits.jsonl") if (SPL / "test_edits.jsonl").exists() else []
    keep = {r["qid"] for r in evalrows}
    tiny_edits = [e for e in all_edits if e["qid"] in keep]
    items += edit_items(tiny_edits, {r["qid"]: r for r in evalrows})
    results = {"BANNER": "PIPELINE CHECK ONLY: one seed, tiny budget. This shows the code path runs end to end. "
                         "It is NOT evidence for or against the hypothesis; differences between arms are within noise.",
               "exists_mode": mode, "selection_manifest": mpath.name, "selection_manifest_sha256": sha256_file(mpath),
               "eval_split": "test (200 questions sampled from data/splits/test.jsonl, seed 0, plus their paired edits)",
               "method": method, "resolution": res, "env": env_info(), "arms": {}}
    for arm, qids in man["arms"].items():                      # arms as written by run_selectors.py; checked before any GPU work
        missing = [q for q in qids if q not in pool]
        leak = [q for q in qids if pool[q]["scene_id"] in test_scenes]
        if missing or leak:
            raise SystemExit(f"arm {arm}: {len(missing)} qids missing from pool, {len(leak)} on test scenes")
    if args.dry_run:
        print("dry-run: arms", {a: len(v) for a, v in man["arms"].items()}, "eval items", len(items), "of which paired edits", 2 * len(tiny_edits), "manifest", mpath.name)
        return
    preflight(args)
    login_hf()
    import torch
    for arm, qids in man["arms"].items():
        print(f"=== arm {arm}")
        seed_all(0)                                            # LoRA init, dropout, shuffling
        model, proc, dtype = load_model(args, for_training=True)
        train_rows = [pool[q] for q in qids][: args.tiny_n]
        with model.disable_adapter():                          # same eval subset before training, for reference
            raw0, norm0, _ = predict(model, proc, items, res, args.batch_size, method)
        tr = train_steps(model, proc, dtype, train_rows, res, args.tiny_bs, 1, args.lr, seed=0, method=method)
        raw1, norm1, spe = predict(model, proc, items, res, args.batch_size, method)
        results["arms"][arm] = {"n_train": len(train_rows), "train_qids_sha": __import__("hashlib").sha256(",".join(qids).encode()).hexdigest(),
                                "steps": tr["steps"], "train_seconds": round(tr["seconds"], 1), "dtype": dtype_name(dtype),
                                "lora": lora_report(model),
                                "loss_first10": round(sum(tr["losses"][:10]) / min(10, len(tr["losses"])), 4),
                                "loss_last10": round(sum(tr["losses"][-10:]) / min(10, len(tr["losses"])), 4),
                                "before": plain_accuracy_report(evalrows, norm0, HELD_OUT), "after": plain_accuracy_report(evalrows, norm1, HELD_OUT),
                                "joint_before": joint_report(tiny_edits, norm0, HELD_OUT), "joint_after": joint_report(tiny_edits, norm1, HELD_OUT)}
        jdump(Path(args.out) / "tiny_finetune.json", results)  # saved after EVERY arm: a lost session costs at most one arm
        del model
        torch.cuda.empty_cache()
        print(results["arms"][arm])
    write_run_meta(Path(args.out), vars(args), stage="5-tiny-finetune")


def cmd_check_env(args):
    inst = formats.check_pins(PINS)
    info = env_info()
    info["dtype_would_be"] = None
    try:
        info["dtype_would_be"] = dtype_name(pick_dtype(args))
    except Exception:
        pass
    print(json.dumps(info, indent=2))
    jdump(Path(args.out) / "env_check.json", {"versions": inst, "env": info})


# ----------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check-env", "prepare", "dev-format", "zero-shot", "grid", "cost", "tiny-finetune"])
    ap.add_argument("--out", default=DEFAULT_OUT, help="results directory (Kaggle default: /kaggle/working/stage5_results)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-version-check", action="store_true")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16"], help="auto = fp16 below compute capability 8.0")
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA-style 4-bit base (needs bitsandbytes); use if fp16 OOMs")
    ap.add_argument("--method", choices=["suffix", "choice", "legacy"], default=None, help="default: the dev-format study's choice")
    ap.add_argument("--resolution", type=int, default=None, help="image side, multiple of 28; default: the dev-format study's choice")
    ap.add_argument("--base-resolution", type=int, default=224, help="dev-format: native resolution")
    ap.add_argument("--compare-resolution", type=int, default=336, help="dev-format: larger resolution to compare")
    ap.add_argument("--batch-size", type=int, default=16, help="inference batch size")
    ap.add_argument("--skip-edits", action="store_true", help="prepare: skip the slow edit build (no joint score then)")
    ap.add_argument("--held-out-only", action="store_true")
    ap.add_argument("--max-questions", type=int, default=None, help="smoke test on a random subset")
    ap.add_argument("--skip-shuffle", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--grid-res", type=int, nargs="+", default=[224, 336, 448])
    ap.add_argument("--grid-bs", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--grid-steps", type=int, default=5)
    ap.add_argument("--cost-res", type=int, default=None, help="default: the chosen resolution")
    ap.add_argument("--cost-bs", type=int, default=4)
    ap.add_argument("--sec-per-eval", type=float, default=0.25)
    ap.add_argument("--budget-hours", type=float, default=20.0)
    ap.add_argument("--exists-mode", default="stratified_quota", choices=["stratified_quota", "fixed_shared_base"])
    ap.add_argument("--tiny-n", type=int, default=400)
    ap.add_argument("--tiny-bs", type=int, default=4)
    ap.add_argument("--tiny-eval-n", type=int, default=100, help="held-out AND seen eval questions each")
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    {"check-env": cmd_check_env, "prepare": cmd_prepare, "dev-format": cmd_dev_format, "zero-shot": cmd_zero_shot,
     "grid": cmd_grid, "cost": cmd_cost, "tiny-finetune": cmd_tiny}[args.cmd](args)


if __name__ == "__main__":
    main()
