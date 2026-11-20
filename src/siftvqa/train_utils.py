"""Training helpers kept separate from the GPU script so they can be unit-tested on CPU with a tiny fake model."""
import random


def seed_all(seed):
    """Seed python, numpy (if present) and torch (CPU and CUDA): LoRA init, dropout and shuffling all follow it."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_batch(proc, prompts, answers, images):
    """Chat prompt + answer + <|im_end|>; loss ONLY on the answer tokens.
    Prompt tokens (including the image tokens) and padding are set to -100. Right padding, so the prompt occupies
    the first `prompt_len` positions of every row."""
    proc.tokenizer.padding_side = "right"
    full = [p + a + "<|im_end|>" for p, a in zip(prompts, answers)]
    enc_p = proc(text=prompts, images=images, return_tensors="pt", padding=True)
    enc = proc(text=full, images=images, return_tensors="pt", padding=True)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    for i, n in enumerate(enc_p["attention_mask"].sum(1).tolist()):
        labels[i, :n] = -100
    enc["labels"] = labels
    return enc


def lora_report(model):
    """Check the adapter setup and return a summary (printed by the dry run and stored in tiny_finetune.json).
    - every trainable parameter is fp32 (needed so the fp16 GradScaler can unscale gradients)
    - no LoRA module sits in the vision tower
    - something is trainable, and the base weights are frozen"""
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not trainable:
        raise RuntimeError("no trainable parameters")
    bad_dtype = sorted({str(p.dtype) for p in trainable.values() if str(p.dtype) != "torch.float32"})
    if bad_dtype:
        raise RuntimeError(f"trainable parameters are not fp32: {bad_dtype}")
    lora_names = [n for n in trainable if "lora_" in n]
    if len(lora_names) != len(trainable):
        raise RuntimeError("trainable parameters that are not LoRA weights: "
                           + str([n for n in trainable if "lora_" not in n][:5]))
    in_vision = [n for n in lora_names if "visual" in n]
    if in_vision:
        raise RuntimeError(f"LoRA modules found in the vision tower: {in_vision[:3]}")
    return {"n_trainable_tensors": len(trainable), "n_trainable_params": sum(p.numel() for p in trainable.values()),
            "trainable_dtypes": ["torch.float32"], "lora_in_vision_tower": 0}
