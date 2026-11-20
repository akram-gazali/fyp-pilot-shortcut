import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location("s4script", ROOT / "colab" / "stage4_feasibility.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m




class FakeTok:
    padding_side = "right"


class FakeProc:
    """Char-level stand-in for the Qwen processor: image tokens are ignored, text -> char codes."""
    tokenizer = FakeTok()

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "<U>" + msgs[0]["content"][1]["text"] + "<A>"

    def __call__(self, text, images, return_tensors, padding):
        ids = [[ord(c) for c in t] for t in text]
        n = max(map(len, ids))
        left = self.tokenizer.padding_side == "left"
        inp = [([0] * (n - len(i)) + i) if left else (i + [0] * (n - len(i))) for i in ids]
        att = [([0] * (n - len(i)) + [1] * len(i)) if left else ([1] * len(i) + [0] * (n - len(i))) for i in ids]
        return {"input_ids": torch.tensor(inp), "attention_mask": torch.tensor(att)}


def test_loss_only_on_answer_tokens():
    from siftvqa.train_utils import build_train_batch
    proc = FakeProc()
    prompts = ["<U>how many?<A>", "<U>is there a red square?<A>"]
    enc = build_train_batch(proc, prompts, ["3", "yes"], [None, None])
    lab = enc["labels"]
    for i, (p, a) in enumerate(zip(prompts, ["3", "yes"])):
        kept = lab[i][lab[i] != -100].tolist()
        assert kept == [ord(c) for c in a + "<|im_end|>"]               # exactly answer + end token
        assert (lab[i][: len(p)] == -100).all()                         # prompt masked
    assert (lab[enc["attention_mask"] == 0] == -100).all()              # padding masked


def test_train_and_eval_use_the_same_prompt_builder(monkeypatch):
    s = load_script()
    calls = []
    monkeypatch.setattr(s, "load_image", lambda path, res: None)
    monkeypatch.setattr(s, "img_path", lambda sid, edited=False: Path(sid))
    real = s.chat_text
    monkeypatch.setattr(s, "chat_text", lambda proc, q, t, m, add_generation_prompt=True: calls.append((q, t, m)) or real(proc, q, t, m))
    rows = [{"scene_id": "s1", "question": "How many red squares?", "template": "count_cs", "answer": "2"}]
    s.make_train_batch(FakeProc(), rows, 224, "cpu", "suffix")
    assert calls == [("How many red squares?", "count_cs", "suffix")]
    assert real(FakeProc(), "How many red squares?", "count_cs", "suffix").endswith("Answer with only a number.<A>")


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.k_proj, self.gate_proj = (torch.nn.Linear(8, 8) for _ in range(3))

    def forward(self, x):
        return self.gate_proj(self.q_proj(x) + self.k_proj(x))


class TinyVLM(torch.nn.Module):
    """Language layers (LoRA targets) plus a 'visual' tower with the same layer names (must NOT get LoRA)."""
    def __init__(self):
        super().__init__()
        self.visual = Block()
        self.language = torch.nn.ModuleList([Block()])

    def forward(self, x):
        return self.language[0](self.visual(x))


def make_lora_model():
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    s = load_script()
    base = TinyVLM().half()                                              # fp16 base, as on the T4
    cfg = LoraConfig(r=4, lora_alpha=8, target_modules=s.LORA_TARGETS)
    return get_peft_model(base, cfg, autocast_adapter_dtype=True)


def test_lora_adapters_fp32_on_fp16_base_and_scaler_step():
    from siftvqa.train_utils import lora_report
    model = make_lora_model()
    rep = lora_report(model)                                             # raises if not fp32 / in vision tower
    print("LoRA check (dry run):", rep)
    assert rep["lora_in_vision_tower"] == 0 and rep["n_trainable_params"] > 0
    assert all(p.dtype == torch.float16 for n, p in model.named_parameters() if not p.requires_grad)   # base stays fp16 and frozen
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    scaler = torch.amp.GradScaler("cpu", enabled=True)                   # fp16 path: loss scaling on
    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    out = model(torch.randn(2, 8).half())
    scaler.scale(out.float().pow(2).mean()).backward()
    scaler.step(opt)                                                     # would raise on fp16 gradients
    scaler.update()
    after = [p for p in model.parameters() if p.requires_grad]
    assert any(not torch.equal(a, b) for a, b in zip(before, after))     # adapters actually moved


def test_lora_report_rejects_fp16_adapters():
    from siftvqa.train_utils import lora_report
    model = make_lora_model()
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.half()
    with pytest.raises(RuntimeError, match="not fp32"):
        lora_report(model)


def test_lora_skips_vision_tower():
    model = make_lora_model()
    names = [n for n, _ in model.named_parameters() if "lora_" in n]
    assert names and not any("visual" in n for n in names)


def test_seed_all_reproducible():
    from siftvqa.train_utils import seed_all
    seed_all(0)
    a = torch.randn(3)
    seed_all(0)
    assert torch.equal(a, torch.randn(3))
