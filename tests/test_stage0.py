import json

from siftvqa.config import load_config, ROOT
from siftvqa.runmeta import write_run_meta
from siftvqa.seeding import make_rng


def test_rng_is_reproducible_and_keyed():
    assert make_rng(0, 1).random() == make_rng(0, 1).random()
    assert make_rng(0, 1).random() != make_rng(0, 2).random()


def test_generator_config_loads():
    cfg = load_config(ROOT / "configs" / "generator.yaml")
    assert cfg["min_objects"] == 3 and cfg["max_objects"] == 8
    assert len(cfg["colors"]) == 6 and len(cfg["shapes"]) == 3


def test_run_meta_has_config_and_commit(tmp_path):
    path = write_run_meta(tmp_path, {"a": 1}, stage="test")
    meta = json.loads(path.read_text())
    assert meta["config"] == {"a": 1}
    assert "commit" in meta["git"]
