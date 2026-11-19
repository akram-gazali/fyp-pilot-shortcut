# SIFT-VQA pilot

Pilot pipeline for shortcut-aware example selection in VQA on synthetic 2D scenes.
Pilot defaults live in `configs/` (not final until the protocol lock). Every run writes
`run_meta.json` (config + git commit) next to its outputs.

## Setup
```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest
```
(Reproduction commands for each stage are added as the stages land.)
