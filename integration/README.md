# PipeGen integration area

This directory is intentionally separate from the main runtime code so the official CraftifAI PipeGen workflow can be integrated without inventing unsupported tooling or claiming a working pipeline.

Planned integration points:
- `pipegen_integration.py` with adapter patterns and validation helpers
- model export and engine export steps for TensorRT
- DeepStream deployment configuration checks
- documentation and runbook notes for official toolchain usage

Do not assume the environment has a valid PipeGen installation or GPU runtime. This project remains functional on a standard Windows laptop without those integrations.
