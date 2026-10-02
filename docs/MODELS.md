# Model artifacts

Only the shipped GGUF is committed, at `model/vitals-v3-Q5_K_M.gguf` (Git LFS).
Everything else lives on the training node's scratch disk. This file is the
manifest: where each artifact is, and the checksum to verify a copy against.

Scratch root: `/ssd_scratch/mahimakopalley`

## Shipped student — v3 GGUF (`gguf_v3/`, built 2026-09-23)

`vitals-v3-Q5_K_M.gguf` is the shipped model (served by `scripts/serve_student.sh`).

| File | Size | SHA-256 |
|---|---|---|
| `vitals-v3-Q5_K_M.gguf` | 401 MB | `9417ddc46efafae056b7fa6af5b341cfe26ab7c7e8b0f33dbc6de3bbada88a1b` |
| `vitals-v3-Q4_K_M.gguf` | 380 MB | `dbe3b058c546f055b62eb74b222e4111aa6be7df07900395c75fc7fec64bdf4a` |
| `vitals-v3-Q8_0.gguf`   | 507 MB | `c302baa33106dd27f5308f069c1e1430935ecd5c5e83e4909315107b02264205` |
| `vitals-v3-f16.gguf`    | 949 MB | `68ef600ebe055a511891663447b3d2e1c0551f871096341c29f0c1d2128afb91` |

Gate and benchmark reports for these files: `docs/results/v3_gguf/`.

## HF checkpoints (`distill_models/`)

| Dir | Size | What |
|---|---|---|
| `v3_vitals_merged/` | 954 MB | v3 student, LoRA merged — source of the v3 GGUFs |
| `v2_merged/`        | 954 MB | v2 student, merged |
| `student/`          | 954 MB | base student before distillation |
| `teacher/`          | 8.1 GB | teacher model |

## Older builds

- `gguf/v2-*.gguf` (2026-09-08): v2 quantisations, superseded by v3.
- Ollama `medgemma-student` (`~/.ollama/models/blobs/sha256-5c0c37b8…`, 2026-07-21):
  matches **neither** v2 nor v3 — it is stale. Re-create it from a v3 GGUF before using it.

## Scripts

All take path overrides via env vars (`GGUF_DIR`, `GGUF`, `LLAMA_SERVER`, `PYTHON`).
`serve_student.sh` defaults to the in-repo GGUF; `run_gate.sh` defaults to the
scratch `gguf_v3/` dir because it gates several quants.

- `scripts/serve_student.sh`: keep the shipped GGUF served on `:8099`.
- `scripts/run_gate.sh <QUANT>`: serve one quant and run `distill/evaluate_gguf.py` against it.
- `scripts/gate_chain_all.sh`: resumable gate run over Q5_K_M and Q8_0.
