# Provenance of vendored files

This repository is a **standalone extract** of the vitals student-model
pipeline. Some files are copied verbatim from the parent agent repository so
that the pipeline can run on its own.

**Source repository:** `MedGemma-Agent`
**Extracted from commit:** see `VENDORED_COMMIT` below
**Extraction date:** 2026-09-24

## Why these files are copied, and why that is a risk

The distillation pipeline deliberately **imports** the agent's prompt and rules
rather than re-implementing them, so that training data and runtime can never
disagree. Splitting this into a separate repository breaks that guarantee: the
copies here can now drift from the originals.

`scripts/check_vendored.py` exists to catch that. Run it before training or
before trusting any evaluation result.

## Vendored files

| File | Role | Must match upstream? |
|---|---|---|
| `agent/prompts.py` | the exact prompt the agent sends at runtime | **YES — critical** |
| `agent/nodes/llm_analyzer.py` | `OUTPUT_SCHEMA`, JSON extraction | **YES — critical** |
| `agent/state.py` | typed state used by the analyzer | yes |
| `guardrails/clinical_rules.py` | NEWS2 + qSOFA -> tier -> action | **YES — critical** |
| `guardrails/thresholds.yaml` | all clinical boundaries | **YES — critical** |
| `guardrails/input_guardrails.py` | plausibility + immediate-critical | yes |
| `guardrails/output_guardrails.py` | the runtime output validator | yes |
| `vitals/schemas.py` | pydantic models, enums | yes |
| `config/settings.py` | settings object the analyzer reads | yes |
| `tests/test_guardrails.py` | the rule/guardrail test suite | yes |

**If a "critical" file drifts, every model trained here becomes invalid** --
it would be trained on a prompt or rubric the agent no longer uses. That exact
failure destroyed three earlier student models; see `docs/STUDENT_MODEL_IDEOLOGY.md`,
trap 4 ("DRIFT").

## Not included

ECG and ABG domains are **not** part of this repository. ECG distillation files
exist upstream but no ECG student has been trained.

No patient data, no EHR store, no model weights and no generated corpora are
included here, by design and by `.gitignore`.
VENDORED_COMMIT=54c99421530867ef23368ec92a51ce261e9d4f13
