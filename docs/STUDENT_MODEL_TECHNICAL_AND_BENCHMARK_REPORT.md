# Student Model — Technical and Benchmark Report

*Audit of the CURRENT implementation. All performance and quality numbers below
were measured by running the current code against the currently-served model on
2026-09-24. No figures are carried over from earlier reports; where an earlier
figure is mentioned it is explicitly labelled HISTORICAL.*

Legend used throughout:
**[MEASURED]** produced by running something today ·
**[CODE]** read from the current source ·
**[OBSERVATION]** interpretation or limitation ·
**NOT MEASURED** with the reason stated.

---

## 1. Overview

**[CODE]** The current student model is a **Qwen2.5-0.5B** causal LM,
LoRA-fine-tuned on answers from a **MedGemma-4B** teacher, merged into the base
weights, converted to GGUF and quantized. The deployed artefact is
`vitals-v3-Q5_K_M.gguf`.

**Parameter count [MEASURED]** — counted directly from the merged safetensors:
**494,032,768 parameters (494.0M)** across 290 tensors.

**What it is intended to do [CODE]** — it writes the *natural-language
explanation* that accompanies a vital-signs alert. It does **not** decide the
patient's risk. `guardrails/clinical_rules.py` computes the tier from NEWS2 and
qSOFA before the model is called, and `agent/nodes/alert_router.py` recomputes
the recommended action from the rules afterwards.

**Task / input / output currently supported [CODE]**

| | |
|---|---|
| Domain | **vitals only** — systolic BP, diastolic BP, heart rate, SpO2 |
| Input | the prompt built by `agent/prompts.py::build_user_prompt()` — the four vitals, the NEWS2 breakdown, the qSOFA breakdown, the rule's alert level, and EHR context |
| Output | one JSON object with 8 required fields (`risk_level, confidence, reasoning, sepsis_risk_flag, contributing_factors, recommended_action, limitations, disclaimer`) |

ABG and ECG are **not** supported by this model. An ECG domain exists in the
repository as generated-but-unused training data (1,237 rows); no ECG student
has been trained.

**Distilled or fine-tuned, and how [CODE]** — knowledge distillation. The
teacher answered synthetic cases; a gate (`distill/gate.py`) discarded any
answer contradicting the deterministic rules; the survivors trained a LoRA
adapter which was then merged.

**Intended deployment environment [CODE/OBSERVATION]** — an edge box beside the
patient (`STUDENT_MODEL_README.md` names a Jetson Nano). Currently served on a
cluster node via `llama-server`.

**Requirements [MEASURED]**

| Requirement | Needed? |
|---|---|
| GPU | **No.** The current llama.cpp build has no CUDA; inference is CPU-only |
| Internet | **No.** Model file is local; server binds to 127.0.0.1 |
| Python | **Not for inference.** `llama-server` is a C++ binary. Python is needed for the agent, the tests and the benchmark |
| External services | **No.** Ollama is *not* used (see §2) |

**Role in the clinical pipeline [CODE]** — step 4 of a 7-node LangGraph
pipeline. It is the only node containing a model.

---

## 2. Model Architecture

| Item | Value | Source |
|---|---|---|
| **Base model** | Qwen2.5-0.5B (`Qwen2ForCausalLM`), hidden 896, 24 layers, 14 heads, 2 KV heads, intermediate 4864, vocab 151,936, tied embeddings | **[MEASURED]** `config.json` |
| **Parameter count** | **494,032,768** | **[MEASURED]** safetensors |
| **Fine-tuning method** | LoRA, merged into base | **[CODE]** |
| **LoRA configuration** | r=**16**, alpha=**32**, dropout=**0.05**, target modules = `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`, task `CAUSAL_LM` | **[MEASURED]** `adapter_config.json` |
| **Training checkpoint** | `distill_runs/v3_vitals/adapter`, merged to `distill_models/v3_vitals_merged` on 2026-09-23T11:07:52Z | **[MEASURED]** `provenance.json` |
| **Checkpoint selection** | **Last epoch — no best-checkpoint selection.** `save_strategy="epoch"`, `save_total_limit=2`, and `train.py` saves the final model after `trainer.train()` returns. The in-run eval loss was logged but never used to select. | **[CODE]** `distill/train.py` |
| **Training objective** | Causal LM, **answer-only** — the prompt is masked with `-100` | **[CODE]** `run_config.json`: `"loss": "answer_only_prompt_masked"` |
| **Training data** | 1,255 rows, sha256 prefix `f0dd969ff89c8d8d`, 3 epochs, lr 2e-4, batch 2 × grad-accum 8, max_len 1024, seed 20260922, 2,687 s | **[MEASURED]** `run_config.json` |
| **Held-out** | 103 rows in region `wide_pulse_pressure`, **`leakage_rows: 0`** | **[MEASURED]** |
| **Input format** | chat template (system + user), prompt ~**491–492 tokens** measured | **[MEASURED]** |
| **Output format** | JSON, schema-constrained at serve time via `response_format: json_schema` | **[CODE]** |
| **Inference framework** | **llama.cpp `llama-server`**, version `0.4.0-dev (build 1, commit f3f1a8f)`, GNU 11.4.0, Linux x86_64 | **[MEASURED]** |
| **Quantization** | **Q5_K_M** (shipped). Q4_K_M and Q8_0 also built | **[MEASURED]** |
| **Model file size** | **401 MB** Q5_K_M · 379 MB Q4_K_M · 506 MB Q8_0 · 948 MB f16 | **[MEASURED]** |
| **Deployment target** | edge device; currently a cluster node | **[CODE/OBSERVATION]** |
| **Runtime dependencies** | `llama-server` binary + the `.gguf` file. Nothing else | **[MEASURED]** |

### Current inference pipeline **[CODE]**

```
rules compute tier  ->  build_user_prompt()  ->  POST /v1/chat/completions
(schema-constrained, temperature 0.0)  ->  JSON  ->  output guardrail  ->  alert
```

### ⚠️ Documentation/implementation inconsistencies found

1. **`config/settings.py` still sets `OLLAMA_MODEL = "medgemma"`** — the agent
   code points at the *teacher* via Ollama, not at this student. **The student
   is NOT wired into the agent.** It is served standalone.
2. **Ollama is not installed.** `~/bin/ollama` is a 9-byte ASCII file containing
   `Not Found` (a failed download). `llama-server` is used instead.
3. **`STUDENT_MODEL_README.md` is stale** — it states the distillation scripts
   do not exist; they do (`distill/`).

---

## 3. Safety Architecture

All mechanisms below were confirmed by reading the current source.

| Mechanism | File | What it actually does **[CODE]** |
|---|---|---|
| **Deterministic clinical rules** | `guardrails/clinical_rules.py` | NEWS2 per-parameter scores + qSOFA → tier. `NEWS2 0-2 NORMAL, 3-4 MEDIUM, 5-6 HIGH, >=7 CRITICAL`; also CRITICAL if any single score = 3 **or** qSOFA high-risk |
| **Thresholds under document control** | `guardrails/thresholds.yaml` | All boundaries in one version-controlled file, header marked ISO 13485 |
| **Input schema + plausibility** | `guardrails/input_guardrails.py` | Rejects impossible values and `sbp < dbp` before anything else runs |
| **CRITICAL bypasses the model entirely** | `agent/graph.py` | `_route_after_risk_scorer`: `if immediate_critical: return "alert_router"` — **the LLM is never called for CRITICAL patients** |
| **Output validation (4 checks)** | `guardrails/output_guardrails.py::validate_llm_output` | 1 schema (4 required fields) · 2 **grounding** — reasoning must cite ≥1 real vital · 3 **prohibited content** — drugs/doses/treatments · 4 **contradiction** — model tier must be within **1 tier** of the rule tier. All four always evaluated; no early return |
| **Action is never taken from the model** | `agent/nodes/alert_router.py` | `final_recommended_action = determine_recommended_action(final_level)`. A repo-wide grep shows the LLM's `recommended_action` is **read nowhere** |
| **CRITICAL never downgraded** | `agent/nodes/alert_router.py` | `if rule_level == CRITICAL: final_level = CRITICAL` |
| **Escalation permitted** | `agent/nodes/alert_router.py` | `final_level = max(rule_level, llm_level)` — the model **can** raise a tier but never lower one |
| **Schema-constrained decoding** | serve-time | `response_format: {"type":"json_schema"}` forces valid JSON |
| **Fallback when the model fails** | `agent/nodes/llm_analyzer.py`, `output_validator.py` | On any exception → `llm_available: False`; the validator returns a null-LLM state and the alert proceeds on rules alone |
| **Retry behaviour** | — | **None.** A failed LLM call is not retried; it falls back **[CODE]** |
| **Audit trail** | `agent/nodes/audit_logger.py` | SHA-256 hash-chained log |

**[OBSERVATION]** What the guardrail does **not** catch: a **one-tier**
disagreement passes validation, and `alert_router` then takes the maximum — so
a model saying HIGH where rules say MEDIUM **will change the alert to HIGH**.
`contributing_factors` and `limitations` are passed through unvalidated.

### Testing **[MEASURED]**

Command: `python -m pytest tests/ -q`

```
51 passed in 0.44s
```

| | |
|---|---|
| Tests collected | **51** |
| Passed | **51** |
| Failed | **0** |
| Skipped | **0** |
| Errors | **0** |

26 `def test_` functions covering NEWS2 scoring, qSOFA, alert-level mapping,
input guardrails (impossible HR, `sbp < dbp`, immediate-critical) and output
guardrails (prohibited content, ungrounded reasoning, large contradiction,
inflected drug forms).

**Crash/error recovery [CODE, not fault-injected]** — the fallback path exists
and is unit-tested at the validator level (`test_output_*`). **NOT MEASURED:**
no fault-injection test was run against a live server (e.g. killing the server
mid-request). The benchmark harness did surface real transport errors during
earlier work and the client reported them correctly rather than silently
scoring them as model failures.

**Known failing cases: none in the test suite.**

---

## 4. Current Task Evaluation — Vitals

**[CODE]** The current student supports **vitals only**. There is no ECG or ABG
path through this model. ECG exists as a separate, untrained data set.

| Aspect | Finding |
|---|---|
| **Input format** | System prompt + user prompt from `agent/prompts.py`. Contains SBP, DBP, HR, SpO2, the NEWS2 per-parameter breakdown, qSOFA flags, `Rule Alert: <tier>`, EHR context (always `"No EHR data available."` in training) |
| **Expected output** | 8-field JSON |
| **Rubric/rule source** | `guardrails/clinical_rules.py` + `guardrails/thresholds.yaml` |
| **Model output behaviour** **[MEASURED]** | 60/60 valid JSON, 60/60 all fields present, tier copied correctly 60/60 |
| **Parser behaviour** **[MEASURED]** | `_extract_json` parsed 60/60 with no fallback to regex extraction needed |
| **Guardrail behaviour** **[MEASURED]** | No contradiction triggered — the model never disagreed with the rule tier in 60 balanced cases |
| **Failure modes** **[MEASURED]** | 1 of 60 — see §6 |
| **Hallucinations/fabrications** **[MEASURED]** | 0/60 fabricated numbers |
| **Unsupported terminology** | **NOT MEASURED** — no clinical-terminology checker exists in the repo, and no clinician reviewed the text |
| **Missing fields** **[MEASURED]** | 0/60 |
| **Invalid outputs** **[MEASURED]** | 0/60 |
| **Token-limit behaviour** **[MEASURED]** | `max_tokens=500`; **0/60 hit the cap**; longest completion observed **366 tokens** |
| **Numerical fidelity** **[MEASURED]** | 59/60 cited both the systolic BP and heart rate verbatim; 0/60 invented numbers |
| **Format compliance** **[MEASURED]** | 60/60 |
| **Agreement with the deterministic rubric** **[MEASURED]** | tier 60/60, action 60/60 |

### Vitals supported

HR ✅ · SBP ✅ · DBP ✅ · SpO2 ✅ · Respiratory rate ❌ · Temperature ❌ ·
Consciousness ❌ (the last three are not among the four monitored vitals).

---

## 5. Benchmark Results

Primary table = the **tier-balanced** run (15 cases per tier). See methodology
for why the unbalanced run is reported only as a secondary result.

| **Metric** | **Result** |
| --- | --- |
| **Latency** | **6.97 s median** (mean 7.10, min 4.62, p95 9.49, max 10.06) |
| **TTFT** | **0.42 s median** (min 0.38, p95 0.44, max 0.46) |
| **Generation speed** | **38.75 tok/s median** (mean 38.9, min 30.4, max 43.1) |
| **RAM usage** | **1,193,288 kB peak RSS (1.14 GiB)**; 562 MB idle after load |
| **Power draw** | **NOT MEASURED** — RAPL `energy_uj` is mode `-r--------` owned by root; `/dev/ipmi0` does not exist; inference is CPU-only so GPU power is irrelevant |
| **Energy per case** | **NOT MEASURED** — follows from power being unavailable |
| **Internal consistency** | **60/60 (100%)** — tier and action both match the rules |
| **Numeric fidelity** | **59/60 (98.3%)** cited both key vitals; **60/60 (100%)** free of invented numbers |
| **Format compliance** | **60/60 (100%)** |
| **Fabrication-free responses** | **60/60 (100%)** |
| **Rubric alignment** | **60/60 (100%)** tier · **60/60 (100%)** action |
| **Under-triage count** | **0** |
| **Under-triage rate** | **0.0%** |
| **Cases evaluated** | **60** (15 NORMAL, 15 MEDIUM, 15 HIGH, 15 CRITICAL) |

### Per-tier breakdown **[MEASURED]**

| tier | n | json | fields | tier | action | fab-free | cites both | under | over |
|---|---|---|---|---|---|---|---|---|---|
| NORMAL | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 0 | 0 |
| MEDIUM | 15 | 15 | 15 | 15 | 15 | 15 | **14** | 0 | 0 |
| HIGH | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 0 | 0 |
| CRITICAL | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 0 | 0 |

| tier | median latency | median completion tokens |
|---|---|---|
| NORMAL | 6.37 s | 224 |
| MEDIUM | 7.36 s | 277 |
| HIGH | **8.35 s** | **303** |
| CRITICAL | 7.09 s | 261 |

### Secondary run — random sampling **[MEASURED]**

Latency median **7.03 s**, TTFT **0.42 s**, generation **39.54 tok/s**, RSS peak
**968,132 kB**, quality 60/60 on every metric, under-triage 0.

**[OBSERVATION] Its tier mix was `NORMAL 6, MEDIUM 2, HIGH 1, CRITICAL 51`.**
Random draws over the physiological space are ~71% CRITICAL, and CRITICAL
**bypasses the model in production**. 51 of those 60 perfect scores therefore
describe cases the model never sees at runtime, and only 2 were MEDIUM. This run
is reported for completeness but should not be used as the quality figure.

### Benchmark methodology

| | |
|---|---|
| **Cases** | 60 benchmark + 3 warm-up (balanced run); 60 + 3 (random run) |
| **Repeated runs** | **1 run per configuration.** Each case measured once. **No repeat-and-average** — treat the medians as single-run estimates |
| **Hardware** | Intel Xeon E5-2640 v4 @ 2.40 GHz, 2 sockets × 10 cores × 2 threads = **40 logical CPUs**; 125 GiB RAM; NVIDIA RTX 2080 Ti present but **unused** |
| **CPU/GPU configuration** | `llama-server` threadpool **n_threads = 20**, affinity `0-39`. **NOTE:** the shell is inside a SLURM allocation restricted to CPUs `0,20`; the server was started with `setsid` and escaped that cgroup. It therefore used more CPU than the scheduler allocated |
| **Node load** | quiet — `load average: 0.32` before the run |
| **RAM measurement** | `VmRSS` of the server pid sampled every 0.5 s from `/proc/<pid>/status` during the run |
| **CPU measurement** | `utime+stime` deltas from `/proc/<pid>/stat`; median **1916.5%** (~19 cores) |
| **Power measurement** | **none available** (see table) |
| **Warm-up** | 3 full requests before timing begins |
| **Model loading time** | **excluded** — the server was already resident |
| **Prompt processing** | **included** — TTFT covers prompt ingestion; end-to-end latency includes it |
| **Generation time** | **included** |
| **Token counts** | from the server's `usage` field via `stream_options.include_usage`: prompt ~491, completion median 260 |
| **Sampling parameters** | `temperature = 0.0`, `max_tokens = 500`, `stream = true`, `response_format = json_schema` |
| **Context length** | `-c 8192` total, `--parallel 4` → **n_ctx_slot = 2048** per slot |
| **Quantization** | Q5_K_M |
| **Runtime** | llama.cpp `llama-server` 0.4.0-dev, commit f3f1a8f |
| **Statistics** | median reported as primary; mean/min/max/p95 in the JSON. Requests issued **sequentially**, one at a time |
| **Raw output** | `gguf_v3/benchmark_v3_balanced.json`, `gguf_v3/benchmark_v3.json` |

**Exact commands**

```bash
python -m pytest tests/ -q

python distill/benchmark_student.py --n 60 --warmup 3 --balanced --seed 4242 \
    --out /ssd_scratch/mahimakopalley/gguf_v3/benchmark_v3_balanced.json

python distill/benchmark_student.py --n 60 --warmup 3 \
    --out /ssd_scratch/mahimakopalley/gguf_v3/benchmark_v3.json
```

---

## 6. Detailed Performance Analysis

### Inference performance **[MEASURED]**

**End-to-end latency** median **6.97 s**, tightly grouped (p95 9.49 s, max
10.06 s). Latency tracks output length: HIGH cases are slowest (8.35 s, 303
tokens) and NORMAL fastest (6.37 s, 224 tokens), which is expected — the model
writes more when there is more to explain.

**TTFT** median **0.42 s** and remarkably stable (0.38–0.46 s across 60 cases).
With a ~491-token prompt this reflects prompt ingestion plus first-token
generation. A dashboard streaming the response would show text within half a
second.

**Generation throughput** median **38.75 tok/s** on CPU.

**Model loading time [MEASURED, excluded from the table]** — server start to
`/health` responding `ok` takes roughly 35 s in our start script, dominated by
reading the 401 MB file. Relevant to restarts, not per-request latency.

**CPU utilisation [MEASURED]** median **1916.5%** ≈ 19 of 20 threads.
**[OBSERVATION]** This exceeds the 2 logical CPUs SLURM allocated to this
session; the server escaped the cgroup via `setsid`. On a correctly-confined
2-CPU allocation, latency would be substantially worse — a HISTORICAL
measurement under heavy contention showed 30–75 s per request.

**RAM [MEASURED]** 562 MB idle after load → **1.14 GiB peak** under load. The
growth is KV cache across 4 slots × 2048 tokens.

**GPU utilisation — NOT MEASURED** (CPU-only build).

**Power / energy — NOT MEASURED** (no readable counter).

### Output quality **[MEASURED, balanced run]**

| | |
|---|---|
| Internal consistency | 60/60 |
| Numeric fidelity | 59/60 cited both key vitals; 60/60 no invented numbers |
| Format compliance | 60/60 |
| Fabrication rate | **0%** |
| Rubric alignment | 60/60 tier, 60/60 action |
| Under-triage | **0/60** |
| Over-triage | **0/60** |

### Failure analysis

**Only one failure occurred in 60 balanced cases.**

#### Failure 1 — incomplete value citation

1. **What happened:** case `SBP 132, DBP 100, HR 113, SpO2 99` (NEWS2 4, qSOFA
   1, rule tier MEDIUM). The reasoning did not quote both the systolic BP and
   the heart rate verbatim.
2. **Expected:** reasoning cites the actual measured values.
3. **Actual:** JSON valid, all 8 fields present, tier MEDIUM correct, action
   `clinician_review` correct — only the citation was incomplete. 185 tokens,
   5.08 s.
4. **Frequency:** **1/60 (1.7%)**, the only one, in the MEDIUM tier.
5. **Corrected by the safety layer?** **[OBSERVATION] Probably not, and it does
   not need to be.** The runtime `grounding` check in
   `guardrails/output_guardrails.py` requires **at least one** vital to be
   cited; our benchmark used a stricter rule (**both** SBP and HR). So this
   response would likely pass production validation. Our metric is deliberately
   harsher than the deployed guardrail.

#### Failure categories that did NOT occur

Zero occurrences of: invalid JSON, missing fields, tier contradiction, action
mismatch, fabricated numbers, token-cap truncation (max observed 366 of 500),
under-triage, over-triage.

#### HISTORICAL contrast (not current)

The **Q4_K_M** build failed its gate suite on 3 MEDIUM cases with
`action_mismatch: urgent_review != clinician_review`. **Q5_K_M — the build
measured here — shows none of these.**

---

## 7. Interpretation

### Strengths **[all MEASURED today]**

1. **Perfect format compliance** — 60/60 valid, complete JSON; schema-constrained decoding makes malformed output structurally difficult.
2. **Perfect rubric alignment** — 60/60 tier, 60/60 action, across all four tiers evenly sampled.
3. **Zero under-triage and zero over-triage** in 60 balanced cases.
4. **Zero fabricated numbers.**
5. **Predictable latency** — median 6.97 s, p95 9.49 s, no long tail; TTFT stable at 0.42 s.
6. **No token-cap truncation** — max 366 of 500.
7. **Small and self-contained** — 401 MB, no GPU, no internet, no Python at inference.
8. **Test suite green** — 51/51.

### Limitations **[demonstrated by current evidence]**

1. **The tier verdict is inside the prompt.** `Rule Alert: <tier>` is given to the model, so "100% tier agreement" is substantially a **copying** task. **[CODE]**
2. **Single run, no repeats.** All timings are one measurement per case. No confidence intervals.
3. **Measured outside its CPU allocation.** 20 threads on 40 CPUs, while SLURM allocated 2. Real edge performance will be worse; a Jetson Nano far worse.
4. **No clinician has reviewed any output.** Every metric is consistency, not medical correctness.
5. **60 cases.** At n=15 per tier, a 5% failure rate could easily show as zero.
6. **MEDIUM remains the weak tier** — the single failure was MEDIUM, and the HISTORICAL Q4_K_M failures were all MEDIUM. Training had 263 MEDIUM rows vs 398 CRITICAL.
7. **Not wired into the agent.** `OLLAMA_MODEL = "medgemma"` still points at the teacher.
8. **No EHR grounding** — trained with `"No EHR data available."` throughout.
9. **Power and energy unmeasurable** on this host.
10. **Synthetic evaluation only** — no real patient data at any stage.

### Safety implications

**What the deterministic rules guarantee [CODE]** — the tier and the
recommended action. Both are computed before and after the model respectively;
neither depends on it. CRITICAL patients **never reach the model at all**.

**What the student model does [MEASURED]** — writes an explanation that, in 60
balanced cases, was correctly formatted, numerically faithful and consistent
with the rules.

**What the guardrail guarantees [CODE]** — schema completeness, ≥1 grounded
value, absence of drug/dose content, and that the model's tier is within one
tier of the rules'.

**What remains unreliable [OBSERVATION]**
- The *clinical quality* of the prose — never assessed.
- Behaviour on a genuinely CPU-constrained device — never measured.
- Behaviour at scale — n=60, single run.
- A **one-tier over-escalation would pass** the guardrail and **raise** the alert, since `alert_router` takes the maximum.

**This model is NOT clinically validated.** The repository contains no
clinician review, no real patient data and no prospective study.

---

## 8. Recommended Role in System

The current implementation is:

```text
Patient Vitals Input
      ↓
Input Validator  ──(implausible)──> reject + audit
      ↓
      ├──(rule tier == CRITICAL)──> Alert Router ──> CRITICAL alert
      │                              (STUDENT MODEL SKIPPED ENTIRELY)
      ↓
EHR Retriever
      ↓
Deterministic Clinical Rules  (NEWS2 + qSOFA)     <== SOURCE OF TRUTH
      ↓
Objective Tier: NORMAL / MEDIUM / HIGH / CRITICAL <== CLASSIFICATION HAPPENS HERE
      ↓
STUDENT MODEL  (Qwen2.5-0.5B, Q5_K_M)             <== LANGUAGE ONLY
      ↓
Natural-Language Explanation (8-field JSON)
      ↓
Output Guardrail  (schema · grounding · prohibited · contradiction)
      ↓
Alert Router   final_action = f(rules);  final_level = max(rule, model)
      ↓
Audit Logger (SHA-256 hash chain)
      ↓
Alert on Dashboard
```

**[CODE]** Note this diagram differs from the generic one in the task: a
CRITICAL patient **bypasses the model**, and the action is recomputed from the
rules after the model runs.

| Question | Answer |
|---|---|
| **Source of truth** | `guardrails/clinical_rules.py` |
| **Performs classification** | The rules — never the model |
| **Generates language** | The student model, only |
| **Where safety checks occur** | Before (input guardrail), around (CRITICAL bypass), after (output guardrail), and at routing (`max`, CRITICAL never downgraded) |
| **If the student fails** | `llm_available: False`; the alert proceeds on rules alone with no explanation text. Measured today: no failures in 60 cases; the fallback path is unit-tested but **not fault-injected against a live server** |
| **Can it operate independently?** | **No — and it should not.** It never sees CRITICAL cases, its action output is discarded, and its tier is bounded to ±1 of the rules' |
| **Must NOT be delegated to it** | Deciding risk level · choosing the recommended action · any decision with no rule-based fallback · anything requiring clinical correctness of prose, which is unvalidated |

---

## 9. Reproducibility

```bash
# --- environment ---
VENV=/home2/mahimakopalley/projects/.venv/bin/python
REPO=/home2/mahimakopalley/projects/MedGemma-Agent
cd "$REPO"

# --- model location ---
# /ssd_scratch/mahimakopalley/gguf_v3/vitals-v3-Q5_K_M.gguf   (401 MB)
# provenance: /ssd_scratch/mahimakopalley/distill_models/v3_vitals_merged/provenance.json

# --- serve the model ---
# NOTE: -c is the TOTAL context, split across --parallel slots.
# -c 2048 --parallel 4 gives only 512 tokens per slot and SILENTLY TRUNCATES.
/ssd_scratch/mahimakopalley/llama.cpp/build/bin/llama-server \
    -m /ssd_scratch/mahimakopalley/gguf_v3/vitals-v3-Q5_K_M.gguf \
    --host 127.0.0.1 --port 8099 -c 8192 --parallel 4

curl -s http://127.0.0.1:8099/health          # {"status":"ok"}

# --- tests ---
$VENV -m pytest tests/ -q

# --- benchmark (primary, tier-balanced) ---
$VENV distill/benchmark_student.py --n 60 --warmup 3 --balanced --seed 4242 \
    --out /ssd_scratch/mahimakopalley/gguf_v3/benchmark_v3_balanced.json

# --- benchmark (secondary, random sampling) ---
$VENV distill/benchmark_student.py --n 60 --warmup 3 \
    --out /ssd_scratch/mahimakopalley/gguf_v3/benchmark_v3.json

# --- functional checks ---
$VENV distill/test_student.py smoke
$VENV distill/test_student.py stress
$VENV distill/test_student.py compare --n 25
$VENV distill/test_student.py gates

# --- hardware monitoring (what the harness does internally) ---
SRV=$(pgrep -f llama-server | head -1)
watch -n1 "grep VmRSS /proc/$SRV/status; grep -E 'Cpus_allowed_list|Threads' /proc/$SRV/status"
taskset -pc $SRV
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv   # GPU UNUSED here

# --- power measurement: NOT AVAILABLE on this host ---
cat /sys/class/powercap/intel-rapl:0/energy_uj    # Permission denied (root-only)
ipmitool dcmi power reading                        # /dev/ipmi0 does not exist
```

No environment variables are required for inference. The agent (not used in
this benchmark) reads `.env` per `config/settings.py`.

**Output locations**

```
/ssd_scratch/mahimakopalley/gguf_v3/benchmark_v3_balanced.json   primary benchmark
/ssd_scratch/mahimakopalley/gguf_v3/benchmark_v3.json            secondary benchmark
/ssd_scratch/mahimakopalley/gguf_v3/eval_Q5_K_M.json             gate suite (shipped build)
/ssd_scratch/mahimakopalley/gguf_v3/eval_Q4_K_M.json             gate suite (rejected build)
/ssd_scratch/mahimakopalley/gguf_v3/eval_Q8_0.json               gate suite (alternative)
```

---

## 10. Files and Evidence Inspected

| File | Purpose | Relevant finding |
|---|---|---|
| `distill_models/student/config.json` | base model config | Qwen2, hidden 896, 24 layers, vocab 151,936, tied embeddings |
| `distill_models/v3_vitals_merged/*.safetensors` | merged weights | **494,032,768 parameters** counted directly |
| `distill_models/v3_vitals_merged/provenance.json` | provenance | data sha256 `f0dd969ff89c8d8d`, `leakage_rows: 0`, merged 2026-09-23 |
| `distill_runs/v3_vitals/adapter/adapter_config.json` | LoRA config | r=16, alpha=32, dropout=0.05, 7 target modules |
| `distill_runs/v3_vitals/run_config.json` | training config | 1,255 rows, 3 epochs, lr 2e-4, answer-only loss, 2,687 s |
| `guardrails/clinical_rules.py` | **source of truth** | NEWS2+qSOFA → tier; tier → action |
| `guardrails/thresholds.yaml` | thresholds | all boundaries; ISO 13485 document-control header |
| `guardrails/input_guardrails.py` | input safety | plausibility + `immediate_critical` |
| `guardrails/output_guardrails.py` | output safety | 4 checks; contradiction tolerance is **±1 tier** |
| `agent/graph.py` | orchestration | **CRITICAL bypasses the LLM** |
| `agent/nodes/alert_router.py` | final decision | action from rules; level = `max(rule, model)` |
| `agent/nodes/llm_analyzer.py` | model call | fallback on exception, **no retry** |
| `agent/prompts.py` | prompt | contains `Rule Alert: <tier>` — the tier is given to the model |
| `config/settings.py` | config | **`OLLAMA_MODEL = "medgemma"` — student not wired in** |
| `tests/test_guardrails.py` | test suite | 26 test functions → **51 passed** |
| `distill/gate.py` | teacher-data filter | rejects tier/action contradictions, fabrication, drugs |
| `distill/evaluate_gguf.py` | gate suite | sets scored separately, verdict = AND |
| `distill/benchmark_student.py` | **benchmark (written for this audit)** | TTFT/latency/throughput/RSS/CPU + quality in one run |
| `distill/test_student.py` | functional tests | smoke / ask / compare / stress / gates |
| `distill_data/vitals_v3/accepted.jsonl` | training data | 1,497 accepted rows |
| `gguf_v3/vitals-v3-Q5_K_M.gguf` | **deployed model** | 401 MB |
| `gguf_v3/eval_Q5_K_M.json` | gate report | PASS, 196 cases, tier agreement 100% |
| `/tmp/llama_server.log` | server log | `n_threads = 20`, `n_ctx_slot = 2048` |
| `STUDENT_MODEL_README.md` | old doc | **stale** — claims distillation scripts do not exist |

---

## 11. Conclusion

1. **What is the current student model?** A Qwen2.5-0.5B causal LM, LoRA-distilled from MedGemma-4B, merged and quantized to Q5_K_M, serving as the *explanation writer* for vitals alerts. It does not classify.

2. **How large is it?** **494,032,768 parameters**; **401 MB** on disk as Q5_K_M.

3. **How fast is it?** **6.97 s median end-to-end**, **0.42 s median TTFT**, **38.75 tok/s** — measured on a quiet 40-CPU Xeon node with 20 threads, prompt ~491 tokens, completion median 260 tokens.

4. **How much memory?** **562 MB** resident after load; **1.14 GiB peak RSS** under load with 4 slots × 2048-token context.

5. **Power/energy cost?** **NOT MEASURED.** RAPL is root-only and `/dev/ipmi0` is absent; inference is CPU-only so GPU counters do not apply.

6. **How well does it match the deterministic reference?** **60/60 (100%)** on both tier and recommended action across 15 cases per tier. Independently, the gate suite on this same build was 100% on tier across 196 cases. **Caveat:** the rule's tier is present in the prompt, so tier agreement is partly a copying task; the action mapping is not in the prompt and is the more informative of the two.

7. **Does it under-triage?** **No — 0/60 under-triage and 0/60 over-triage** in the balanced benchmark; 0 tier mismatches in 196 gate cases. Architecturally it also cannot under-triage in production: `alert_router` takes `max(rule, model)` and CRITICAL is never downgraded.

8. **Major failure modes?** One in 60: an incomplete value citation on a MEDIUM case (tier and action still correct), which would likely pass the production guardrail since that requires only one cited value. HISTORICALLY, the **Q4_K_M** build failed on 3 MEDIUM `action_mismatch` cases — not present in the shipped Q5_K_M. **MEDIUM is the recurring weak tier**, consistent with it being under-represented in training (263 vs 398 rows).

9. **Suitable as an independent clinical inference engine?** **No.** There is no clinician review, no real patient data and no prospective validation in the repository. All evaluation is synthetic and consistency-based. It has also never been measured on its intended edge hardware, and the measurement here used more CPU than the scheduler allocated.

10. **What role should it have?** Exactly its current one: **the language layer after a deterministic decision.** The rules classify; the model explains; the guardrail bounds it; the router recomputes the action from the rules; CRITICAL patients bypass it entirely. Within that envelope the measurements support its use. Outside it, they do not.
