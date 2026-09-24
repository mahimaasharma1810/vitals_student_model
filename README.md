# Vitals Student Model

A small language model that writes the **clinical explanation** for a vital-signs
alert. It does **not** decide the patient's risk — deterministic rules do that
before and after it runs.

```
   Qwen2.5-0.5B  +  LoRA distilled from MedGemma-4B
   494,032,768 parameters  ->  401 MB as Q5_K_M
```

---

## The one thing to understand first

```
      vitals
        |
        v
  +-------------+
  |    RULES    |  NEWS2 + qSOFA  ->  tier        <== THE DECISION
  +-------------+
        |
        v
  +-------------+
  |   STUDENT   |  writes the explanation only
  +-------------+
        |
        v
  +-------------+
  |  GUARDRAIL  |  if the model disagrees, THE RULES WIN
  +-------------+
        |
        v
      alert
```

Calling the model without running the rules first removes the entire reason a
0.5B model is acceptable here.

---

## Measured performance

From `docs/STUDENT_MODEL_TECHNICAL_AND_BENCHMARK_REPORT.md`, measured
2026-09-24 on a tier-balanced set of 60 cases (15 per tier):

| Metric | Result |
|---|---|
| Latency (median) | 6.97 s |
| TTFT (median) | 0.42 s |
| Generation | 38.75 tok/s |
| RAM peak | 1.14 GiB |
| Rubric alignment | 60/60 tier, 60/60 action |
| Under-triage | **0 / 60** |
| Format compliance | 60/60 |
| Fabrication-free | 60/60 |
| Tests | 51 passed |

Hardware: Intel Xeon E5-2640 v4, 20 threads, **CPU only**. No GPU used at
inference. Power and energy were **not measured** — see the report.

**Read the caveat in the report before quoting "100%".** The rule's verdict is
included in the prompt, so tier agreement is partly a copying task. The
genuinely learned behaviours are the action mapping and the output format,
where the untrained base model scored **0%**.

**This model is not clinically validated.** No clinician has reviewed any
output; all evaluation is synthetic and consistency-based.

---

## Layout

```
  distill/          the pipeline: sample -> render -> generate -> gate -> train -> evaluate
  guardrails/       deterministic clinical rules and validators   [vendored]
  agent/            the exact runtime prompt and output schema    [vendored]
  vitals/           pydantic schemas                              [vendored]
  config/           settings                                      [vendored]
  tests/            rule and guardrail test suite                 [vendored]
  scripts/          check_vendored.py -- drift detector
  docs/             technical report, design rationale, integration guide
```

`[vendored]` = copied from the parent `MedGemma-Agent` repository.
**Run `scripts/check_vendored.py` before training.** See `PROVENANCE.md`.

---

## Quick start

### Serve an existing model

```bash
llama-server -m vitals-v3-Q5_K_M.gguf --host 127.0.0.1 --port 8099 \
             -c 8192 --parallel 4
```

`-c` is the **total** context, divided across `--parallel` slots.
`-c 2048 --parallel 4` gives 512 tokens per slot and **silently truncates every
answer**.

### Check it works

```bash
python distill/test_student.py smoke
python distill/test_student.py stress
python distill/test_student.py compare --n 25
```

### Benchmark

```bash
python distill/benchmark_student.py --n 60 --warmup 3 --balanced --seed 4242 \
       --out benchmark.json
```

Use `--balanced`. Random sampling over the physiological space is ~71%
CRITICAL, and CRITICAL bypasses the model in production — an unbalanced run
mostly measures cases the model never sees.

---

## Rebuilding the model from scratch

Needs a CUDA GPU (~8 GB for the teacher) and `requirements-train.txt`.

```bash
python scripts/check_vendored.py --upstream /path/to/MedGemma-Agent   # do this first
python distill/generate.py  --out DATA --per-tier 400 --held-out-per-tier 40
python distill/train.py     --data DATA/accepted.jsonl --out RUN
python distill/evaluate.py  --adapter RUN/adapter --out RUN/eval.json
python distill/package_model.py --adapter RUN/adapter --out MERGED
# convert to GGUF with llama.cpp, then re-gate:
python distill/evaluate_gguf.py --out eval_gguf.json
```

**Always re-gate after quantizing.** Q4_K_M passed at full precision and then
failed the gate suite on 3 MEDIUM cases — quantization degraded the action
mapping, which is the behaviour the model actually learned.

---

## How the evaluation avoids fooling itself

Four guards, one per failure mode. Full reasoning in
`docs/STUDENT_MODEL_IDEOLOGY.md`.

| Guard | Stops |
|---|---|
| **WIDE** | sampling physiology, not patient archetypes — nothing to memorise |
| **HIDDEN** | a region of the space is never trained on, and tested only there |
| **HARD** | every set scored separately; **nothing pooled or averaged** |
| **REAL** | training uses the agent's own prompt, imported not copied |

Pooling is not a hypothetical: an earlier build turned an 8.0% failure rate on
hard cases into a 3.6% "pass" by adding 200 easy rows to the denominator.

---

## What this model cannot do

| | |
|---|---|
| Decide risk level | the rules do — by design |
| ABG or ECG | not trained |
| Use patient history | trained with `"No EHR data available."` |
| Respiratory rate, temperature, consciousness | not among the four monitored vitals |

---

## Not included

No model weights, no training corpora, no patient data. All are excluded by
`.gitignore`. Weights are distributed separately.
