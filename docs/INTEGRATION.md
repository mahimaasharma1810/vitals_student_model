# Vitals Student Model — Integration Guide

*The deployable model, and how to call it from your dashboard.*
*Written: 23 September 2026*

---

# WHAT YOU ARE GETTING

A small model that writes the **clinical explanation** for a vitals reading.

```
   Qwen2.5-0.5B  +  LoRA trained on MedGemma 4B answers
   merged, converted to GGUF, quantized
```

| File | Size | Gate verdict | Use |
|---|---|---|---|
| `vitals-v3-Q4_K_M.gguf` | 379 MB | ❌ **FAIL** | do not deploy -- see below |
| `vitals-v3-Q5_K_M.gguf` | **401 MB** | ✅ **PASS** (100% on every set) | **USE THIS ONE** |
| `vitals-v3-Q8_0.gguf` | 506 MB | see report | larger alternative |
| `vitals-v3-f16.gguf` | 948 MB | reference | not for deployment |

## Why not Q4_K_M

It was measured, not assumed. Q4_K_M failed the gate suite:

```
set          n     json   tier  action  clean    verdict
held_out   103    100.0  100.0    98.1   98.1    FAIL
boundary    45    100.0  100.0   100.0  100.0    PASS
indist      40    100.0  100.0    97.5   97.5    FAIL
```

Three cases, all MEDIUM, all escalating `clinician_review` -> `urgent_review`.
Quantization degraded the action mapping -- the one behaviour the model actually
LEARNED (the tier is given in the prompt, so tier agreement is partly copying).

Q5_K_M, only 22 MB larger, is 100% clean on all three sets. Use it.

**Location:** `/ssd_scratch/mahimakopalley/gguf_v3/`

---

# THE MOST IMPORTANT THING TO UNDERSTAND

> **This model does NOT decide the patient's risk level.**
> **The rules decide. The model only writes the explanation.**

```
      vitals from your dashboard
              |
              v
   +----------------------------+
   |  RULES  (plain Python)     |   guardrails/clinical_rules.py
   |  NEWS2 + qSOFA             |   <-- THE DECISION HAPPENS HERE
   +----------------------------+
              |
              |  tier = NORMAL / MEDIUM / HIGH / CRITICAL
              v
   +----------------------------+
   |  STUDENT MODEL             |   writes reasoning in JSON
   +----------------------------+
              |
              v
   +----------------------------+
   |  OUTPUT CHECK              |   if the model disagrees with the
   |                            |   rules, the RULES WIN
   +----------------------------+
              |
              v
          alert on dashboard
```

If you call the model **without** running the rules first, you lose the entire
safety design. Please do not do that.

---

# HOW TO RUN IT

We use **llama-server**, not Ollama. (Ollama on this machine is a failed
download — the file is 9 bytes of text saying "Not Found".)

llama-server gives an **OpenAI-compatible API**, so most dashboard code can talk
to it without changes.

## Start the server

```bash
/ssd_scratch/mahimakopalley/llama.cpp/build/bin/llama-server \
    -m /ssd_scratch/mahimakopalley/gguf_v3/vitals-v3-Q4_K_M.gguf \
    --host 0.0.0.0 --port 8099 \
    -c 2048 --parallel 4
```

Add `-ngl 99` if your llama.cpp build has CUDA. (The current build here does
not, so it runs on CPU. For a 0.5B model that is acceptable.)

## Check it is alive

```bash
curl http://127.0.0.1:8099/health
# {"status":"ok"}
```

---

# HOW TO CALL IT

## Step 1 — score the vitals with the RULES (not the model)

```python
from guardrails.clinical_rules import (
    calculate_news2, calculate_qsofa,
    determine_alert_level, determine_recommended_action)

news2 = calculate_news2(systolic_bp, diastolic_bp, heart_rate, spo2)
qsofa = calculate_qsofa(systolic_bp, heart_rate, spo2)
tier   = determine_alert_level(news2, qsofa)      # <-- this is the decision
action = determine_recommended_action(tier)
```

## Step 2 — build the prompt

```python
from agent.prompts import SYSTEM_PROMPT, build_user_prompt

user = build_user_prompt(
    snapshot=snapshot,          # patient_id, timestamp, the 4 vitals
    news2=news2, qsofa=qsofa,
    rule_alert_level=tier.value,
    ehr_context="No EHR data available.")
```

**Do not write your own prompt text.** The model was trained on exactly this
format. A different format will give worse answers, silently.

## Step 3 — ask the model, with the schema enforced

```python
import httpx
from agent.nodes.llm_analyzer import OUTPUT_SCHEMA

resp = httpx.post("http://127.0.0.1:8099/v1/chat/completions", timeout=120, json={
    "model": "vitals-v3",
    "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user",   "content": user}],
    "temperature": 0.0,
    "max_tokens": 400,
    "response_format": {"type": "json_schema",
                        "json_schema": {"name": "assessment",
                                        "schema": OUTPUT_SCHEMA}},
})
answer = resp.json()["choices"][0]["message"]["content"]
```

`temperature: 0.0` matters — clinical output should be reproducible.
`response_format` matters — it forces valid JSON instead of hoping for it.

## Step 4 — check the answer against the rules

```python
import json
obj = json.loads(answer)

if obj["risk_level"] != tier.value:
    obj["risk_level"] = tier.value          # rules win
    log_override("model disagreed with rules")

if obj["recommended_action"] != action.value:
    obj["recommended_action"] = action.value
    log_override("model action overridden")
```

**Never skip step 4.** It is what makes a small model safe here.

---

# THE ANSWER SHAPE

```json
{
  "risk_level": "NORMAL | MEDIUM | HIGH | CRITICAL",
  "confidence": 0.0 - 1.0,
  "reasoning": "plain English, quoting the actual values",
  "sepsis_risk_flag": true | false,
  "contributing_factors": ["..."],
  "recommended_action": "immediate_response | urgent_review | clinician_review | observation",
  "limitations": "what was missing, e.g. temperature, respiratory rate",
  "disclaimer": "CLINICAL DECISION SUPPORT ONLY..."
}
```

---

# IF THE MODEL IS DOWN

The dashboard must keep working. The rules do not need the model:

```
   model available   ->  alert + written explanation
   model down        ->  alert + "explanation unavailable"
```

The risk level, the score and the recommended action all come from the rules.
Only the sentence is lost. Please build the dashboard so a model outage
degrades the display, never the alert.

---

# HOW IT WAS TESTED

Each test set was scored **separately**, with its own pass mark. Nothing was
pooled or averaged — pooling is how the earlier version of this project turned
an 8% failure rate into a 3.6% "pass" by adding easy cases to the denominator.

```
   set          n     json   tier  action  clean    verdict
   held_out   103    100.0  100.0   100.0  100.0    PASS
   boundary    45    100.0  100.0   100.0  100.0    PASS
   indist     100    100.0  100.0   100.0  100.0    PASS
   OVERALL: PASS

   untrained base model : 0.0% clean
   trained student      : 100.0% clean
```

`held_out` is the meaningful one: a region of the vital-sign space the model
**never saw during training**. Passing there means it generalised rather than
memorised.

## Please read this before quoting "100%"

The rule's verdict is written **inside the question we give the model**. So
agreeing with the tier is partly a copying task, not deep medical reasoning.

What the model genuinely **learned** is:

- the recommended-action mapping (this is **not** in the question), and
- the exact output format.

The untrained base model scored **0%** on both. So the training did real work.
But please always attach this qualification when reporting the number.

---

# WHAT THIS MODEL CANNOT DO

| | |
|---|---|
| Decide risk level | ❌ the rules do that |
| ABG interpretation | ❌ not trained yet |
| ECG interpretation | ❌ paused, ~1,237 rows already generated |
| Use patient history (EHR) | ❌ trained with "No EHR data available." |
| Temperature, resp. rate, consciousness | ❌ not among the 4 monitored vitals |

The EHR point is worth noting: the live EHR store held only **8 documents
across 5 patients**, far too little to learn from, so the model was trained to
work without it. If you later populate the EHR, the model must be retrained to
use it.

---

# PROVENANCE

Every deployed file can be traced to its training run:

```
   /ssd_scratch/mahimakopalley/distill_models/v3_vitals_merged/provenance.json
```

It records the base model, the adapter, the training data **and its SHA-256
fingerprint**, the held-out region, row counts, and `leakage_rows: 0`.

If someone asks "which data produced this model?", that file answers it.

---

# QUICK REFERENCE

```bash
# start
llama-server -m gguf_v3/vitals-v3-Q4_K_M.gguf --host 0.0.0.0 --port 8099 -c 2048

# health
curl http://127.0.0.1:8099/health

# the four steps
#   1. rules decide the tier
#   2. build_user_prompt() -- do not hand-write it
#   3. POST /v1/chat/completions with response_format json_schema
#   4. override anything the model says that contradicts the rules
```
