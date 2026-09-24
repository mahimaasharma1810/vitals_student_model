# Student Model — The Ideology

How we build a small student model that we can actually trust, and how we stop
ourselves from being fooled by good-looking numbers.

Written: 2026-09-22
Applies to: branch `mod_1`, code in `distill/`

> **Note on the older document.** `STUDENT_MODEL_README.md` (2026-09-20) explains
> what a student model is and lists four blockers. Three of its statements are
> now out of date: distillation data and three trained runs exist, and a
> quantisation sweep was done. Two of its blockers are still true: **Ollama is
> not installed** (the file at `~/bin/ollama` is ASCII text containing
> `Not Found`, i.e. a failed download) and the **EHR store is nearly empty**
> (5 patients, 8 documents). Read that file for the beginner explanation, and
> this file for the method.

---

# THE ONE SENTENCE

> **The rules decide. The model only explains.**
> **Our job is to copy the explaining, and to make sure we never fool ourselves**
> **about how well we copied it.**

Everything below follows from this.

---

# PART 1 — WHAT ARE WE ACTUALLY COPYING?

This is the most misunderstood point, so please read it slowly.

Our system is **not** one big model. It is a pipeline of 7 steps, and only ONE
of them is a model.

```
   THE AGENT  (agent/graph.py)

   Step 1   Input Validator ............ plain Python rules      no model
   Step 2   EHR Retriever .............. database search         no model
   Step 3   Risk Scorer (NEWS2+qSOFA) .. plain arithmetic        no model
   Step 4   LLM Analyzer ............... MedGemma 4B         <== ONLY THIS
   Step 5   Output Validator ........... plain Python rules      no model
   Step 6   Alert Router ............... plain Python rules      no model
   Step 7   Audit Logger ............... database write         no model
```

Steps 1, 2, 3, 5, 6 and 7 are ordinary code. They run anywhere, they are fast,
and they need no GPU. **We do not distill them. We do not train anything for
them.** They are simply carried across as they are.

So the student model has exactly **one** job:

```
   GIVEN   the vitals
           + the NEWS2 score breakdown
           + the qSOFA score breakdown
           + the alert level the RULES already decided
           + patient EHR context

   WRITE   one JSON object with 8 fields:
           risk_level, confidence, reasoning, sepsis_risk_flag,
           contributing_factors, recommended_action, limitations, disclaimer
```

## The safety consequence

The student **never decides** whether a patient is critical. The rules decided
that before the model was even called, and a validator checks the model
afterwards.

```
     vitals
        |
        v
   +-----------+
   |   RULES   |  NEWS2 + qSOFA  ->  tier = CRITICAL
   +-----------+
        |
        v
   +------------------+
   | student explains |   <- even if this writes poor text,
   +------------------+      the ALERT still says CRITICAL
        |
        v
   +------------------+
   | output validator |   <- catches contradiction, missing fields,
   +------------------+      prohibited content
        |
        v
      ALERT
```

This is why a 0.5B model is acceptable here. It is not carrying the clinical
decision. It is carrying the sentence that explains the decision.

---

# PART 2 — THE MASTER BLOCK DIAGRAM

```
  +========================================================================+
  |                     STAGE A  --  MAKE THE CASES                        |
  +========================================================================+

     PHYSIOLOGICAL SPACE  (every possible patient)
       sbp 40-300 | dbp 20-200 | hr 1-300 | spo2 50-100
                       |
                       |  each vital drawn INDEPENDENTLY of the others
                       |  => no syndrome template => nothing to memorise
                       v
             +----------------------+
             |  physical validity   |  dbp < sbp, pulse pressure 15-120
             +----------------------+     (physics, NOT a clinical pattern)
                       |
           +-----------+-----------+
           |                       |
   falls inside the          falls outside
   HIDDEN REGION?                  |
           |                       |
           v                       v
   +---------------+       +----------------+
   |   HELD-OUT    |       |     TRAIN      |
   | never trained |       |  1,600 cases   |
   |  160 cases    |       |  balanced over |
   | tested ONLY   |       |   4 tiers      |
   +---------------+       +----------------+
           |                       |
           +-----------+-----------+
                       |
                       v
       +----------------------------------+
       |   DETERMINISTIC RULES decide     |  guardrails/clinical_rules.py
       |   NEWS2 + qSOFA  ->  tier        |  the SAME file the agent runs
       +----------------------------------+
            we NEVER choose the tier ourselves

  +========================================================================+
  |                    STAGE B  --  ASK THE TEACHER                        |
  +========================================================================+
                       |
                       v
       +----------------------------------+
       |  RENDER the agent's EXACT prompt |  imports agent/prompts.py
       +----------------------------------+  so the two cannot drift apart
                       |
                       v
       +----------------------------------+
       |  TEACHER   MedGemma 4B           |  8.0 GB on the RTX 2080 Ti
       |  writes the 8-field JSON         |  batch 8, ~8 s per case
       +----------------------------------+
                       |
                       v
       +----------------------------------+
       |  CONSISTENCY GATE                |
       |   - valid JSON with all 8 fields?|
       |   - does it CONTRADICT the rule  |
       |     verdict?                     |
       |   - recommended_action matches   |
       |     the rule mapping?            |
       |   - invented numbers?            |
       |   - drugs, doses, treatments?    |
       +----------------------------------+
              |                  |
          accepted            rejected --> rejects.jsonl WITH the reason
              |                            (accept rate stays auditable)
              v
        TRAINING DATA

  +========================================================================+
  |                  STAGE C  --  TRAIN AND JUDGE                          |
  +========================================================================+
              |
              v
       +----------------------------------+
       |  TRAIN the student (LoRA)        |  Qwen2.5-0.5B
       +----------------------------------+
              |
              v
   +--------------------------------------------------------------+
   |  EVALUATE -- every set scored SEPARATELY, never pooled        |
   |                                                              |
   |    gold        hand-written hard cases    own threshold      |
   |    boundary    sitting on the thresholds  own threshold      |
   |    HELD-OUT    the hidden region          own threshold  <== |
   |    robustness  impossible values          reported apart     |
   +--------------------------------------------------------------+
              |
              v
       +----------------------------------+
       |  QUANTISE -> GGUF, then re-judge |  must pass the SAME gates
       +----------------------------------+
              |
              v
       +----------------------------------+
       |  PLUG INTO THE AGENT             |  replaces step 4 only
       +----------------------------------+
```

---

# PART 3 — THE FOUR WAYS A STUDENT MODEL FOOLS YOU

This part is the actual ideology. Four traps, four guards.

## TRAP 1 — **COPY**: the model memorises the pattern, not the physiology

```
   If every training patient is built from 7 syndrome templates,
   the model learns 7 shapes.
   It will score beautifully. It has understood nothing.
```

### GUARD — **WIDE**

Draw every vital **independently** across the whole physiological range. There
is no template to copy, because we never created one. What kind of patient it
turns out to be is decided afterwards, by the rules.

*Measured check (`distill/sampler.py`):*

```
          sbp    dbp     hr   spo2
   sbp   1.00   0.71   0.02  -0.14
   dbp   0.71   1.00  -0.02  -0.10
   hr    0.02  -0.02   1.00  -0.13
   spo2 -0.14  -0.10  -0.13   1.00
```

Heart rate is uncorrelated with blood pressure (0.02). There is no hidden
"shock shape". The 0.71 between sbp and dbp is the pulse-pressure validity
rule -- physics, not a clinical pattern. The mild -0.1 on spo2 comes from tier
balancing; we state it rather than claim a clean zero.

## TRAP 2 — **MIRROR**: the exam is set from the practice book

```
   ONE generator  ->  1,800 "train"  +  200 "validation"
   Both from the same book, same 7 templates.
   Model scores 98%. The number means almost nothing.
```

**This was the real defect in the previous version.** Its `val.jsonl` was 0 bytes
and the 200 "validation" rows were a split of the same generated pool. The only
truly independent evidence in the whole project was 25 hand-written cases, where
the score dropped from 98% to 88%.

### GUARD — **HIDDEN**

Cut one region out of the map. Never show it during training. Test only inside
it.

```
        HIGH hr
           ^
           |  . . . .       +-------------+
           | . . . . .      |   HIDDEN    |   trained on: never
           |  . . . .       |   REGION    |   tested on : only here
           | . . . . .      +-------------+
           +-----------------------------> BP

   learned real physiology  ->  handles the hidden region
   only memorised examples  ->  collapses there
```

*Choosing the region matters.* Our first choice, the shock corner
(SBP <= 100 AND HR >= 110), was measured and found to be **100% CRITICAL** --
5,369 of 5,369 cases. SBP <= 100 scores a qSOFA point, HR >= 110 scores another,
and qSOFA 2 forces CRITICAL. Hiding it would have tested one tier only. We
switched the default to `wide_pulse_pressure`, which spans all four tiers.

| region | n | NORMAL | MEDIUM | HIGH | CRITICAL |
|---|---|---|---|---|---|
| shock_corner | 5,369 | 0 | 0 | 0 | 5,369 |
| wide_pulse_pressure | 42,930 | 6,883 | 2,531 | 168 | 33,348 |
| silent_hypoxia | 42,019 | 1,648 | 993 | 72 | 39,306 |
| hypertensive_brady | 2,084 | 166 | 58 | 8 | 1,852 |

## TRAP 3 — **DILUTE**: easy questions hide the hard failures

```
   hard cases only    :  4 wrong of  50  =  8.0%   FAILS the gate
   add 200 easy rows  :  9 wrong of 250  =  3.6%   PASSES the gate

   Same weights. Same failures. Only the denominator changed.
```

This is precisely what the existing `quant_results.json` does: the row marked
`PASSES_ALL_GATES: true` reaches 3.6% only by pooling 200 easy rows with the 50
hard ones. On gold and boundary alone it is 8.0%, which fails.

### GUARD — **HARD**

Every set is scored on its own, against its own threshold. Gold is never mixed
with easy rows. Impossible-value probes (SpO2 110, HR 15) are reported in a
separate robustness set, because failing to narrate an impossible reading is not
a clinical failure and must not be averaged into one.

## TRAP 4 — **DRIFT**: you train a task the system never asks for

```
   the old student was trained to write:
       "Clinical Assessment: ... Primary Concern: ..."

   the agent actually asks for:
       {"risk_level": ..., "confidence": ..., ... 8 fields}

   Result: a well-trained model that cannot be plugged in AT ALL.
```

This is what happened to v0, v1 and v2. Their prose does not parse as the
agent's JSON (`_extract_json` returns nothing usable), and the input format
differs too. It is the reason `OLLAMA_MODEL` still points at the teacher:
integration was never possible, not merely postponed.

### GUARD — **REAL**

`distill/render.py` **imports** `agent/prompts.py` and the agent's
`OUTPUT_SCHEMA`. It does not copy them into a second place. If anyone edits the
agent's prompt tomorrow, the training data changes with it automatically, and
the two can never silently disagree.

---

# THE MNEMONIC:  **WIDE · HIDDEN · HARD · REAL**

| Guard | Stops the trap | In one line |
|---|---|---|
| **WIDE** | COPY | sample physiology, never templates |
| **HIDDEN** | MIRROR | hide a region, test only there |
| **HARD** | DILUTE | score hard sets alone, never pooled |
| **REAL** | DRIFT | train on the agent's own prompt |

---

# PART 4 — WHAT "ACCURATE" WILL MEAN, IN NUMBERS

"The student is accurate" is not a feeling. It is this checklist, with each line
measured separately:

```
  CONTRACT (must be 100%, these are not negotiable)
  [ ] valid agent JSON with all 8 required fields
  [ ] risk_level never contradicts the rule tier
  [ ] recommended_action matches the rule mapping exactly
  [ ] no invented numbers in the narrative
  [ ] no drugs, doses or treatment instructions
  [ ] disclaimer present

  QUALITY (each set has its OWN threshold, never pooled)
  [ ] gold set          -- hand-written hard cases
  [ ] boundary set      -- values sitting on the thresholds
  [ ] HELD-OUT region   -- the hidden region      <== answers "did it overfit?"
  [ ] robustness set    -- impossible values, reported separately

  DEPLOYMENT
  [ ] after quantisation to GGUF, every line above still holds
  [ ] the agent runs end to end on the student, not the teacher
```

The **HELD-OUT** line is the one that actually answers the overfitting question.
Everything else can look perfect while the model has learned nothing general.

---

# PART 5 — WHERE THE CODE LIVES

| File | Role |
|---|---|
| `distill/regions.py` | Held-out region definitions and their measured tier mix |
| `distill/sampler.py` | Physiology sampler; independent draws, no templates |
| `distill/render.py` | Renders the agent's exact runtime prompt |
| `distill/gate.py` | Consistency gate; rejects contradictions, keeps reasons |
| `distill/generate.py` | Batched, resumable teacher generation |
| `guardrails/clinical_rules.py` | The rules -- **reused, never reimplemented** |
| `agent/prompts.py` | The prompt -- **imported, never copied** |

## Design decisions on record

| Decision | Choice | Why |
|---|---|---|
| Case generation | sample physiology, no templates | removes the archetype to memorise |
| Held-out test | hidden region of the vital space | automatic, needs no clinician time |
| Held-out region | `wide_pulse_pressure` | spans all 4 tiers; shock corner is 100% CRITICAL |
| EHR context | option (a): always "No EHR data available." | only 8 documents exist; inventing a second synthetic distribution would create a new overfitting risk |
| Domains | vitals only for now | ABG and ECG follow the same pattern later |
| Clinical review | none available yet | hand-written gold cases wait for a doctor; the hidden region does not |

## An open question for the team

About **one third** of accepted teacher rows set `sepsis_risk_flag` to true where
qSOFA scores below 2. The agent accepts the model's own flag
(`agent/nodes/output_validator.py`, which uses qSOFA only as a default), so the
student will inherit a more liberal sepsis flag than the rule layer would give.
This is the agent's existing design, not a change introduced here. It should be
a conscious decision before deployment, not an accident. The gate records it as
a note on every affected row so it can be measured.

---

# ONE-PAGE SUMMARY

We are copying **one step out of seven**. The rules decide the patient's risk;
the student only writes the explanation, and a validator checks it afterwards.

We build the training cases by sampling **physiology**, not templates, so there
is no pattern to memorise. We **hide one region** of the vital-sign space and
test only there, because that is the only honest way to ask whether the model
generalised. We score **hard cases on their own**, so an easy majority can never
hide a dangerous failure. And we train on the **agent's own prompt**, so the
finished student actually drops into the system.

**WIDE · HIDDEN · HARD · REAL.**
