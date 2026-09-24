"""
Honest evaluation for the vitals student.

THE RULE OF THIS FILE: every set is scored on its own, against its own
threshold, and the overall verdict is the AND of them all. Nothing is pooled and
nothing is averaged.

Why so emphatic: the previous pipeline's `quant_results.json` marked a model
PASSES_ALL_GATES at 3.6% fallback. That 3.6% came from pooling 25 gold + 25
boundary (4 failures, 8.0%) with 200 easy in-distribution rows (5 failures).
The weights never improved -- only the denominator changed. On the hard sets
alone it failed. That must never be possible here.

Sets:
  held_out    the hidden region, never trained on  -> the generalisation answer
  boundary    values on a scoring threshold        -> off-by-one errors
  indist      same distribution as training        -> REFERENCE ONLY
  robustness  impossible values                    -> reported, NOT gated
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distill.render import render_user, render_system
from distill.gate import check
from distill.evalsets import boundary_cases, robustness_cases
from distill.sampler import make_case
from agent.nodes.llm_analyzer import _extract_json, OUTPUT_SCHEMA

STUDENT = "/ssd_scratch/mahimakopalley/distill_models/student"

# Contract gates. These are not quality preferences -- they are the agent's
# output contract. A row failing any of them cannot be used by the agent at all.
GATES = {
    "json_valid_pct":       100.0,   # else the agent falls back to rules
    "tier_agreement_pct":   100.0,   # must never contradict the rule verdict
    "action_agreement_pct": 100.0,   # action mapping is deterministic
    "no_fabrication_pct":   100.0,
    "no_prohibited_pct":    100.0,
}
GATED_SETS = ("held_out", "boundary", "indist")   # robustness is reported only


def load_model(base, adapter):
    tok = AutoTokenizer.from_pretrained(adapter or base)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return tok, model.eval()


def run_set(name, cases, tok, model, bs, max_new):
    SYS = render_system()
    rows = []
    for i in range(0, len(cases), bs):
        chunk = cases[i:i + bs]
        texts = [tok.apply_chat_template(
            [{"role": "system", "content": SYS},
             {"role": "user", "content": render_user(c)}],
            add_generation_prompt=True, tokenize=False) for c in chunk]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda:0")
        n_in = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for c, o in zip(chunk, out):
            raw = tok.decode(o[n_in:], skip_special_tokens=True)
            obj = _extract_json(raw)
            ok, reasons, notes = check(c, obj, raw, render_user(c))
            rows.append(dict(case=c.values(), tier=c.tier, ok=ok,
                             reasons=reasons, notes=notes,
                             model_tier=(obj or {}).get("risk_level"),
                             raw_len=len(raw)))
    return rows


def score(rows):
    n = len(rows) or 1
    def pct(f): return round(100.0 * sum(1 for r in rows if f(r)) / n, 2)
    has = lambda r, k: any(x.startswith(k) for x in r["reasons"])
    return dict(
        n=len(rows),
        json_valid_pct       = pct(lambda r: not has(r, "not_json") and not has(r, "missing_fields")),
        tier_agreement_pct   = pct(lambda r: not has(r, "risk_level_contradiction")),
        action_agreement_pct = pct(lambda r: not has(r, "action_mismatch")),
        no_fabrication_pct   = pct(lambda r: not has(r, "fabricated_numbers")),
        no_prohibited_pct    = pct(lambda r: not has(r, "prohibited_treatment")),
        fully_clean_pct      = pct(lambda r: r["ok"]),
        fallback_pct         = round(100.0 - pct(lambda r: not has(r, "not_json")
                                                 and not has(r, "missing_fields")), 2),
    )


def verdict(name, s):
    if name not in GATED_SETS:
        return None, {}
    per = {g: (s.get(g, 0.0) >= thr) for g, thr in GATES.items()}
    return all(per.values()), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="LoRA dir; omit to score the base model")
    ap.add_argument("--base", default=STUDENT)
    ap.add_argument("--data", default="/ssd_scratch/mahimakopalley/distill_data/vitals_v3/accepted.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--indist-n", type=int, default=100)
    a = ap.parse_args()

    # held-out and in-distribution cases come from the generated corpus
    rows = [json.loads(l) for l in open(a.data)]
    def to_case(r):
        c = make_case(r["case"]["systolic_bp"], r["case"]["diastolic_bp"],
                      r["case"]["heart_rate"], r["case"]["spo2"], None)
        return c
    held = [to_case(r) for r in rows if r.get("split") == "heldout"]
    ind  = [to_case(r) for r in rows if r.get("split") == "train"][: a.indist_n]

    sets = {"held_out": held, "boundary": boundary_cases(),
            "indist": ind, "robustness": robustness_cases()}
    sets = {k: v for k, v in sets.items() if v}

    tok, model = load_model(a.base, a.adapter)
    report, t0 = {}, time.time()
    for name, cases in sets.items():
        r = run_set(name, cases, tok, model, a.batch_size, a.max_new_tokens)
        s = score(r)
        passed, per = verdict(name, s)
        report[name] = dict(metrics=s, gated=(name in GATED_SETS),
                            passed=passed, per_gate=per,
                            failures=[x for x in r if not x["ok"]][:10])

    gated = {k: v for k, v in report.items() if v["gated"]}
    overall = all(v["passed"] for v in gated.values()) if gated else False

    print(f"\n{'set':<12}{'n':>5}{'json':>8}{'tier':>8}{'action':>8}{'fabric':>8}"
          f"{'clean':>8}{'fallback':>10}   verdict")
    print("-" * 82)
    for name, v in report.items():
        m = v["metrics"]
        vd = "PASS" if v["passed"] else ("FAIL" if v["gated"] else "reported only")
        print(f"{name:<12}{m['n']:>5}{m['json_valid_pct']:>8.1f}{m['tier_agreement_pct']:>8.1f}"
              f"{m['action_agreement_pct']:>8.1f}{m['no_fabrication_pct']:>8.1f}"
              f"{m['fully_clean_pct']:>8.1f}{m['fallback_pct']:>10.2f}   {vd}")
    print("-" * 82)
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}"
          f"   (AND of {', '.join(gated)} -- nothing pooled, nothing averaged)")
    if not overall:
        for name, v in gated.items():
            if not v["passed"]:
                bad = [g for g, ok in v["per_gate"].items() if not ok]
                print(f"   {name} failed: {', '.join(bad)}")

    out = Path(a.out or ((a.adapter or a.base) + "/eval_report.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        adapter=a.adapter, base=a.base, data=a.data, gates=GATES,
        gated_sets=list(GATED_SETS), overall_pass=overall,
        seconds=round(time.time() - t0), report=report,
        note="sets are scored separately; the overall verdict is the AND of the "
             "gated sets. robustness is reported but never gated.",
    ), indent=2, default=str))
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
