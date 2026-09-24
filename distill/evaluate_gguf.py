"""
Evaluate a GGUF build through llama-server's HTTP API.

This is the DEPLOYED path, not a laboratory one: the same endpoint the dashboard
will call, with the same JSON-schema constraint. Quantization can change model
behaviour, so the gates are re-run here rather than assumed to carry over.

As in distill/evaluate.py: every set is scored on its own and the overall
verdict is the AND. Nothing pooled.
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, sys, time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distill.sampler import sample, make_case
from distill.render import render_user, render_system
from distill.gate import check
from distill.evaluate import score, GATES, GATED_SETS
from distill.evalsets import boundary_cases, robustness_cases
from agent.nodes.llm_analyzer import OUTPUT_SCHEMA, _extract_json


def ask(client, url, model, system, user, schema, timeout):
    r = client.post(f"{url}/v1/chat/completions", timeout=timeout, json={
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.0, "max_tokens": 400,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "assessment", "schema": schema}},
    })
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--model", default="gguf")
    ap.add_argument("--data", default="/ssd_scratch/mahimakopalley/distill_data/vitals_v3/accepted.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--indist-n", type=int, default=60)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--concurrency", type=int, default=4,
                    help="must not exceed llama-server --parallel")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data)]
    to_case = lambda r: make_case(r["case"]["systolic_bp"], r["case"]["diastolic_bp"],
                                  r["case"]["heart_rate"], r["case"]["spo2"], None)
    held = [to_case(r) for r in rows if r.get("split") == "heldout"]
    ind = [to_case(r) for r in rows if r.get("split") == "train"][: a.indist_n]
    sets = {"held_out": held, "boundary": boundary_cases(),
            "indist": ind, "robustness": robustness_cases()}
    sets = {k: v for k, v in sets.items() if v}

    SYS = render_system()
    report, t0 = {}, time.time()
    def one(c):
        up = render_user(c)
        with httpx.Client() as cl:
            try:
                raw = ask(cl, a.url, a.model, SYS, up, OUTPUT_SCHEMA, a.timeout)
                obj = _extract_json(raw)
                ok, reasons, notes = check(c, obj, raw, up)
            except Exception as e:
                ok, reasons, notes, obj = False, [f"transport_error:{type(e).__name__}"], [], None
        return dict(ok=ok, reasons=reasons, notes=notes, case=c.values(),
                    tier=c.tier, model_tier=(obj or {}).get("risk_level"))

    if True:
        for name, cases in sets.items():
            with cf.ThreadPoolExecutor(a.concurrency) as ex:
                out = list(ex.map(one, cases))
            s = score(out)
            passed = all(s.get(g, 0.0) >= thr for g, thr in GATES.items()) if name in GATED_SETS else None
            report[name] = dict(metrics=s, gated=name in GATED_SETS, passed=passed,
                                failures=[x for x in out if not x["ok"]][:8])
            m = s
            print(f"  {name:<12} n={m['n']:>3} json={m['json_valid_pct']:>6.1f} "
                  f"tier={m['tier_agreement_pct']:>6.1f} action={m['action_agreement_pct']:>6.1f} "
                  f"clean={m['fully_clean_pct']:>6.1f}  "
                  f"{'PASS' if passed else ('FAIL' if name in GATED_SETS else 'reported')}", flush=True)

    gated = {k: v for k, v in report.items() if v["gated"]}
    overall = all(v["passed"] for v in gated.values()) if gated else False
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}  (AND of {', '.join(gated)})")
    Path(a.out).write_text(json.dumps(dict(
        url=a.url, gates=GATES, gated_sets=list(GATED_SETS), overall_pass=overall,
        seconds=round(time.time() - t0), report=report), indent=2, default=str))
    print(f"report -> {a.out}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
