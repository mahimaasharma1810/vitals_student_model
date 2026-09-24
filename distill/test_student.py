#!/usr/bin/env python
"""
Test the vitals student model.

    python distill/test_student.py smoke          one request, is it alive and correct?
    python distill/test_student.py ask 88 52 124 89     one case you choose
    python distill/test_student.py compare --n 25      rules vs model on random cases
    python distill/test_student.py stress             tricky cases that should be got right
    python distill/test_student.py gates              the full pass/fail suite

Every mode asks the same question the agent asks, through the same HTTP API the
dashboard will use. Nothing here is a special laboratory path.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distill.sampler import make_case, physically_valid
from distill.render import render_user, render_system
from agent.nodes.llm_analyzer import OUTPUT_SCHEMA, _extract_json
from guardrails.clinical_rules import determine_recommended_action
from vitals.schemas import AlertLevel

URL = "http://127.0.0.1:8099"
OK, BAD, WARN = "\033[92mOK\033[0m", "\033[91mFAIL\033[0m", "\033[93mWARN\033[0m"


def ask_model(case, url=URL, timeout=300.0):
    t0 = time.time()
    r = httpx.post(f"{url}/v1/chat/completions", timeout=timeout, json={
        "model": "vitals-v3",
        "messages": [{"role": "system", "content": render_system()},
                     {"role": "user", "content": render_user(case)}],
        "temperature": 0.0, "max_tokens": 500,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "assessment", "schema": OUTPUT_SCHEMA}}})
    r.raise_for_status()
    ch = r.json()["choices"][0]
    return _extract_json(ch["message"]["content"]), ch.get("finish_reason"), time.time() - t0


def judge(case, obj):
    """What SHOULD be true. The rules are the reference, never the model."""
    problems = []
    if obj is None:
        return ["model returned nothing parseable"]
    missing = [k for k in OUTPUT_SCHEMA["required"] if k not in obj]
    if missing: problems.append(f"missing fields: {missing}")
    if obj.get("risk_level") != case.tier:
        problems.append(f"risk_level {obj.get('risk_level')} but rules say {case.tier}")
    exp = determine_recommended_action(AlertLevel(case.tier))
    exp = exp.value if hasattr(exp, "value") else str(exp)
    if obj.get("recommended_action") != exp:
        problems.append(f"action {obj.get('recommended_action')} but rules say {exp}")
    c = obj.get("confidence")
    if not isinstance(c, (int, float)) or not 0 <= float(c) <= 1:
        problems.append(f"confidence out of range: {c}")
    # did it actually mention the patient's numbers?
    txt = str(obj.get("reasoning", ""))
    for v in (case.systolic_bp, case.heart_rate):
        if str(int(v)) not in txt:
            problems.append(f"reasoning never mentions {int(v)}")
            break
    return problems


def show(case, obj, secs, problems):
    print(f"  vitals     : SBP {case.systolic_bp}  DBP {case.diastolic_bp}  "
          f"HR {case.heart_rate}  SpO2 {case.spo2}")
    print(f"  RULES say  : {case.tier}   (NEWS2 total {case.news2['total_score']}, "
          f"qSOFA {case.qsofa['score']})")
    if obj:
        print(f"  MODEL says : {obj.get('risk_level')}   action={obj.get('recommended_action')}   "
              f"confidence={obj.get('confidence')}")
        print(f"  reasoning  : {str(obj.get('reasoning'))[:160]}")
    print(f"  time       : {secs:.1f}s")
    print(f"  verdict    : {OK if not problems else BAD}")
    for p in problems: print(f"               - {p}")


def cmd_smoke(a):
    case = make_case(120, 75, 72, 98, None)
    print("SMOKE TEST — one healthy patient\n")
    try:
        obj, fin, secs = ask_model(case, a.url)
    except Exception as e:
        print(f"  {BAD} cannot reach the server at {a.url}: {type(e).__name__}: {e}")
        print("  Is llama-server running?  curl {a.url}/health")
        return 1
    problems = judge(case, obj)
    if fin == "length":
        problems.append("output was CUT OFF (finish_reason=length) -- "
                        "server context per slot is too small; use -c 8192 --parallel 4")
    show(case, obj, secs, problems)
    return 0 if not problems else 1


def cmd_ask(a):
    if not physically_valid(a.sbp, a.dbp, a.hr, a.spo2):
        print(f"  {WARN} those values are outside the plausibility bounds; "
              f"the agent's input validator would reject them before the model.")
    case = make_case(a.sbp, a.dbp, a.hr, a.spo2, None)
    obj, fin, secs = ask_model(case, a.url)
    show(case, obj, secs, judge(case, obj))
    if a.full and obj:
        print("\n  full answer:"); print(json.dumps(obj, indent=4))
    return 0


def cmd_compare(a):
    rng = random.Random(a.seed)
    print(f"COMPARE — {a.n} random cases, rules vs model\n")
    print(f"  {'vitals':<28}{'rules':<10}{'model':<10}{'action':<8}{'sec':>6}  verdict")
    print("  " + "-" * 72)
    bad = 0
    for _ in range(a.n):
        while True:
            v = (rng.randint(70, 220), rng.randint(40, 120), rng.randint(35, 170), rng.randint(85, 100))
            if physically_valid(*v): break
        c = make_case(*v, None)
        try:
            obj, fin, secs = ask_model(c, a.url)
            probs = judge(c, obj)
        except Exception as e:
            obj, secs, probs = None, 0.0, [f"error {type(e).__name__}"]
        if probs: bad += 1
        act = "ok" if obj and not any("action" in p for p in probs) else "BAD"
        print(f"  {str(v):<28}{c.tier:<10}{str((obj or {}).get('risk_level')):<10}"
              f"{act:<8}{secs:>6.1f}  {OK if not probs else BAD}")
        for p in probs: print(f"      - {p}")
    print(f"\n  {a.n - bad}/{a.n} correct")
    return 0 if bad == 0 else 1


STRESS = [
    ((95, 70, 95, 96),  "qSOFA trap: NEWS2 only 3, but qSOFA 2 forces CRITICAL"),
    ((90, 70, 75, 98),  "boundary: SBP exactly 90 -> scores 3 -> CRITICAL"),
    ((91, 70, 75, 98),  "boundary: SBP 91 -> one higher -> NORMAL"),
    ((120, 70, 131, 98),"boundary: HR exactly 131 -> scores 3"),
    ((205, 70, 58, 96), "wide pulse pressure: HELD-OUT region, never trained on"),
    ((180, 60, 95, 94), "wide pulse pressure: HELD-OUT region"),
    ((120, 70, 75, 91), "SpO2 exactly 91 -> scores 3 -> CRITICAL"),
    ((120, 70, 75, 92), "SpO2 92 -> scores 2 only"),
]


def cmd_stress(a):
    print("STRESS TEST — cases chosen to be hard\n")
    bad = 0
    for vals, why in STRESS:
        c = make_case(*vals, None)
        try:
            obj, fin, secs = ask_model(c, a.url); probs = judge(c, obj)
        except Exception as e:
            obj, secs, probs = None, 0.0, [f"error {type(e).__name__}"]
        if probs: bad += 1
        print(f"  {str(vals):<24} {c.tier:<9} model={str((obj or {}).get('risk_level')):<9} "
              f"{OK if not probs else BAD}   {why}")
        for p in probs: print(f"      - {p}")
    print(f"\n  {len(STRESS)-bad}/{len(STRESS)} correct")
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=URL)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    p = sub.add_parser("ask"); p.add_argument("sbp", type=int); p.add_argument("dbp", type=int)
    p.add_argument("hr", type=int); p.add_argument("spo2", type=int)
    p.add_argument("--full", action="store_true")
    p = sub.add_parser("compare"); p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    sub.add_parser("stress")
    p = sub.add_parser("gates"); p.add_argument("--out", default="/tmp/gate_report.json")
    a = ap.parse_args()

    if a.cmd == "gates":
        import subprocess
        return subprocess.call([sys.executable, str(Path(__file__).parent / "evaluate_gguf.py"),
                                "--url", a.url, "--out", a.out])
    return {"smoke": cmd_smoke, "ask": cmd_ask, "compare": cmd_compare,
            "stress": cmd_stress}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
