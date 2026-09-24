#!/usr/bin/env python
"""
Benchmark the currently-served student model through its deployment API.

Measures, per case:
  TTFT              time from request sent to FIRST streamed token
  generation time   first token -> last token
  end-to-end        request sent -> response complete
  tokens generated  from the server's usage field (falls back to a count)

Samples the server process RSS and CPU% throughout, from /proc.

Quality is computed on the SAME responses, so performance and quality numbers
describe one run and not two different ones.

Power/energy is NOT measured here -- see the report for why.
"""
from __future__ import annotations
import argparse, json, statistics as st, sys, time
from pathlib import Path
from threading import Event, Thread

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distill.sampler import make_case, physically_valid
from distill.render import render_user, render_system
from agent.nodes.llm_analyzer import OUTPUT_SCHEMA, _extract_json
from guardrails.clinical_rules import determine_recommended_action
from vitals.schemas import AlertLevel

TIER_ORDER = {"NORMAL": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def server_pid():
    import subprocess
    out = subprocess.run(["pgrep", "-f", "llama-ser" "ver"], capture_output=True, text=True)
    pids = [int(x) for x in out.stdout.split()]
    return pids[0] if pids else None


class ResourceSampler(Thread):
    """Sample RSS and CPU jiffies of the server process while the run proceeds."""
    def __init__(self, pid, interval=0.5):
        super().__init__(daemon=True)
        self.pid, self.interval, self.stop = pid, interval, Event()
        self.rss_kb, self.cpu_pct = [], []

    def run(self):
        prev_cpu, prev_t = self._cpu(), time.time()
        while not self.stop.is_set():
            time.sleep(self.interval)
            try:
                with open(f"/proc/{self.pid}/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            self.rss_kb.append(int(line.split()[1])); break
                c, t = self._cpu(), time.time()
                hz = 100.0
                self.cpu_pct.append(100.0 * (c - prev_cpu) / hz / max(t - prev_t, 1e-6))
                prev_cpu, prev_t = c, t
            except Exception:
                break

    def _cpu(self):
        with open(f"/proc/{self.pid}/stat") as f:
            p = f.read().split()
        return int(p[13]) + int(p[14])        # utime + stime, in jiffies


def one_case(client, url, case, max_tokens, timeout):
    body = {"model": "vitals-v3",
            "messages": [{"role": "system", "content": render_system()},
                         {"role": "user", "content": render_user(case)}],
            "temperature": 0.0, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True},
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "assessment", "schema": OUTPUT_SCHEMA}}}
    t_send = time.perf_counter()
    ttft = None; chunks = []; usage = None
    with client.stream("POST", f"{url}/v1/chat/completions", json=body, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith("data: "): continue
            payload = line[6:]
            if payload.strip() == "[DONE]": break
            try: obj = json.loads(payload)
            except json.JSONDecodeError: continue
            if obj.get("usage"): usage = obj["usage"]
            ch = obj.get("choices") or []
            if ch:
                delta = (ch[0].get("delta") or {}).get("content")
                if delta:
                    if ttft is None: ttft = time.perf_counter() - t_send
                    chunks.append(delta)
    total = time.perf_counter() - t_send
    text = "".join(chunks)
    n_out = (usage or {}).get("completion_tokens")
    return dict(ttft_s=ttft, total_s=total, text=text,
                completion_tokens=n_out, prompt_tokens=(usage or {}).get("prompt_tokens"))


def quality(case, text):
    obj = _extract_json(text)
    q = dict(json_parsed=obj is not None, fields_ok=False, tier_match=False,
             action_match=False, fabrication=False, under_triage=False,
             over_triage=False, model_tier=None, cites_values=False)
    if obj is None: return q, obj
    q["fields_ok"] = not [k for k in OUTPUT_SCHEMA["required"] if k not in obj]
    mt = obj.get("risk_level"); q["model_tier"] = mt
    q["tier_match"] = mt == case.tier
    if mt in TIER_ORDER and case.tier in TIER_ORDER:
        q["under_triage"] = TIER_ORDER[mt] < TIER_ORDER[case.tier]
        q["over_triage"] = TIER_ORDER[mt] > TIER_ORDER[case.tier]
    exp = determine_recommended_action(AlertLevel(case.tier))
    exp = exp.value if hasattr(exp, "value") else str(exp)
    q["action_match"] = obj.get("recommended_action") == exp
    txt = " ".join(str(obj.get(k, "")) for k in ("reasoning", "limitations"))
    import re
    allowed = {int(n) for n in re.findall(r"\b\d{1,3}\b", render_user(case))} | set(range(0, 101)) \
              | {110, 120, 130, 140, 180, 200, 220}
    q["fabrication"] = bool({int(n) for n in re.findall(r"\b\d{1,3}\b", txt)} - allowed)
    q["cites_values"] = all(str(int(v)) in txt for v in (case.systolic_bp, case.heart_rate))
    return q, obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--out", default="/ssd_scratch/mahimakopalley/gguf_v3/benchmark_v3.json")
    ap.add_argument("--balanced", action="store_true",
                    help="equal cases per tier. Random draws are ~71% CRITICAL, and "
                         "CRITICAL bypasses the LLM in production, so an unbalanced "
                         "run says little about the cases the model actually sees.")
    a = ap.parse_args()

    import random
    rng = random.Random(a.seed)
    if a.balanced:
        per = a.n // 4
        buckets = {t: [] for t in ("NORMAL", "MEDIUM", "HIGH", "CRITICAL")}
        tries = 0
        while any(len(v) < per for v in buckets.values()) and tries < 5_000_000:
            tries += 1
            v = (rng.randint(60, 230), rng.randint(35, 125), rng.randint(30, 175), rng.randint(84, 100))
            if not physically_valid(*v): continue
            c = make_case(*v, None)
            if len(buckets[c.tier]) < per: buckets[c.tier].append(c)
        bench = [c for t in buckets for c in buckets[t]]
        rng.shuffle(bench)
        warm = bench[:a.warmup]
    else:
        cases = []
        while len(cases) < a.n + a.warmup:
            v = (rng.randint(60, 230), rng.randint(35, 125), rng.randint(30, 175), rng.randint(84, 100))
            if physically_valid(*v): cases.append(make_case(*v, None))
        warm, bench = cases[:a.warmup], cases[a.warmup:]

    pid = server_pid()
    print(f"server pid={pid}  warmup={len(warm)}  cases={len(bench)}", flush=True)

    with httpx.Client() as client:
        for c in warm:
            one_case(client, a.url, c, a.max_tokens, a.timeout)
        print("warmup done", flush=True)

        sampler = ResourceSampler(pid) if pid else None
        if sampler: sampler.start()
        rows = []
        t_run = time.perf_counter()
        for i, c in enumerate(bench, 1):
            r = one_case(client, a.url, c, a.max_tokens, a.timeout)
            q, obj = quality(c, r["text"])
            rows.append({**r, **q, "case": c.values(), "rule_tier": c.tier,
                         "news2": c.news2["total_score"], "qsofa": c.qsofa["score"]})
            del rows[-1]["text"]
            if i % 10 == 0:
                print(f"  {i}/{len(bench)}  ttft={r['ttft_s']:.2f}s total={r['total_s']:.2f}s", flush=True)
        wall = time.perf_counter() - t_run
        if sampler: sampler.stop.set(); sampler.join(timeout=3)

    ok = [r for r in rows if r["ttft_s"] is not None]
    tt = [r["ttft_s"] for r in ok]; tot = [r["total_s"] for r in ok]
    toks = [r["completion_tokens"] for r in ok if r["completion_tokens"]]
    gen_s = [r["total_s"] - r["ttft_s"] for r in ok]
    tps = [t / g for t, g in zip(toks, gen_s) if g > 0] if toks else []
    pct = lambda xs, p: st.quantiles(xs, n=100)[p-1] if len(xs) > 2 else (max(xs) if xs else 0)

    res = dict(
        n_cases=len(rows), wall_seconds=round(wall, 1),
        latency_s=dict(median=round(st.median(tot),2), mean=round(st.mean(tot),2),
                       min=round(min(tot),2), max=round(max(tot),2), p95=round(pct(tot,95),2)),
        ttft_s=dict(median=round(st.median(tt),2), mean=round(st.mean(tt),2),
                    min=round(min(tt),2), max=round(max(tt),2), p95=round(pct(tt,95),2)),
        completion_tokens=dict(median=st.median(toks) if toks else None,
                               mean=round(st.mean(toks),1) if toks else None,
                               min=min(toks) if toks else None, max=max(toks) if toks else None),
        prompt_tokens=dict(median=st.median([r["prompt_tokens"] for r in ok if r["prompt_tokens"]])
                           if any(r["prompt_tokens"] for r in ok) else None),
        gen_tokens_per_s=dict(median=round(st.median(tps),2) if tps else None,
                              mean=round(st.mean(tps),2) if tps else None,
                              min=round(min(tps),2) if tps else None,
                              max=round(max(tps),2) if tps else None),
        rss_kb=dict(peak=max(sampler.rss_kb) if sampler and sampler.rss_kb else None,
                    median=st.median(sampler.rss_kb) if sampler and sampler.rss_kb else None),
        server_cpu_pct=dict(median=round(st.median(sampler.cpu_pct),1) if sampler and sampler.cpu_pct else None,
                            peak=round(max(sampler.cpu_pct),1) if sampler and sampler.cpu_pct else None),
        quality=dict(
            json_parsed=sum(r["json_parsed"] for r in rows),
            fields_ok=sum(r["fields_ok"] for r in rows),
            tier_match=sum(r["tier_match"] for r in rows),
            action_match=sum(r["action_match"] for r in rows),
            fabrication_free=sum(not r["fabrication"] for r in rows),
            cites_values=sum(r["cites_values"] for r in rows),
            under_triage=sum(r["under_triage"] for r in rows),
            over_triage=sum(r["over_triage"] for r in rows)),
        tier_mix={t: sum(1 for r in rows if r["rule_tier"]==t) for t in ("NORMAL","MEDIUM","HIGH","CRITICAL")},
        rows=rows)
    Path(a.out).write_text(json.dumps(res, indent=2, default=str))
    n = len(rows)
    print(f"\n=== RESULTS (n={n}) ===")
    print(f"  latency  median {res['latency_s']['median']}s  p95 {res['latency_s']['p95']}s  max {res['latency_s']['max']}s")
    print(f"  TTFT     median {res['ttft_s']['median']}s  p95 {res['ttft_s']['p95']}s")
    print(f"  gen      median {res['gen_tokens_per_s']['median']} tok/s")
    print(f"  tokens   prompt~{res['prompt_tokens']['median']}  completion median {res['completion_tokens']['median']}")
    print(f"  RSS peak {res['rss_kb']['peak']} kB   server CPU median {res['server_cpu_pct']['median']}%")
    for k,v in res["quality"].items(): print(f"  {k:<18}{v}/{n}  ({100*v/n:.1f}%)")
    print(f"  tier mix {res['tier_mix']}")
    print(f"report -> {a.out}")


if __name__ == "__main__":
    main()
