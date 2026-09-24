"""
Teacher generation for the vitals domain.

Produces training rows in EXACTLY the format the agent uses at runtime
(see distill/render.py for why that matters).

Resumable: rows already present in accepted.jsonl / rejects.jsonl are skipped,
so the run can be stopped and restarted without losing work or duplicating it.
"""
from __future__ import annotations
import argparse, json, os, sys, time, warnings
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distill.sampler import sample
from distill.render import render_user, render_system, NO_EHR
from distill.gate import check
from distill.regions import DEFAULT_HELD_OUT
from agent.nodes.llm_analyzer import _extract_json

warnings.filterwarnings("ignore")
TEACHER = "/ssd_scratch/mahimakopalley/distill_models/teacher"


def load_done(*paths) -> set[str]:
    done = set()
    for p in paths:
        if Path(p).exists():
            for line in open(p):
                try: done.add(json.loads(line)["case_id"])
                except Exception: pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/ssd_scratch/mahimakopalley/distill_data/vitals_v3")
    ap.add_argument("--per-tier", type=int, default=400)
    ap.add_argument("--held-out-per-tier", type=int, default=40)
    ap.add_argument("--region", default=DEFAULT_HELD_OUT)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: cap total cases")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f_acc, f_rej = out / "accepted.jsonl", out / "rejects.jsonl"

    train, held = sample(a.per_tier, a.seed, a.region, a.held_out_per_tier,
                         max_draws=20_000_000)
    cases = [("train", c) for c in train] + [("heldout", c) for c in held]
    if a.limit:
        cases = cases[:a.limit]

    done = load_done(f_acc, f_rej)
    todo = [(s, c) for s, c in cases if c.case_id not in done]
    print(f"cases total={len(cases)}  already done={len(done)}  to generate={len(todo)}", flush=True)
    if not todo:
        print("nothing to do"); return

    tok = AutoTokenizer.from_pretrained(TEACHER)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER, dtype=torch.bfloat16, device_map="cuda:0").eval()
    SYS = render_system()

    n_acc = n_rej = 0
    t_start = time.time()
    with open(f_acc, "a") as fa, open(f_rej, "a") as fr:
        for i in range(0, len(todo), a.batch_size):
            chunk = todo[i:i + a.batch_size]
            texts = [tok.apply_chat_template(
                [{"role": "system", "content": SYS},
                 {"role": "user", "content": render_user(c)}],
                add_generation_prompt=True, tokenize=False) for _, c in chunk]
            enc = tok(texts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda:0")
            n_in = enc["input_ids"].shape[1]
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=a.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            for (split, c), o in zip(chunk, gen):
                raw = tok.decode(o[n_in:], skip_special_tokens=True)
                up = render_user(c)
                # A single unexpected row must never abort a multi-hour run.
                # (It did once: check() had an early return with 2 values while
                # the caller unpacked 3, and the first unparseable teacher reply
                # killed the job at case 384.)
                try:
                    obj = _extract_json(raw)
                    ok, reasons, notes = check(c, obj, raw, up)
                except Exception as e:
                    obj, ok, notes = None, False, []
                    reasons = [f"gate_error:{type(e).__name__}:{str(e)[:80]}"]
                row = dict(domain="vitals", split=split, tier=c.tier,
                           system=SYS, user=up,
                           assistant=json.dumps(obj, ensure_ascii=False) if obj else raw,
                           case=dict(systolic_bp=c.systolic_bp, diastolic_bp=c.diastolic_bp,
                                     heart_rate=c.heart_rate, spo2=c.spo2),
                           news2=c.news2, qsofa=c.qsofa,
                           region=c.region, case_id=c.case_id, ehr_context=NO_EHR,
                           notes=notes)
                if ok:
                    fa.write(json.dumps(row, ensure_ascii=False) + "\n"); n_acc += 1
                else:
                    row["reject_reasons"] = reasons
                    fr.write(json.dumps(row, ensure_ascii=False) + "\n"); n_rej += 1
            fa.flush(); fr.flush()
            done_n = n_acc + n_rej
            rate = (time.time() - t_start) / max(done_n, 1)
            eta = rate * (len(todo) - done_n) / 60
            print(f"  {done_n}/{len(todo)}  accepted={n_acc} rejected={n_rej} "
                  f"({100*n_acc/max(done_n,1):.1f}% accept)  {rate:.1f}s/case  ETA {eta:.0f} min",
                  flush=True)

    manifest = dict(domain="vitals", generator="physiology_sampler_v3",
                    seed=a.seed, per_tier=a.per_tier,
                    held_out_region=a.region, held_out_per_tier=a.held_out_per_tier,
                    ehr_policy="option_a_no_ehr", teacher=TEACHER,
                    batch_size=a.batch_size, max_new_tokens=a.max_new_tokens,
                    accepted=n_acc, rejected=n_rej,
                    accept_rate=round(n_acc / max(n_acc + n_rej, 1), 4),
                    prompt_source="agent/prompts.py", schema_source="agent/nodes/llm_analyzer.py",
                    generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
