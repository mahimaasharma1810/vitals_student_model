"""
MEDIUM-only top-up generation (Option B: OLD prompt, unchanged).

Why this exists: the teacher disagrees with the deterministic MEDIUM boundary
about 38.6% of the time, so the first run produced only 270 accepted MEDIUM rows
against 438 CRITICAL. This tops MEDIUM up to parity WITHOUT changing the prompt,
which would invalidate the already-trained student.

Guarantees:
  * uses agent/prompts.py unchanged, via distill/render.py -- no drift
  * skips any case_id already generated, so no duplicates
  * excludes the held-out region, so the generalisation test stays valid
  * no qSOFA ratio is imposed; whatever lands, lands, and is reported
Output goes to its OWN directory. Merging into the training corpus is a
separate, explicit step.
"""
from __future__ import annotations
import argparse, json, random, sys, time, warnings
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distill.sampler import _draw, physically_valid, make_case
from distill.regions import get as get_region, DEFAULT_HELD_OUT
from distill.render import render_user, render_system
from distill.gate import check
from agent.nodes.llm_analyzer import _extract_json

warnings.filterwarnings("ignore")
TEACHER = "/ssd_scratch/mahimakopalley/distill_models/teacher"
EXISTING = "/ssd_scratch/mahimakopalley/distill_data/vitals_v3"


def existing_ids() -> set[str]:
    s = set()
    for f in ("accepted.jsonl", "rejects.jsonl"):
        p = Path(EXISTING) / f
        if p.exists():
            for line in open(p):
                try: s.add(json.loads(line)["case_id"])
                except Exception: pass
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/ssd_scratch/mahimakopalley/distill_data/vitals_v3_medium_topup")
    ap.add_argument("--target-accepted", type=int, default=170,
                    help="new ACCEPTED MEDIUM rows wanted (270 + 170 = 440)")
    ap.add_argument("--region", default=DEFAULT_HELD_OUT)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--max-draws", type=int, default=3_000_000)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f_acc, f_rej = out / "accepted.jsonl", out / "rejects.jsonl"
    already = existing_ids()
    for f in (f_acc, f_rej):
        if f.exists():
            for line in open(f):
                try: already.add(json.loads(line)["case_id"])
                except Exception: pass
    have = sum(1 for _ in open(f_acc)) if f_acc.exists() else 0
    print(f"existing case_ids to avoid: {len(already):,}")
    print(f"already accepted in this top-up: {have}  | target {a.target_accepted}", flush=True)
    if have >= a.target_accepted:
        print("target already met"); return

    region = get_region(a.region)
    rng = random.Random(a.seed)

    tok = AutoTokenizer.from_pretrained(TEACHER); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.bfloat16,
                                                 device_map="cuda:0").eval()
    SYS = render_system()
    n_acc, n_rej, draws = have, 0, 0
    t0 = time.time()

    with open(f_acc, "a") as fa, open(f_rej, "a") as fr:
        while n_acc < a.target_accepted and draws < a.max_draws:
            # collect a batch of NEW, MEDIUM, non-held-out cases
            batch = []
            while len(batch) < a.batch_size and draws < a.max_draws:
                draws += 1
                v = (_draw(rng, "systolic_bp"), _draw(rng, "diastolic_bp"),
                     _draw(rng, "heart_rate"), _draw(rng, "spo2"))
                if not physically_valid(*v): continue
                c = make_case(*v, region)
                if c.tier != "MEDIUM" or c.region is not None: continue
                if c.case_id in already: continue
                already.add(c.case_id); batch.append(c)
            if not batch: break

            texts = [tok.apply_chat_template(
                [{"role": "system", "content": SYS},
                 {"role": "user", "content": render_user(c)}],
                add_generation_prompt=True, tokenize=False) for c in batch]
            enc = tok(texts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda:0")
            n_in = enc["input_ids"].shape[1]
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=a.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            for c, o in zip(batch, gen):
                raw = tok.decode(o[n_in:], skip_special_tokens=True)
                up = render_user(c)
                try:
                    obj = _extract_json(raw)
                    ok, reasons, notes = check(c, obj, raw, up)
                except Exception as e:
                    obj, ok, notes, reasons = None, False, [], [f"gate_error:{type(e).__name__}"]
                row = dict(domain="vitals", split="train", tier=c.tier, system=SYS, user=up,
                           assistant=json.dumps(obj, ensure_ascii=False) if obj else raw,
                           case=dict(systolic_bp=c.systolic_bp, diastolic_bp=c.diastolic_bp,
                                     heart_rate=c.heart_rate, spo2=c.spo2),
                           news2=c.news2, qsofa=c.qsofa, region=c.region,
                           case_id=c.case_id, notes=notes, source="medium_topup")
                if ok: fa.write(json.dumps(row, ensure_ascii=False) + "\n"); n_acc += 1
                else:
                    row["reject_reasons"] = reasons
                    fr.write(json.dumps(row, ensure_ascii=False) + "\n"); n_rej += 1
            fa.flush(); fr.flush()
            done = (n_acc - have) + n_rej
            rate = (time.time() - t0) / max(done, 1)
            print(f"  accepted {n_acc}/{a.target_accepted}  rejected {n_rej}  "
                  f"({100*(n_acc-have)/max(done,1):.1f}% accept)  {rate:.1f}s/case  "
                  f"ETA {rate*max(a.target_accepted-n_acc,0)/0.614/60:.0f} min", flush=True)

    (out / "manifest.json").write_text(json.dumps(dict(
        domain="vitals", kind="medium_topup", prompt="UNCHANGED (agent/prompts.py)",
        seed=a.seed, target_accepted=a.target_accepted, accepted=n_acc, rejected=n_rej,
        draws=draws, held_out_region_excluded=a.region,
        accept_rate=round((n_acc - have) / max((n_acc - have) + n_rej, 1), 4),
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), indent=2))
    print(f"\ndone: accepted={n_acc} rejected={n_rej}")


if __name__ == "__main__":
    main()
