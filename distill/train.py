"""
LoRA training for the vitals student.

Two things this script guarantees, and asserts rather than assumes:

1. HELD-OUT ROWS NEVER ENTER TRAINING.
   The whole value of the hidden region is that the model has not seen it. A
   single leaked row destroys the only honest generalisation measurement we
   have, so leakage is checked twice (by split label AND by region predicate)
   and the run aborts if either finds anything.

2. LOSS IS COMPUTED ON THE ANSWER ONLY.
   The prompt is masked out with -100. Training on the prompt tokens would teach
   the model to reproduce the question, which wastes capacity on a 0.5B model.

The in-run "val" split is for watching over/under-fitting ONLY. It is drawn from
the same generator as training, so it is in-distribution and says nothing about
generalisation. Quality claims come from distill/evaluate.py, which uses the
held-out region, gold and boundary sets scored separately.
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys, time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distill.regions import get as get_region

STUDENT = "/ssd_scratch/mahimakopalley/distill_models/student"
IGNORE = -100


class ChatDataset(Dataset):
    """system+user -> assistant, with the prompt masked out of the loss."""

    def __init__(self, rows, tok, max_len):
        self.rows, self.tok, self.max_len = rows, tok, max_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        prompt = self.tok.apply_chat_template(
            [{"role": "system", "content": r["system"]},
             {"role": "user", "content": r["user"]}],
            add_generation_prompt=True, tokenize=False)
        full = prompt + r["assistant"] + self.tok.eos_token

        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = self.tok(full, add_special_tokens=False)["input_ids"][: self.max_len]
        labels = list(f_ids)
        for j in range(min(len(p_ids), len(labels))):
            labels[j] = IGNORE                      # <- answer-only loss
        return {"input_ids": f_ids, "labels": labels}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids, lab, att = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * k)
        lab.append(b["labels"] + [IGNORE] * k)
        att.append([1] * len(b["input_ids"]) + [0] * k)
    return {"input_ids": torch.tensor(ids),
            "labels": torch.tensor(lab),
            "attention_mask": torch.tensor(att)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/ssd_scratch/mahimakopalley/distill_data/vitals_v3/accepted.jsonl")
    ap.add_argument("--out", default="/ssd_scratch/mahimakopalley/distill_runs/v3_vitals")
    ap.add_argument("--region", default="wide_pulse_pressure")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--resume", action="store_true",
                    help="resume from the newest checkpoint in --out")
    a = ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed)
    rows = [json.loads(l) for l in open(a.data)]
    print(f"loaded {len(rows):,} accepted rows")

    # ---- GUARD 1: split label ----------------------------------------------
    train_rows = [r for r in rows if r.get("split") == "train"]
    held_rows  = [r for r in rows if r.get("split") == "heldout"]
    print(f"  split=train   : {len(train_rows):,}")
    print(f"  split=heldout : {len(held_rows):,}  (excluded from training)")

    # ---- GUARD 2: re-check with the region predicate itself -----------------
    R = get_region(a.region)
    leaked = [r for r in train_rows
              if R.contains(r["case"]["systolic_bp"], r["case"]["diastolic_bp"],
                            r["case"]["heart_rate"], r["case"]["spo2"])]
    if leaked:
        raise SystemExit(
            f"ABORT: {len(leaked)} training rows fall inside the held-out region "
            f"'{a.region}'. The generalisation test would be invalid. "
            f"First offender: {leaked[0]['case']}")
    print(f"  leakage check : 0 training rows inside '{a.region}'  OK")
    if not train_rows:
        raise SystemExit("ABORT: no training rows")

    random.shuffle(train_rows)
    n_val = max(1, int(len(train_rows) * a.val_frac))
    val, tr = train_rows[:n_val], train_rows[n_val:]
    print(f"  train={len(tr):,}  in-distribution val={len(val):,} "
          f"(loss monitoring only, NOT a quality measure)")

    tok = AutoTokenizer.from_pretrained(STUDENT)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT, dtype=torch.bfloat16, device_map="cuda:0")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"]))
    model.print_trainable_parameters()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # transformers 5.x dropped warmup_ratio; compute the equivalent in steps.
    eff_batch = a.batch_size * a.grad_accum
    total_steps = max(1, int((len(tr) / eff_batch) * a.epochs))
    warmup_steps = max(1, int(0.03 * total_steps))
    print(f"  steps: {total_steps} total, {warmup_steps} warmup (3%)")
    args = TrainingArguments(
        output_dir=str(out), num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=warmup_steps,
        logging_steps=10, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=2, bf16=True, seed=a.seed,
        report_to=[], remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=ChatDataset(tr, tok, a.max_len),
        eval_dataset=ChatDataset(val, tok, a.max_len),
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    t0 = time.time()
    ckpts = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if a.resume and ckpts:
        print(f"resuming from {ckpts[-1].name}")
        trainer.train(resume_from_checkpoint=str(ckpts[-1]))
    else:
        trainer.train()
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")

    fingerprint = hashlib.sha256(open(a.data, "rb").read()).hexdigest()[:16]
    (out / "run_config.json").write_text(json.dumps(dict(
        domain="vitals", base_model=STUDENT, data=a.data,
        data_sha256_16=fingerprint, accepted_rows=len(rows),
        train_rows=len(tr), indist_val_rows=len(val),
        held_out_rows_excluded=len(held_rows), held_out_region=a.region,
        leakage_rows=0, epochs=a.epochs, lr=a.lr,
        batch_size=a.batch_size, grad_accum=a.grad_accum,
        lora_r=a.lora_r, lora_alpha=a.lora_alpha, max_len=a.max_len,
        seed=a.seed, loss="answer_only_prompt_masked",
        train_seconds=round(time.time() - t0),
        note="in-distribution val is for loss monitoring only; quality comes "
             "from distill/evaluate.py on held-out/gold/boundary scored separately",
    ), indent=2))
    print(f"\nsaved adapter -> {out/'adapter'}")


if __name__ == "__main__":
    main()
