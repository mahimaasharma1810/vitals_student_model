"""
Merge the LoRA adapter into the base weights and save a standalone model.

The adapter on its own is not deployable: it needs PEFT at runtime and cannot be
converted to GGUF. Merging folds the learned weights in so the result is an
ordinary model directory.
"""
from __future__ import annotations
import argparse, json, shutil, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/ssd_scratch/mahimakopalley/distill_models/student")
    ap.add_argument("--adapter", default="/ssd_scratch/mahimakopalley/distill_runs/v3_vitals/adapter")
    ap.add_argument("--out", default="/ssd_scratch/mahimakopalley/distill_models/v3_vitals_merged")
    a = ap.parse_args()

    t0 = time.time()
    print(f"base    : {a.base}")
    print(f"adapter : {a.adapter}")
    tok = AutoTokenizer.from_pretrained(a.adapter)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, a.adapter)
    print("merging...")
    model = model.merge_and_unload()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)

    # carry provenance forward so the deployed file can be traced to its training run
    src_cfg = Path(a.adapter).parent / "run_config.json"
    prov = json.loads(src_cfg.read_text()) if src_cfg.exists() else {}
    prov.update(merged_from_adapter=a.adapter, merged_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                merge_seconds=round(time.time() - t0))
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"saved -> {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
