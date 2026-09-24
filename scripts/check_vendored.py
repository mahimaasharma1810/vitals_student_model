#!/usr/bin/env python
"""
Verify the vendored files still match the parent agent repository.

Run this before training, and before trusting an evaluation result. If a
critical file has drifted, any model trained here is invalid -- it would have
learned a prompt or rubric the agent no longer uses.

    python scripts/check_vendored.py --upstream /path/to/MedGemma-Agent
"""
from __future__ import annotations
import argparse, hashlib, sys
from pathlib import Path

CRITICAL = {"agent/prompts.py", "agent/nodes/llm_analyzer.py",
            "guardrails/clinical_rules.py", "guardrails/thresholds.yaml"}
VENDORED = sorted(CRITICAL | {"agent/state.py", "guardrails/input_guardrails.py",
                              "guardrails/output_guardrails.py", "vitals/schemas.py",
                              "config/settings.py", "tests/test_guardrails.py"})


def sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True, help="path to the MedGemma-Agent checkout")
    a = ap.parse_args()
    up, here = Path(a.upstream), Path(__file__).resolve().parents[1]
    drift, missing = [], []
    print(f"{'file':<42}{'status'}")
    print("-" * 62)
    for rel in VENDORED:
        h_here, h_up = sha(here / rel), sha(up / rel)
        if h_up is None:
            missing.append(rel); status = "MISSING UPSTREAM"
        elif h_here == h_up:
            status = "match"
        else:
            drift.append(rel)
            status = "DRIFTED  <-- CRITICAL" if rel in CRITICAL else "drifted"
        print(f"{rel:<42}{status}")
    print("-" * 62)
    crit = [d for d in drift if d in CRITICAL]
    if crit:
        print(f"\nFAIL: {len(crit)} CRITICAL file(s) have drifted: {crit}")
        print("Any model trained from this repo is INVALID until these are re-synced.")
        return 2
    if drift or missing:
        print(f"\nWARN: {len(drift)} drifted, {len(missing)} missing upstream (none critical)")
        return 1
    print("\nOK: all vendored files match upstream")
    return 0


if __name__ == "__main__":
    sys.exit(main())
