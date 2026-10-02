#!/usr/bin/env bash
# Run the gate suite on Q5_K_M then Q8_0. Resumable: skips any that already
# produced a report. Safe to re-run from cron.
set -u
GGUF_DIR=${GGUF_DIR:-/ssd_scratch/mahimakopalley/gguf_v3}
HERE=$(cd "$(dirname "$0")" && pwd)
LOCK=$GGUF_DIR/.gate.lock
[ -f "$LOCK/pid" ] && kill -0 "$(cat $LOCK/pid 2>/dev/null)" 2>/dev/null && { echo "already running"; exit 3; }
rm -rf "$LOCK"; mkdir -p "$LOCK"; echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT INT TERM
for Q in Q5_K_M Q8_0; do
  OUT=$GGUF_DIR/eval_$Q.json
  [ -f "$OUT" ] && { echo "[$Q] already done"; continue; }
  GGUF_DIR=$GGUF_DIR "$HERE/run_gate.sh" "$Q"
done
touch "$GGUF_DIR/.gates_complete"
echo "ALL GATES DONE"
