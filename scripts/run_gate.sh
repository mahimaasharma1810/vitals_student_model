#!/usr/bin/env bash
# Serve one GGUF and run the identical gate suite against it.
# Usage: scripts/run_gate.sh <QUANT>   e.g. Q5_K_M
set -u
Q="$1"
GGUF_DIR=${GGUF_DIR:-/ssd_scratch/mahimakopalley/gguf_v3}
LLAMA_SERVER=${LLAMA_SERVER:-/ssd_scratch/mahimakopalley/llama.cpp/build/bin/llama-server}
PYTHON=${PYTHON:-/home2/mahimakopalley/projects/.venv/bin/python}
REPO=$(cd "$(dirname "$0")/.." && pwd)
G=$GGUF_DIR/vitals-v3-$Q.gguf
OUT=$GGUF_DIR/eval_$Q.json
LOG=/tmp/gate_$Q.log
SRV=/tmp/llama_server_$Q.log
pkill -9 -f "llama-serv""er" 2>/dev/null; sleep 3
setsid nohup "$LLAMA_SERVER" \
  -m "$G" --host 127.0.0.1 --port 8099 -c 8192 --parallel 4 > "$SRV" 2>&1 < /dev/null &
for i in $(seq 1 60); do
  [ "$(curl -s --max-time 5 http://127.0.0.1:8099/health 2>/dev/null)" = '{"status":"ok"}' ] && break
  sleep 5
done
echo "[$Q] server up, running gates" > "$LOG"
cd "$REPO"
"$PYTHON" distill/evaluate_gguf.py \
  --out "$OUT" --indist-n 40 --concurrency 4 >> "$LOG" 2>&1
echo "[$Q] done rc=$?" >> "$LOG"
