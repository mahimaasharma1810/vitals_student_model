#!/usr/bin/env bash
# Keep the shipped vitals student served. No-op if already healthy.
# Exists because setsid/nohup alone do not survive session teardown on this node.
GGUF=${GGUF:-$(cd "$(dirname "$0")/.." && pwd)/model/vitals-v3-Q5_K_M.gguf}
LLAMA_SERVER=${LLAMA_SERVER:-/ssd_scratch/mahimakopalley/llama.cpp/build/bin/llama-server}
URL=http://127.0.0.1:8099
[ "$(curl -s --max-time 5 $URL/health 2>/dev/null)" = '{"status":"ok"}' ] && exit 0
"$LLAMA_SERVER" \
  -m "$GGUF" \
  --host 127.0.0.1 --port 8099 -c 8192 --parallel 4 \
  >> /tmp/llama_server.log 2>&1 &
