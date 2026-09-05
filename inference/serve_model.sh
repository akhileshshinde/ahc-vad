#!/bin/bash
# Launch llama-server locally with GPU offload for the fine-tuned Qwen2.5-VL-3B GGUF.
# All inference happens on-device (RTX 3050) -- no cloud calls at runtime.
set -e

MODEL_DIR="${MODEL_DIR:-/home/akhilesh/ahc-hackathon/model}"
MODEL="${MODEL:-$MODEL_DIR/qwen25vl3b-vad-Q4_K_M.gguf}"
MMPROJ="${MMPROJ:-$MODEL_DIR/mmproj-qwen25vl3b-vad-f16.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-/home/extra_space/akhilesh/llama.cpp/build/bin/llama-server}"
PORT="${PORT:-8080}"

if [ ! -f "$MODEL" ]; then
  echo "Model not found at $MODEL — download the trained GGUF from Kaggle first." >&2
  exit 1
fi
if [ ! -f "$MMPROJ" ]; then
  echo "mmproj not found at $MMPROJ — download it from Kaggle first." >&2
  exit 1
fi
if [ ! -x "$LLAMA_SERVER" ]; then
  echo "llama-server binary not found/executable at $LLAMA_SERVER — build llama.cpp with GGML_CUDA=ON first." >&2
  exit 1
fi

exec "$LLAMA_SERVER" \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  -ngl 999 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --ctx-size 4096 \
  --temp 0.1
