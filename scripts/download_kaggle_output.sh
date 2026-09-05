#!/bin/bash
# Pull the trained GGUF + mmproj + LoRA adapter back from the finished Kaggle kernel.
set -e

KERNEL="akhileshshinde477/fork-of-ahc-vad-qwen2-5-vl-train"
OUT_DIR="${1:-/home/akhilesh/ahc-hackathon/model}"

mkdir -p "$OUT_DIR"
conda run -n vad kaggle kernels output "$KERNEL" -p "$OUT_DIR"
echo "--- downloaded to $OUT_DIR ---"
ls -lh "$OUT_DIR"
