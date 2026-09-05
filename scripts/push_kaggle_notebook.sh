#!/bin/bash
# Push the training notebook to Kaggle and kick off the run.
set -e
cd /home/akhilesh/ahc-hackathon/notebooks
conda run -n vad kaggle kernels push -p .
echo "--- pushed. poll status with: ---"
echo 'conda run -n vad kaggle kernels status akhileshshinde477/fork-of-ahc-vad-qwen2-5-vl-train'
