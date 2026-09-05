#!/bin/bash
# Package the extracted-frames dataset and push it to Kaggle as a Dataset,
# so the training notebook can attach it as an input.
set -e

FRAMES_DIR="${1:-/home/extra_space/akhilesh/ahc_frames}"
KAGGLE_USER="akhileshshinde477"
SLUG="ahc-vad-frames"

if [ ! -f "$FRAMES_DIR/train_manifest.jsonl" ]; then
  echo "ERROR: $FRAMES_DIR/train_manifest.jsonl not found. Run build_dataset.py first." >&2
  exit 1
fi

cat > "$FRAMES_DIR/dataset-metadata.json" <<EOF
{
  "title": "AHC VAD Frames",
  "id": "${KAGGLE_USER}/${SLUG}",
  "licenses": [{"name": "other"}]
}
EOF

echo "Contents of $FRAMES_DIR:"
du -sh "$FRAMES_DIR"
find "$FRAMES_DIR" -type f | wc -l

conda run -n vad kaggle datasets list -m --search "$SLUG" 2>&1 | grep -q "$KAGGLE_USER/$SLUG" && EXISTS=1 || EXISTS=0

if [ "$EXISTS" = "1" ]; then
  echo "Dataset exists, creating a new version..."
  conda run -n vad kaggle datasets version -p "$FRAMES_DIR" -m "update $(date -u +%Y-%m-%dT%H:%M:%SZ)" -r zip
else
  echo "Creating new dataset..."
  conda run -n vad kaggle datasets create -p "$FRAMES_DIR" -r zip
fi
