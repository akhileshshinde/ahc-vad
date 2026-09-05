#!/bin/bash
# Upload the extracted test-window frames as a Kaggle Dataset for inference.
set -e
DIR="${1:-/home/extra_space/akhilesh/ahc_test_windows}"
USER="akhileshshinde477"
SLUG="ahc-vad-test-windows"

cat > "$DIR/dataset-metadata.json" <<EOM
{
  "title": "AHC VAD Test Windows",
  "id": "${USER}/${SLUG}",
  "licenses": [{"name": "other"}]
}
EOM

du -sh "$DIR"
find "$DIR" -type f | wc -l

if conda run -n vad kaggle datasets list -m --search "$SLUG" 2>&1 | grep -q "$USER/$SLUG"; then
  echo "creating new version..."
  conda run -n vad kaggle datasets version -p "$DIR" -m "test windows $(date -u +%H:%M)" -r zip
else
  echo "creating dataset..."
  conda run -n vad kaggle datasets create -p "$DIR" -r zip
fi
