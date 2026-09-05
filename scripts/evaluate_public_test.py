#!/usr/bin/env python3
"""
Evaluate the deployed local pipeline (llama-server + vad_pipeline classify
call) against the AHC public test set (test/videos + test/ground_truth.csv),
which ships its own ground truth precisely so teams can validate before the
private evaluation.

For each test video, samples frames across the whole video (or the event's
window when start/end are given) the same way training examples were built,
classifies once, and compares to the ground-truth class_name / is_anomaly.

Usage:
  python evaluate_public_test.py --test_dir /home/extra_space/akhilesh/ahc_dataset/test \
      --server http://127.0.0.1:8080
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))
from vad_pipeline import classify_window, extract_frame_jpeg_b64, ffprobe_duration  # noqa: E402


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--frames_per_video", type=int, default=4)
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    videos_meta = {r["video_id"]: r for r in load_csv(test_dir / "videos.csv")}
    gt_rows = load_csv(test_dir / "ground_truth.csv")

    correct = 0
    anomaly_correct = 0
    total = 0
    confusion = {}
    results = []

    for row in gt_rows:
        vid = row["video_id"]
        filename = videos_meta.get(vid, {}).get("filename", f"videos/{vid}.mp4")
        video_path = test_dir / filename
        if not video_path.exists():
            print(f"SKIP missing video: {video_path}")
            continue

        start_s, end_s = row.get("start_time_sec", "").strip(), row.get("end_time_sec", "").strip()
        if start_s and end_s:
            start, end = float(start_s), float(end_s)
        else:
            start, end = 0.0, ffprobe_duration(str(video_path))

        n = args.frames_per_video
        times = [start + (end - start) * i / max(n - 1, 1) for i in range(n)] if n > 1 else [(start + end) / 2]
        frames_b64 = [b for b in (extract_frame_jpeg_b64(str(video_path), t) for t in times) if b]
        if not frames_b64:
            print(f"SKIP no frames extracted: {video_path}")
            continue

        try:
            result, raw = classify_window(args.server, frames_b64)
        except Exception as e:
            print(f"ERROR {vid}: {e}")
            continue

        gt_class = row["class_name"]
        gt_anom = row.get("is_anomaly", "").strip().lower() in ("1", "true", "yes")
        pred_class = result.get("class_name")
        pred_anom = result.get("is_anomaly")

        total += 1
        correct += int(pred_class == gt_class)
        anomaly_correct += int(pred_anom == gt_anom)
        confusion.setdefault(gt_class, {}).setdefault(pred_class, 0)
        confusion[gt_class][pred_class] += 1
        results.append({"video_id": vid, "gt_class": gt_class, "pred_class": pred_class,
                         "gt_is_anomaly": gt_anom, "pred_is_anomaly": pred_anom})
        print(f"{vid}: gt={gt_class} pred={pred_class} {'OK' if pred_class==gt_class else 'X'}")

    print(f"\n=== class accuracy: {correct}/{total} = {correct/max(total,1):.3f} ===")
    print(f"=== anomaly (binary) accuracy: {anomaly_correct}/{total} = {anomaly_correct/max(total,1):.3f} ===")
    print(json.dumps(confusion, indent=2))

    out_path = Path(__file__).resolve().parent.parent / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
