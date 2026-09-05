#!/usr/bin/env python3
"""
Extract sampled frames from the AHC video anomaly dataset and build a JSONL
manifest suitable for Qwen2.5-VL vision fine-tuning (Unsloth).

For each ground_truth.csv row:
  - anomaly with a timed window (level 2/3): sample N frames evenly across [start,end]
  - anomaly with no window (level 1, whole clip is the event): sample N frames across whole video
  - normal: sample a smaller number of frames evenly across the whole video

Frames are extracted with ffmpeg (precise seek + scale), not opencv, since
opencv isn't installed locally and ffmpeg/ffprobe already are.

Output:
  <out_dir>/frames/<class_name>/<video_id>_<idx>.jpg
  <out_dir>/train_manifest.jsonl   (one JSON object per event/example)
  <out_dir>/val_manifest.jsonl     (held-out videos, per-class stratified)
"""
import argparse
import csv
import json
import random
import subprocess
from pathlib import Path

CLASSES = [
    "normal",
    "traffic_accident",
    "traffic_congestion",
    "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic",
    "wrong_way_driving",
    "road_spill_or_debris",
    "waterlogging_or_flood",
    "fire",
    "smoke",
    "fighting_or_violence",
    "loitering_or_suspicious_presence",
]

MAX_SIDE = 672  # cap longest edge to keep image-token count sane on a T4


def ffprobe_duration(video_path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def extract_frame(video_path: Path, t: float, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scale = f"scale='min({MAX_SIDE},iw)':'min({MAX_SIDE},ih)':force_original_aspect_ratio=decrease"
    cmd = [
        "ffmpeg", "-y", "-ss", f"{max(t, 0):.3f}", "-i", str(video_path),
        "-frames:v", "1", "-q:v", "3", "-vf", scale, str(out_path),
    ]
    res = subprocess.run(cmd, capture_output=True)
    return out_path.exists() and out_path.stat().st_size > 0


def sample_times(start: float, end: float, n: int) -> list:
    if n <= 1:
        return [(start + end) / 2]
    span = end - start
    if span <= 0:
        return [start] * n
    return [start + span * i / (n - 1) for i in range(n)]


def load_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def process_split(root: Path, out_dir: Path, frames_per_event: int, frames_per_normal: int,
                   val_fraction: float, seed: int, max_videos_per_class: int = None):
    random.seed(seed)
    train_rows, val_rows = [], []

    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    for class_dir in class_dirs:
        class_name = class_dir.name
        if class_name not in CLASSES:
            print(f"WARNING: unexpected class dir '{class_name}', skipping")
            continue
        videos_csv = class_dir / "videos.csv"
        gt_csv = class_dir / "ground_truth.csv"
        video_dir = class_dir / "videos"
        if not gt_csv.exists() or not video_dir.exists():
            print(f"WARNING: missing files for class '{class_name}', skipping")
            continue

        videos_meta = {r["video_id"]: r for r in load_csv(videos_csv)} if videos_csv.exists() else {}
        gt_rows = load_csv(gt_csv)

        by_video = {}
        for r in gt_rows:
            by_video.setdefault(r["video_id"], []).append(r)

        video_ids = sorted(by_video.keys())
        random.shuffle(video_ids)
        if max_videos_per_class and len(video_ids) > max_videos_per_class:
            video_ids = video_ids[:max_videos_per_class]
        n_val = max(1, int(len(video_ids) * val_fraction)) if len(video_ids) > 3 else 0
        val_ids = set(video_ids[:n_val])

        for video_id in video_ids:
            events = by_video[video_id]
            filename = None
            if video_id in videos_meta:
                filename = videos_meta[video_id].get("filename") or videos_meta[video_id].get("file_name")
            if not filename:
                candidates = list(video_dir.glob(f"{video_id}.*"))
                filename = candidates[0].relative_to(class_dir) if candidates else Path("videos") / f"{video_id}.mp4"
            # videos.csv 'filename' is already relative to the class dir (e.g. "videos/T001.mp4")
            video_path = class_dir / filename
            if not video_path.exists():
                print(f"WARNING: video file not found: {video_path}")
                continue

            duration = ffprobe_duration(video_path)
            if duration <= 0:
                print(f"WARNING: zero duration for {video_path}")
                continue

            for ev_idx, ev in enumerate(events):
                is_anom = ev.get("is_anomaly", "").strip().lower() in ("1", "true", "yes")
                ev_class = ev.get("class_name", class_name).strip() or class_name
                desc = (ev.get("description_summary") or "").strip()
                start_s = ev.get("start_time_sec", "").strip()
                end_s = ev.get("end_time_sec", "").strip()

                if start_s and end_s:
                    start, end = float(start_s), float(end_s)
                    start = max(0.0, min(start, duration))
                    end = max(start, min(end, duration))
                    n_frames = frames_per_event
                else:
                    start, end = 0.0, duration
                    n_frames = frames_per_event if is_anom else frames_per_normal

                times = sample_times(start, end, n_frames)
                image_paths = []
                for i, t in enumerate(times):
                    img_rel = f"frames/{class_name}/{video_id}_{ev_idx}_{i}.jpg"
                    img_abs = out_dir / img_rel
                    if not img_abs.exists():
                        ok = extract_frame(video_path, t, img_abs)
                        if not ok:
                            continue
                    image_paths.append(img_rel)

                if not image_paths:
                    continue

                record = {
                    "video_id": video_id,
                    "class_name": ev_class,
                    "is_anomaly": is_anom,
                    "description_summary": desc,
                    "images": image_paths,
                }
                if video_id in val_ids:
                    val_rows.append(record)
                else:
                    train_rows.append(record)

        print(f"{class_name}: {len(video_ids)} videos processed")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train_manifest.jsonl", "w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "val_manifest.jsonl", "w") as f:
        for r in val_rows:
            f.write(json.dumps(r) + "\n")
    print(f"train examples: {len(train_rows)}  val examples: {len(val_rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="path to ahc_dataset/train")
    ap.add_argument("--out_dir", required=True, help="output dir for frames + manifests")
    ap.add_argument("--frames_per_event", type=int, default=4)
    ap.add_argument("--frames_per_normal", type=int, default=2)
    ap.add_argument("--val_fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_videos_per_class", type=int, default=None,
                     help="cap videos per class to balance an imbalanced dataset")
    args = ap.parse_args()

    process_split(
        Path(args.data_root), Path(args.out_dir),
        args.frames_per_event, args.frames_per_normal,
        args.val_fraction, args.seed, args.max_videos_per_class,
    )
