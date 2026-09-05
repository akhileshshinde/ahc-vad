#!/usr/bin/env python3
"""
Extract sliding-window frames from the 34 public test videos so inference can
run on Kaggle without uploading the raw ~2GB of video.

Window scheme (derived from the scoring rules):
  - L1 videos (short clips, classify-only, no timestamps scored): one window
    spanning the whole clip.
  - L2/L3 videos (long, need timed intervals at IoU >= 0.5): sliding window of
    WINDOW_SEC with STRIDE_SEC hop. Stride must stay <= ~3.3s so a predicted
    interval can still clear the IoU 0.5 gate against a real event.

Output:
  <out_dir>/frames/<video_id>/w<idx>_<f>.jpg
  <out_dir>/windows.jsonl   one row per window: video_id, level, start, end, images
"""
import argparse
import json
import subprocess
from pathlib import Path

WINDOW_SEC = 5.0
STRIDE_SEC = 3.0
FRAMES_PER_WINDOW = 3
MAX_SIDE = 384  # friend's sweep: 384px gave ~11x speedup AND lowered false alarms


def ffprobe_duration(video_path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def extract_frame(video_path: Path, t: float, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    scale = f"scale='min({MAX_SIDE},iw)':'min({MAX_SIDE},ih)':force_original_aspect_ratio=decrease"
    cmd = ["ffmpeg", "-y", "-ss", f"{max(t,0):.3f}", "-i", str(video_path),
           "-frames:v", "1", "-q:v", "3", "-vf", scale, str(out_path)]
    subprocess.run(cmd, capture_output=True)
    return out_path.exists() and out_path.stat().st_size > 0


def windows_for(level: int, duration: float):
    """Yield (start, end) windows for a video given its difficulty level."""
    if level == 1:
        # classify-only: one window over the whole clip
        yield 0.0, duration
        return
    t = 0.0
    while t < duration:
        end = min(t + WINDOW_SEC, duration)
        yield t, end
        if end >= duration:
            break
        t += STRIDE_SEC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", default="/home/extra_space/akhilesh/ahc_dataset/test")
    ap.add_argument("--manifest", default="/home/extra_space/akhilesh/manifest.json")
    ap.add_argument("--out_dir", default="/home/extra_space/akhilesh/ahc_test_windows")
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    out_dir = Path(args.out_dir)
    manifest = json.loads(Path(args.manifest).read_text())

    rows = []
    for v in manifest["videos"]:
        vid, level = v["video_id"], int(v["level"])
        video_path = test_dir / "videos" / f"{vid}.mp4"
        if not video_path.exists():
            print(f"WARNING: missing {video_path}")
            continue
        duration = v.get("duration_sec") or ffprobe_duration(video_path)

        n_win = 0
        for wi, (ws, we) in enumerate(windows_for(level, float(duration))):
            span = we - ws
            if FRAMES_PER_WINDOW > 1 and span > 0:
                times = [ws + span * i / (FRAMES_PER_WINDOW - 1) for i in range(FRAMES_PER_WINDOW)]
            else:
                times = [(ws + we) / 2]
            images = []
            for fi, t in enumerate(times):
                rel = f"frames/{vid}/w{wi:04d}_{fi}.jpg"
                if extract_frame(video_path, t, out_dir / rel):
                    images.append(rel)
            if images:
                rows.append({"video_id": vid, "level": level, "window_index": wi,
                              "start_sec": round(ws, 3), "end_sec": round(we, 3),
                              "images": images})
                n_win += 1
        print(f"{vid} (L{level}, {duration:.1f}s): {n_win} windows", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "windows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"total windows: {len(rows)}")


if __name__ == "__main__":
    main()
