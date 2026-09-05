#!/usr/bin/env python3
"""Rebuild windows.jsonl from frames already extracted on disk.

build_test_windows.py only writes its manifest at the very end, so if it is
stopped early (or we deliberately only want a subset) this reconstructs the
manifest from whatever frames exist, recomputing each window's timestamps from
the same window/stride scheme.
"""
import argparse
import json
import re
from pathlib import Path

WINDOW_SEC = 5.0
STRIDE_SEC = 3.0

FRAME_RE = re.compile(r"^w(\d+)_(\d+)\.jpg$")


def window_bounds(level: int, idx: int, duration: float):
    if level == 1:
        return 0.0, duration
    start = idx * STRIDE_SEC
    return start, min(start + WINDOW_SEC, duration)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/extra_space/akhilesh/ahc_test_windows")
    ap.add_argument("--manifest", default="/home/extra_space/akhilesh/manifest.json")
    ap.add_argument("--levels", default="1", help="comma-separated levels to include, e.g. '1' or '1,2,3'")
    ap.add_argument("--out", default=None, help="defaults to <dir>/windows.jsonl")
    args = ap.parse_args()

    root = Path(args.dir)
    want_levels = {int(x) for x in args.levels.split(",")}
    meta = {v["video_id"]: v for v in json.loads(Path(args.manifest).read_text())["videos"]}

    rows = []
    for vid_dir in sorted((root / "frames").iterdir()):
        vid = vid_dir.name
        if vid not in meta:
            continue
        level = int(meta[vid]["level"])
        if level not in want_levels:
            continue
        duration = float(meta[vid]["duration_sec"])

        by_window = {}
        for f in vid_dir.iterdir():
            m = FRAME_RE.match(f.name)
            if not m:
                continue
            by_window.setdefault(int(m.group(1)), []).append((int(m.group(2)), f.name))

        for widx in sorted(by_window):
            frames = [n for _, n in sorted(by_window[widx])]
            ws, we = window_bounds(level, widx, duration)
            if ws >= duration:
                continue
            rows.append({
                "video_id": vid,
                "level": level,
                "window_index": widx,
                "start_sec": round(ws, 3),
                "end_sec": round(we, 3),
                "images": [f"frames/{vid}/{n}" for n in frames],
            })

    out = Path(args.out) if args.out else root / "windows.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    per_level = {}
    per_video = {}
    for r in rows:
        per_level[r["level"]] = per_level.get(r["level"], 0) + 1
        per_video[r["video_id"]] = per_video.get(r["video_id"], 0) + 1
    print(f"wrote {out}: {len(rows)} windows across {len(per_video)} videos")
    print("per level:", per_level)


if __name__ == "__main__":
    main()
