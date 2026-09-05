#!/usr/bin/env python3
"""Rebuild the fine L2/L3 windows.jsonl from frames already on disk.

build_l23_fine.py only writes its manifest after every video finishes, so this
reconstructs it mid-flight from whatever frames exist, recomputing window
timestamps from the same 6s/3s scheme.
"""
import json
import re
from pathlib import Path

WINDOW_SEC = 6.0
STRIDE_SEC = 3.0
ROOT = Path("/home/extra_space/akhilesh/ahc_l23_fine")
MANIFEST = Path("/home/extra_space/akhilesh/manifest.json")
FRAME_RE = re.compile(r"^w(\d+)_(\d+)\.jpg$")


def main():
    meta = {v["video_id"]: v for v in json.loads(MANIFEST.read_text())["videos"]}
    rows = []
    for vdir in sorted((ROOT / "frames").iterdir()):
        vid = vdir.name
        if vid not in meta:
            continue
        level = int(meta[vid]["level"])
        duration = float(meta[vid]["duration_sec"])
        by_window = {}
        for f in vdir.iterdir():
            m = FRAME_RE.match(f.name)
            if m:
                by_window.setdefault(int(m.group(1)), []).append((int(m.group(2)), f.name))
        for wi in sorted(by_window):
            start = wi * STRIDE_SEC
            if start >= duration:
                continue
            frames = [n for _, n in sorted(by_window[wi])]
            rows.append({
                "video_id": vid, "level": level, "window_index": wi,
                "start_sec": round(start, 3),
                "end_sec": round(min(start + WINDOW_SEC, duration), 3),
                "images": [f"frames/{vid}/{n}" for n in frames],
            })

    out = ROOT / "windows.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    per_level, per_video = {}, {}
    for r in rows:
        per_level[r["level"]] = per_level.get(r["level"], 0) + 1
        per_video[r["video_id"]] = per_video.get(r["video_id"], 0) + 1
    print(f"wrote {out}: {len(rows)} windows, per level {per_level}")
    for v in sorted(per_video):
        print(f"   {v} L{meta[v]['level']}: {per_video[v]} windows")


if __name__ == "__main__":
    main()
