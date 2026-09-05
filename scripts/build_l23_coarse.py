#!/usr/bin/env python3
"""
Re-extract L2/L3 windows on a coarser, contiguous grid so the full inference
pass fits in the remaining time budget.

The fine grid (5s window / 3s stride) gives the best shot at the IoU>=0.5 gate
but costs ~1036 windows (~2.9h at the measured ~10.6s/window). A contiguous
10s grid covers the same footage with ~310 windows (~45min) at the cost of
coarser interval boundaries.
"""
import json
import subprocess
from pathlib import Path

WINDOW_SEC = 10.0
STRIDE_SEC = 10.0     # contiguous, no overlap
FRAMES_PER_WINDOW = 2
MAX_SIDE = 384

TEST_DIR = Path("/home/extra_space/akhilesh/ahc_dataset/test")
OUT_DIR = Path("/home/extra_space/akhilesh/ahc_l23_coarse")
MANIFEST = Path("/home/extra_space/akhilesh/manifest.json")


def extract_frame(video_path: Path, t: float, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    scale = f"scale='min({MAX_SIDE},iw)':'min({MAX_SIDE},ih)':force_original_aspect_ratio=decrease"
    subprocess.run(["ffmpeg", "-y", "-ss", f"{max(t,0):.3f}", "-i", str(video_path),
                    "-frames:v", "1", "-q:v", "3", "-vf", scale, str(out_path)],
                   capture_output=True)
    return out_path.exists() and out_path.stat().st_size > 0


def main():
    manifest = json.loads(MANIFEST.read_text())
    rows = []
    for v in manifest["videos"]:
        level = int(v["level"])
        if level == 1:
            continue
        vid = v["video_id"]
        duration = float(v["duration_sec"])
        video_path = TEST_DIR / "videos" / f"{vid}.mp4"
        if not video_path.exists():
            print(f"WARNING missing {video_path}", flush=True)
            continue

        n = 0
        t = 0.0
        wi = 0
        while t < duration:
            end = min(t + WINDOW_SEC, duration)
            span = end - t
            times = ([t + span * i / (FRAMES_PER_WINDOW - 1) for i in range(FRAMES_PER_WINDOW)]
                     if FRAMES_PER_WINDOW > 1 and span > 0 else [(t + end) / 2])
            images = []
            for fi, ft in enumerate(times):
                rel = f"frames/{vid}/w{wi:04d}_{fi}.jpg"
                if extract_frame(video_path, ft, OUT_DIR / rel):
                    images.append(rel)
            if images:
                rows.append({"video_id": vid, "level": level, "window_index": wi,
                              "start_sec": round(t, 2), "end_sec": round(end, 2),
                              "images": images})
                n += 1
            wi += 1
            t += STRIDE_SEC
        print(f"{vid} L{level} {duration:.0f}s -> {n} windows", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "windows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    import shutil
    shutil.copy(MANIFEST, OUT_DIR / "manifest.json")
    print(f"TOTAL {len(rows)} windows", flush=True)


if __name__ == "__main__":
    main()
