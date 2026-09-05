#!/usr/bin/env python3
"""
Normalize a submission.json to the arena's schema:
  - Level 1 is not temporally scored -> both time fields must be null
  - every video in manifest.json must be present exactly once
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--manifest", default="/home/extra_space/akhilesh/manifest.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sub = json.loads(Path(args.submission).read_text())
    manifest = json.loads(Path(args.manifest).read_text())
    levels = {v["video_id"]: int(v["level"]) for v in manifest["videos"]}

    existing = {p["video_id"]: p for p in sub["predictions"]}
    fixed = []
    n_nulled = 0

    for vid in [v["video_id"] for v in manifest["videos"]]:
        p = existing.get(vid, {
            "video_id": vid,
            "events": [],
            "runtime_metadata": {"frames_processed": 0, "chunks_processed": 0,
                                  "end_to_end_internal_time_ms": 0, "model_runtimes": []},
        })
        if levels[vid] == 1:
            for ev in p.get("events", []):
                if ev.get("start_time_sec") is not None or ev.get("end_time_sec") is not None:
                    n_nulled += 1
                ev["start_time_sec"] = None
                ev["end_time_sec"] = None
        fixed.append(p)

    sub["predictions"] = fixed
    Path(args.out).write_text(json.dumps(sub, indent=2))

    missing = set(levels) - set(existing)
    print(f"wrote {args.out}")
    print(f"  videos: {len(fixed)} (added {len(missing)} missing: {sorted(missing)})")
    print(f"  L1 events with times nulled: {n_nulled}")
    counts = {}
    for p in fixed:
        counts[levels[p['video_id']]] = counts.get(levels[p['video_id']], 0) + len(p["events"])
    print(f"  events per level: {counts}")


if __name__ == "__main__":
    main()
