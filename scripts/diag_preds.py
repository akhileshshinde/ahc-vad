#!/usr/bin/env python3
"""Print what the CLIP probe actually predicts across a video's timeline,
next to the ground-truth intervals, so we can see whether the failure is
classification or temporal assembly."""
import argparse
import csv
import json
from collections import defaultdict

import numpy as np

CLASSES = [
    "normal", "traffic_accident", "traffic_congestion",
    "stalled_or_broken_down_vehicle", "vehicle_blocking_traffic",
    "wrong_way_driving", "road_spill_or_debris", "waterlogging_or_flood",
    "fire", "smoke", "fighting_or_violence", "loitering_or_suspicious_presence",
]


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def per_row(feats, row_idx, n):
    out = np.zeros((n, feats.shape[1]), dtype=np.float32)
    cnt = np.zeros(n, dtype=np.int32)
    for f, r in zip(feats, row_idx):
        out[r] += f
        cnt[r] += 1
    out /= np.maximum(cnt, 1)[:, None]
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--probe", default="/home/extra_space/akhilesh/clip_probe.npz")
    ap.add_argument("--gt", default="/home/extra_space/akhilesh/ahc_dataset/test/ground_truth.csv")
    ap.add_argument("--videos", nargs="+", required=True)
    args = ap.parse_args()

    pr = np.load(args.probe, allow_pickle=True)
    d = np.load(args.npz, allow_pickle=True)
    X = per_row(d["feats"], d["row_idx"], len(d["video_id"]))
    probs = softmax(X @ pr["coef"].T + pr["intercept"])

    gt = defaultdict(list)
    with open(args.gt, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("is_anomaly", "").lower() == "true" and r.get("start_time_sec"):
                gt[r["video_id"]].append((r["class_name"], float(r["start_time_sec"]),
                                           float(r["end_time_sec"])))

    for vid in args.videos:
        idx = [i for i, v in enumerate(d["video_id"]) if str(v) == vid]
        if not idx:
            print(f"{vid}: no windows\n")
            continue
        print(f"=== {vid} ===")
        print("  GT:", gt.get(vid, "none"))
        idx.sort(key=lambda i: d["start_sec"][i])
        counts = defaultdict(int)
        for i in idx:
            counts[CLASSES[int(np.argmax(probs[i]))]] += 1
        print("  predicted class histogram:", dict(sorted(counts.items(), key=lambda kv: -kv[1])))
        # timeline, coarse-grained to keep it readable
        step = max(1, len(idx) // 40)
        print("  timeline (top class per sampled window):")
        for i in idx[::step]:
            p = probs[i]
            j = int(np.argmax(p))
            jn = int(np.argmax(p[1:]) + 1)
            print(f"    {d['start_sec'][i]:6.0f}-{d['end_sec'][i]:6.0f}s  {CLASSES[j]:32s}"
                  f" p={p[j]:.2f}   best_anom={CLASSES[jn]}({p[jn]:.2f})")
        print()


if __name__ == "__main__":
    main()
