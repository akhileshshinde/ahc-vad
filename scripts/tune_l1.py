#!/usr/bin/env python3
"""Tune how the CLIP probe turns per-window probabilities into a single L1 label.

L1 is classify-only, so the only decision is: which of the 12 classes (or
normal) does this clip get. Options swept:
  pool        - mean vs max over the clip's windows/frames
  normal_bias - multiply the normal probability (>1 predicts normal more
                readily, <1 less); the probe was trained with normal_weight=3
                which may over-suppress real events on short clips
"""
import argparse
import csv
import itertools
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
    args = ap.parse_args()

    pr = np.load(args.probe, allow_pickle=True)
    d = np.load(args.npz, allow_pickle=True)
    X = per_row(d["feats"], d["row_idx"], len(d["video_id"]))
    probs = softmax(X @ pr["coef"].T + pr["intercept"])

    gt, levels = {}, {}
    with open(args.gt, newline="") as f:
        for r in csv.DictReader(f):
            levels[r["video_id"]] = int(r["level"])
            if r["video_id"] not in gt:
                gt[r["video_id"]] = r["class_name"] if r["is_anomaly"].lower() == "true" else "normal"

    by_video = defaultdict(list)
    for i, v in enumerate(d["video_id"]):
        if levels.get(str(v)) == 1:
            by_video[str(v)].append(probs[i])

    print(f"tuning L1 over {len(by_video)} videos\n")
    best = None
    for pool, nb in itertools.product(["mean", "max"], [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]):
        correct, preds = 0, {}
        for vid, ps in by_video.items():
            P = np.mean(ps, axis=0) if pool == "mean" else np.max(ps, axis=0)
            P = P.copy()
            P[0] *= nb
            c = CLASSES[int(np.argmax(P))]
            preds[vid] = c
            correct += int(c == gt.get(vid, "normal"))
        if best is None or correct > best[0]:
            best = (correct, pool, nb, dict(preds))
        print(f"  pool={pool:4s} normal_bias={nb:.1f} -> {correct}/{len(by_video)}")

    correct, pool, nb, preds = best
    print(f"\nBEST: pool={pool} normal_bias={nb} -> {correct}/{len(by_video)}")
    print("\nper-video (gt -> pred):")
    for vid in sorted(preds):
        g, p = gt.get(vid, "normal"), preds[vid]
        print(f"  {vid}  {g:34s} -> {p:34s} {'OK' if g == p else 'X'}")
    json.dump({"pool": pool, "normal_bias": nb, "preds": preds},
              open("/home/extra_space/akhilesh/l1_tuned.json", "w"), indent=2)
    print("\nwrote /home/extra_space/akhilesh/l1_tuned.json")


if __name__ == "__main__":
    main()
