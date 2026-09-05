#!/usr/bin/env python3
"""
Score a submission.json against the public test ground truth, approximating the
arena's rules as described in the hackathon brief:

  L1 (24 clips, 25 marks)  - classification only, no timing
  L2 (6 videos, 35 marks)  - events + timing, IoU >= 0.5 to count
  L3 (4 videos, 40 marks)  - same as L2, longer videos

  A false alarm on a `normal` L2/L3 video zeroes that video.

Usage:
  python score_submission.py --submission submission.json \
      --ground_truth /home/extra_space/akhilesh/ahc_dataset/test/ground_truth.csv
"""
import argparse
import csv
import json
from collections import defaultdict

IOU_GATE = 0.5
MARKS = {1: 25, 2: 35, 3: 40}


def load_gt(path):
    gt = defaultdict(list)
    levels = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row["video_id"]
            levels[vid] = int(row["level"])
            is_anom = row.get("is_anomaly", "").strip().lower() in ("1", "true", "yes")
            if not is_anom:
                continue
            s, e = row.get("start_time_sec", "").strip(), row.get("end_time_sec", "").strip()
            gt[vid].append({
                "class_name": row["class_name"],
                "start": float(s) if s else None,
                "end": float(e) if e else None,
            })
    return gt, levels


def iou(a_s, a_e, b_s, b_e):
    inter = max(0.0, min(a_e, b_e) - max(a_s, b_s))
    union = max(a_e, b_e) - min(a_s, b_s)
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--ground_truth", default="/home/extra_space/akhilesh/ahc_dataset/test/ground_truth.csv")
    args = ap.parse_args()

    sub = json.load(open(args.submission))
    gt, levels = load_gt(args.ground_truth)
    preds = {p["video_id"]: p.get("events", []) for p in sub["predictions"]}

    per_level = defaultdict(lambda: {"videos": 0, "credit": 0.0})
    detail = []

    for vid, level in sorted(levels.items()):
        gt_events = gt.get(vid, [])
        pred_events = preds.get(vid, [])
        gt_is_normal = len(gt_events) == 0
        per_level[level]["videos"] += 1

        if level == 1:
            # classification only: compare the single predicted class to gt class
            pred_cls = pred_events[0]["class_name"] if pred_events else "normal"
            gt_cls = gt_events[0]["class_name"] if gt_events else "normal"
            ok = pred_cls == gt_cls
            per_level[level]["credit"] += 1.0 if ok else 0.0
            detail.append((vid, level, gt_cls, pred_cls, "OK" if ok else "WRONG"))
            continue

        # L2/L3: false alarm on a normal video zeroes it
        if gt_is_normal:
            ok = len(pred_events) == 0
            per_level[level]["credit"] += 1.0 if ok else 0.0
            detail.append((vid, level, "normal", f"{len(pred_events)} events",
                            "OK" if ok else "FALSE ALARM -> 0"))
            continue

        # match predicted events to gt events by class + IoU
        matched = 0
        used = set()
        for ge in gt_events:
            if ge["start"] is None or ge["end"] is None:
                continue
            for i, pe in enumerate(pred_events):
                if i in used or pe["class_name"] != ge["class_name"]:
                    continue
                if iou(ge["start"], ge["end"],
                        float(pe["start_time_sec"]), float(pe["end_time_sec"])) >= IOU_GATE:
                    matched += 1
                    used.add(i)
                    break
        timed_gt = [g for g in gt_events if g["start"] is not None]
        recall = matched / len(timed_gt) if timed_gt else 0.0
        precision = matched / len(pred_events) if pred_events else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_level[level]["credit"] += f1
        detail.append((vid, level, f"{len(timed_gt)} events", f"{len(pred_events)} pred",
                        f"matched={matched} f1={f1:.2f}"))

    print(f"{'video':<8}{'lvl':<5}{'ground truth':<28}{'prediction':<32}result")
    print("-" * 100)
    for row in detail:
        print(f"{row[0]:<8}L{row[1]:<4}{str(row[2]):<28}{str(row[3]):<32}{row[4]}")

    print("\n=== score estimate ===")
    total = 0.0
    for lvl in (1, 2, 3):
        d = per_level[lvl]
        if not d["videos"]:
            continue
        pts = MARKS[lvl] * d["credit"] / d["videos"]
        total += pts
        print(f"L{lvl}: {d['credit']:.2f}/{d['videos']} videos -> {pts:.1f}/{MARKS[lvl]}")
    print(f"TOTAL (of videos present): {total:.1f}/100")


if __name__ == "__main__":
    main()
