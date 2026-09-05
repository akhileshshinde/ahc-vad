#!/usr/bin/env python3
"""
Re-tune event assembly from saved per-window predictions, offline.

Inference is the expensive part (~10s/window). Once raw_window_preds.json
exists we can search assembly hyper-parameters for free and pick the setting
that scores best against the public ground truth:

  min_consec  - how many consecutive same-class windows before emitting
  pad_sec     - symmetric padding on each emitted interval (fixes the
                "intervals come out narrow, IoU lands at 0.49" failure mode)
  bridge_gap  - merge two same-class runs separated by <= this many windows
  suppress    - classes to drop entirely (our model over-triggers some)

Usage:
  python retune_assembly.py --raw raw_window_preds.json \
      --l1_submission submission_fixed.json --out best_submission.json
"""
import argparse
import csv
import json
import itertools
from collections import defaultdict
from pathlib import Path

IOU_GATE = 0.5
MARKS = {1: 25, 2: 35, 3: 40}


def load_gt(path):
    gt = defaultdict(list)
    levels = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row["video_id"]
            levels[vid] = int(row["level"])
            if row.get("is_anomaly", "").strip().lower() not in ("1", "true", "yes"):
                continue
            s, e = row.get("start_time_sec", "").strip(), row.get("end_time_sec", "").strip()
            gt[vid].append({"class_name": row["class_name"],
                             "start": float(s) if s else None,
                             "end": float(e) if e else None})
    return gt, levels


def iou(a_s, a_e, b_s, b_e):
    inter = max(0.0, min(a_e, b_e) - max(a_s, b_s))
    union = max(a_e, b_e) - min(a_s, b_s)
    return inter / union if union > 0 else 0.0


def assemble(windows, min_consec, pad_sec, bridge_gap, suppress, duration=None):
    ws = sorted(windows, key=lambda x: x["window_index"])
    runs, run = [], []
    for w in ws + [None]:
        pos = w is not None and w["pred"] != "normal" and w["pred"] not in suppress
        if pos and (not run or run[-1]["pred"] == w["pred"]):
            run.append(w)
            continue
        if len(run) >= min_consec:
            runs.append(run)
        run = [w] if pos else []

    # bridge same-class runs separated by a small gap
    merged = []
    for r in runs:
        if merged and merged[-1][0]["pred"] == r[0]["pred"] and \
           (r[0]["window_index"] - merged[-1][-1]["window_index"]) <= bridge_gap + 1:
            merged[-1] = merged[-1] + r
        else:
            merged.append(r)

    events = []
    for r in merged:
        s = max(0.0, r[0]["start_sec"] - pad_sec)
        e = r[-1]["end_sec"] + pad_sec
        if duration:
            e = min(e, duration)
        events.append({"class_name": r[0]["pred"],
                        "start_time_sec": round(s, 2),
                        "end_time_sec": round(e, 2)})
    return events


def score_l23(preds_by_video, gt, levels):
    """Return (total_points, per_level_detail) for L2/L3 only."""
    per_level = defaultdict(lambda: {"videos": 0, "credit": 0.0})
    for vid, level in levels.items():
        if level == 1:
            continue
        per_level[level]["videos"] += 1
        gt_events = gt.get(vid, [])
        pred_events = preds_by_video.get(vid, [])
        if not gt_events:                       # normal video: any event zeroes it
            per_level[level]["credit"] += 1.0 if not pred_events else 0.0
            continue
        matched, used = 0, set()
        for ge in gt_events:
            if ge["start"] is None:
                continue
            for i, pe in enumerate(pred_events):
                if i in used or pe["class_name"] != ge["class_name"]:
                    continue
                if iou(ge["start"], ge["end"], pe["start_time_sec"], pe["end_time_sec"]) >= IOU_GATE:
                    matched += 1
                    used.add(i)
                    break
        timed = [g for g in gt_events if g["start"] is not None]
        rec = matched / len(timed) if timed else 0.0
        prec = matched / len(pred_events) if pred_events else 0.0
        per_level[level]["credit"] += (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    total = sum(MARKS[l] * d["credit"] / d["videos"] for l, d in per_level.items() if d["videos"])
    return total, per_level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw_window_preds.json from the Kaggle run")
    ap.add_argument("--l1_submission", required=True, help="existing submission with L1 predictions")
    ap.add_argument("--ground_truth", default="/home/extra_space/akhilesh/ahc_dataset/test/ground_truth.csv")
    ap.add_argument("--manifest", default="/home/extra_space/akhilesh/manifest.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = json.loads(Path(args.raw).read_text())
    gt, levels = load_gt(args.ground_truth)
    manifest = {v["video_id"]: v for v in json.loads(Path(args.manifest).read_text())["videos"]}
    base = json.loads(Path(args.l1_submission).read_text())
    l1_preds = {p["video_id"]: p for p in base["predictions"]}

    grid = {
        "min_consec": [1, 2, 3],
        "pad_sec": [0.0, 2.0, 5.0],
        "bridge_gap": [0, 1, 2],
        "suppress": [frozenset(), frozenset({"loitering_or_suspicious_presence"}),
                      frozenset({"road_spill_or_debris"}),
                      frozenset({"loitering_or_suspicious_presence", "road_spill_or_debris"})],
    }

    best = None
    for mc, pad, bg, sup in itertools.product(grid["min_consec"], grid["pad_sec"],
                                               grid["bridge_gap"], grid["suppress"]):
        preds = {vid: assemble(ws, mc, pad, bg, sup,
                                duration=float(manifest[vid]["duration_sec"]) if vid in manifest else None)
                 for vid, ws in raw.items()}
        pts, detail = score_l23(preds, gt, levels)
        if best is None or pts > best[0]:
            best = (pts, mc, pad, bg, sup, preds, detail)

    pts, mc, pad, bg, sup, preds, detail = best
    print(f"best L2+L3 = {pts:.1f} marks with "
          f"min_consec={mc} pad_sec={pad} bridge_gap={bg} suppress={sorted(sup)}")
    for lvl in (2, 3):
        if detail[lvl]["videos"]:
            print(f"  L{lvl}: {detail[lvl]['credit']:.2f}/{detail[lvl]['videos']} videos"
                  f" -> {MARKS[lvl]*detail[lvl]['credit']/detail[lvl]['videos']:.1f}/{MARKS[lvl]}")

    out_preds = []
    for vid in [v["video_id"] for v in manifest.values()]:
        if levels.get(vid) == 1:
            p = l1_preds.get(vid, {"video_id": vid, "events": [],
                "runtime_metadata": {"frames_processed": 0, "chunks_processed": 0,
                                      "end_to_end_internal_time_ms": 0, "model_runtimes": []}})
            for ev in p.get("events", []):
                ev["start_time_sec"] = None
                ev["end_time_sec"] = None
            out_preds.append(p)
        else:
            ws = raw.get(vid, [])
            out_preds.append({"video_id": vid, "events": preds.get(vid, []),
                "runtime_metadata": {"frames_processed": len(ws) * 2,
                    "chunks_processed": len(ws),
                    "end_to_end_internal_time_ms": sum(w.get("ms", 0) for w in ws),
                    "model_runtimes": []}})

    base["predictions"] = out_preds
    Path(args.out).write_text(json.dumps(base, indent=2))
    print(f"wrote {args.out} ({len(out_preds)} videos)")


if __name__ == "__main__":
    main()
