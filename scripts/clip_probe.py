#!/usr/bin/env python3
"""
Train / evaluate a classifier in CLIP latent space.

Borrows the practical parts of Alert-CLIP without needing its (unreleased)
code:
  * zero-shot head   - cosine similarity against class text prompts, no training
  * linear probe     - logistic regression on frozen CLIP features (the
                       "latent-enhanced representation tuning" done cheaply)
  * class weighting  - our data is heavily imbalanced, and the arena punishes
                       false alarms on normal videos much harder than misses
  * hard-negative awareness - reports the confusion pairs the model actually
                       collapses (fighting vs loitering, fire vs smoke) so we
                       can see whether separation improved

A window's score is the mean of its frames' probabilities.
"""
import argparse
import json
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import CLIPModel, CLIPProcessor

CLASSES = [
    "normal", "traffic_accident", "traffic_congestion",
    "stalled_or_broken_down_vehicle", "vehicle_blocking_traffic",
    "wrong_way_driving", "road_spill_or_debris", "waterlogging_or_flood",
    "fire", "smoke", "fighting_or_violence", "loitering_or_suspicious_presence",
]

# richer prompts than the bare class name - CLIP responds much better to
# natural descriptions of the scene
PROMPTS = {
    "normal": ["a normal street scene with ordinary traffic",
                "an ordinary aerial view of a road with nothing unusual",
                "a routine surveillance view, nothing happening"],
    "traffic_accident": ["a traffic accident with crashed vehicles",
                          "a car crash on a road", "vehicles collided in an accident"],
    "traffic_congestion": ["heavy traffic congestion, many vehicles queued",
                            "a traffic jam with bumper to bumper cars"],
    "stalled_or_broken_down_vehicle": ["a broken down vehicle stopped on the roadside",
                                         "a stalled car halted on a highway shoulder"],
    "vehicle_blocking_traffic": ["a vehicle parked badly blocking the road",
                                   "a car obstructing traffic flow"],
    "wrong_way_driving": ["a vehicle driving the wrong way against traffic",
                           "a car going in the opposite direction on a one way road"],
    "road_spill_or_debris": ["debris or spilled cargo scattered on the road",
                              "obstacles and litter blocking a road surface"],
    "waterlogging_or_flood": ["a flooded road covered in water",
                               "waterlogging with vehicles driving through deep water"],
    "fire": ["a large fire with visible flames", "a burning vehicle or building on fire"],
    "smoke": ["thick smoke rising", "a smoky scene with haze and no visible flames"],
    "fighting_or_violence": ["people physically fighting each other",
                              "a violent altercation between people"],
    "loitering_or_suspicious_presence": ["a person loitering suspiciously in an area",
                                           "someone standing around suspiciously at night"],
}


def text_head(model_name, device):
    """Zero-shot classifier weights from class text prompts."""
    model = CLIPModel.from_pretrained(model_name, dtype=torch.float16).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_name)
    W = []
    with torch.no_grad():
        for c in CLASSES:
            inp = proc(text=PROMPTS[c], return_tensors="pt", padding=True).to(device)
            f = model.get_text_features(**inp)
            if not torch.is_tensor(f):
                f = f.pooler_output
            f = f / f.norm(dim=-1, keepdim=True)
            W.append(f.mean(0).float().cpu().numpy())
    W = np.stack(W)
    return W / np.linalg.norm(W, axis=1, keepdims=True)


def per_row(feats, row_idx, n_rows):
    """Mean-pool frame features into one vector per window/example."""
    dim = feats.shape[1]
    out = np.zeros((n_rows, dim), dtype=np.float32)
    cnt = np.zeros(n_rows, dtype=np.int32)
    for f, r in zip(feats, row_idx):
        out[r] += f
        cnt[r] += 1
    cnt = np.maximum(cnt, 1)
    out /= cnt[:, None]
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="clip_train.npz")
    ap.add_argument("--val", help="clip_val.npz (optional)")
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--out", default="/home/extra_space/akhilesh/clip_probe.npz")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--normal_weight", type=float, default=3.0,
                     help=">1 biases toward predicting normal (false alarms are costly)")
    args = ap.parse_args()

    d = np.load(args.train, allow_pickle=True)
    X = per_row(d["feats"], d["row_idx"], len(d["class_name"]))
    y_names = d["class_name"]
    y = np.array([CLASSES.index(c) if c in CLASSES else 0 for c in y_names])
    print(f"train: {X.shape[0]} windows, {X.shape[1]}-dim")
    for i, c in enumerate(CLASSES):
        n = int((y == i).sum())
        if n:
            print(f"   {c:34s} {n}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- zero-shot baseline ----
    W = text_head(args.model, device)
    zs_pred = (X @ W.T).argmax(1)
    print(f"\nzero-shot train accuracy: {(zs_pred == y).mean():.3f}")

    # ---- linear probe ----
    cw = {i: 1.0 for i in range(len(CLASSES))}
    cw[0] = args.normal_weight
    clf = LogisticRegression(max_iter=3000, C=args.C, class_weight=cw, n_jobs=-1)
    clf.fit(X, y)
    print(f"linear-probe train accuracy: {clf.score(X, y):.3f}")

    if args.val:
        dv = np.load(args.val, allow_pickle=True)
        Xv = per_row(dv["feats"], dv["row_idx"], len(dv["class_name"]))
        yv = np.array([CLASSES.index(c) if c in CLASSES else 0 for c in dv["class_name"]])
        zs_v = (Xv @ W.T).argmax(1)
        pv = clf.predict(Xv)
        print(f"\nVAL zero-shot   accuracy: {(zs_v == yv).mean():.3f}  ({(zs_v==yv).sum()}/{len(yv)})")
        print(f"VAL linear-probe accuracy: {(pv == yv).mean():.3f}  ({(pv==yv).sum()}/{len(yv)})")

        # the confusions we actually care about
        print("\nconfusions (gt -> pred, linear probe):")
        conf = {}
        for a, b in zip(yv, pv):
            if a != b:
                conf[(CLASSES[a], CLASSES[b])] = conf.get((CLASSES[a], CLASSES[b]), 0) + 1
        for (a, b), n in sorted(conf.items(), key=lambda kv: -kv[1])[:12]:
            print(f"   {a:32s} -> {b:32s} {n}")

    np.savez(args.out, coef=clf.coef_, intercept=clf.intercept_, classes=np.array(CLASSES),
             text_head=W)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
