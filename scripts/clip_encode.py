#!/usr/bin/env python3
"""
Encode frames with CLIP and cache the embeddings.

This is the cheap half of the Alert-CLIP idea: instead of having a 3B
generative VLM emit tokens for every window (~10s each), embed each frame once
with CLIP and classify in that latent space (~10ms each). Everything runs on
the local RTX 3050.

Caches to .npy so probes can be retrained instantly without re-encoding.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CLASSES = [
    "normal", "traffic_accident", "traffic_congestion",
    "stalled_or_broken_down_vehicle", "vehicle_blocking_traffic",
    "wrong_way_driving", "road_spill_or_debris", "waterlogging_or_flood",
    "fire", "smoke", "fighting_or_violence", "loitering_or_suspicious_presence",
]


def load_clip(model_name, device):
    model = CLIPModel.from_pretrained(model_name, dtype=torch.float16).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_name)
    return model, proc


@torch.no_grad()
def encode_images(paths, model, proc, device, batch_size=64):
    feats = []
    t0 = time.time()
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        imgs = []
        for p in batch:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                imgs.append(Image.new("RGB", (224, 224)))
        inputs = proc(images=imgs, return_tensors="pt").to(device)
        inputs["pixel_values"] = inputs["pixel_values"].half()
        f = model.get_image_features(**inputs)
        if not torch.is_tensor(f):          # transformers 5.x returns an output object
            f = f.pooler_output
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.float().cpu().numpy())
        if (i // batch_size) % 10 == 0:
            done = min(i + batch_size, len(paths))
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{len(paths)} ({rate:.0f} img/s)", flush=True)
    return np.concatenate(feats, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="jsonl with 'images' + optional 'class_name'")
    ap.add_argument("--root", required=True, help="dir the image paths are relative to")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.model} on {device}", flush=True)
    model, proc = load_clip(args.model, device)

    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    root = Path(args.root)

    # flatten: one entry per image, remembering which row it came from
    paths, row_idx = [], []
    for i, r in enumerate(rows):
        for rel in r["images"]:
            paths.append(str(root / rel))
            row_idx.append(i)
    print(f"{len(rows)} rows -> {len(paths)} frames", flush=True)

    feats = encode_images(paths, model, proc, device, args.batch_size)

    meta = {
        "row_idx": np.array(row_idx),
        "class_name": np.array([r.get("class_name", "") for r in rows]),
        "video_id": np.array([r.get("video_id", "") for r in rows]),
        "level": np.array([r.get("level", 0) for r in rows]),
        "window_index": np.array([r.get("window_index", 0) for r in rows]),
        "start_sec": np.array([r.get("start_sec", 0.0) for r in rows], dtype=float),
        "end_sec": np.array([r.get("end_sec", 0.0) for r in rows], dtype=float),
    }
    np.savez_compressed(args.out, feats=feats, **meta)
    print(f"wrote {args.out}: feats {feats.shape}", flush=True)


if __name__ == "__main__":
    main()
