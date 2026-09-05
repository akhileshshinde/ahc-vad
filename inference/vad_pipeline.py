#!/usr/bin/env python3
"""
Real-time video anomaly detection pipeline for the AHC hackathon.

Runs the fine-tuned Qwen2.5-VL-3B GGUF model (served locally by llama-server
with GPU offload on the RTX 3050) over a sliding window of sampled frames.
No cloud calls at inference time -- llama-server runs entirely on-device.

Usage:
  python vad_pipeline.py --source path/to/video.mp4 [--server http://127.0.0.1:8080]
  python vad_pipeline.py --source 0                      # webcam / device index
  python vad_pipeline.py --source rtsp://...              # CCTV/drone stream

For each window of WINDOW_SECONDS, samples FRAMES_PER_WINDOW frames evenly,
sends them (as one multi-image chat message, matching the training format
exactly) to the local llama-server OpenAI-compatible /v1/chat/completions
endpoint, and prints/logs any anomaly with debouncing so a sustained event
doesn't spam duplicate alerts.
"""
import argparse
import base64
import json
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import requests

CLASSES = [
    "normal", "traffic_accident", "traffic_congestion",
    "stalled_or_broken_down_vehicle", "vehicle_blocking_traffic",
    "wrong_way_driving", "road_spill_or_debris", "waterlogging_or_flood",
    "fire", "smoke", "fighting_or_violence", "loitering_or_suspicious_presence",
]

# Must exactly match the instruction used during training (scripts/make_notebook.py)
INSTRUCTION = (
    "You are a real-time anomaly detection system analyzing frames sampled "
    "from drone, CCTV, or dashcam footage, in temporal order. Classify the "
    "scene into exactly one of these classes: " + ", ".join(CLASSES) + ". "
    "Respond ONLY with compact JSON: "
    '{"class_name": "<one of the classes>", "is_anomaly": <true|false>, "description": "<one short sentence>"}'
)

CLASS_RE = re.compile(r'"class_name"\s*:\s*"([a-z_]+)"')
ANOM_RE = re.compile(r'"is_anomaly"\s*:\s*(true|false)', re.IGNORECASE)
DESC_RE = re.compile(r'"description"\s*:\s*"([^"]*)"')


def parse_response(text: str) -> dict:
    """Best-effort parse: try strict JSON first, fall back to regex since a
    3B model under real-time constraints can occasionally emit near-JSON."""
    text = text.strip()
    try:
        obj = json.loads(text)
        return {
            "class_name": obj.get("class_name"),
            "is_anomaly": bool(obj.get("is_anomaly")),
            "description": obj.get("description", ""),
        }
    except (json.JSONDecodeError, TypeError):
        pass
    cls = CLASS_RE.search(text)
    anom = ANOM_RE.search(text)
    desc = DESC_RE.search(text)
    return {
        "class_name": cls.group(1) if cls else None,
        "is_anomaly": (anom.group(1).lower() == "true") if anom else None,
        "description": desc.group(1) if desc else "",
    }


def ffprobe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def extract_frame_jpeg_b64(path: str, t: float, max_side: int = 672) -> str:
    scale = f"scale='min({max_side},iw)':'min({max_side},ih)':force_original_aspect_ratio=decrease"
    cmd = ["ffmpeg", "-y", "-ss", f"{max(t,0):.3f}", "-i", path,
           "-frames:v", "1", "-q:v", "3", "-vf", scale, "-f", "image2pipe",
           "-vcodec", "mjpeg", "pipe:1"]
    res = subprocess.run(cmd, capture_output=True)
    if not res.stdout:
        return None
    return base64.b64encode(res.stdout).decode("ascii")


def classify_window(server: str, frames_b64: list, timeout: float = 30.0) -> dict:
    content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in frames_b64]
    content.append({"type": "text", "text": INSTRUCTION})
    payload = {
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1,
        "max_tokens": 100,
    }
    resp = requests.post(f"{server}/v1/chat/completions", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    return parse_response(text), text


def run_on_video(source: str, server: str, window_seconds: float, frames_per_window: int,
                  stride_seconds: float, debounce_windows: int):
    duration = ffprobe_duration(source)
    if duration <= 0:
        print(f"ERROR: could not read duration for {source}", file=sys.stderr)
        return

    last_class = None
    same_class_streak = 0
    t = 0.0
    while t < duration:
        w_start, w_end = t, min(t + window_seconds, duration)
        times = [w_start + (w_end - w_start) * i / max(frames_per_window - 1, 1)
                 for i in range(frames_per_window)] if frames_per_window > 1 else [(w_start + w_end) / 2]

        frames_b64 = []
        for ft in times:
            b64 = extract_frame_jpeg_b64(source, ft)
            if b64:
                frames_b64.append(b64)
        if not frames_b64:
            t += stride_seconds
            continue

        t0 = time.time()
        try:
            result, raw = classify_window(server, frames_b64)
        except Exception as e:
            print(f"[{w_start:6.1f}s] ERROR calling llama-server: {e}", file=sys.stderr)
            t += stride_seconds
            continue
        latency = time.time() - t0

        cls = result.get("class_name")
        is_anom = result.get("is_anomaly")
        desc = result.get("description")

        if cls == last_class:
            same_class_streak += 1
        else:
            same_class_streak = 0
            last_class = cls

        alert = is_anom and same_class_streak < debounce_windows
        tag = "ALERT" if alert else ("anomaly(debounced)" if is_anom else "normal")
        print(f"[{w_start:6.1f}s-{w_end:6.1f}s] ({latency:5.2f}s) {tag:20s} class={cls} desc={desc}")

        t += stride_seconds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video file path (webcam/RTSP support to follow)")
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--window_seconds", type=float, default=2.0)
    ap.add_argument("--frames_per_window", type=int, default=3)
    ap.add_argument("--stride_seconds", type=float, default=2.0, help="hop between windows; == window_seconds for non-overlapping")
    ap.add_argument("--debounce_windows", type=int, default=3, help="suppress repeat alerts for N consecutive same-class windows")
    args = ap.parse_args()

    run_on_video(args.source, args.server, args.window_seconds, args.frames_per_window,
                 args.stride_seconds, args.debounce_windows)
