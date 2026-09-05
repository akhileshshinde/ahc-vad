# ============================================================
# PASTE THIS AS A NEW CELL IN THE KAGGLE NOTEBOOK
# Runs the fine-tuned model over test windows, writes submission.json
# INCREMENTALLY (L1 first, then L2, then L3) so there is always a valid
# submittable file on disk even if you run out of time and stop early.
# Requires: `ahc-vad-test-windows` dataset attached as an input.
# ============================================================
import os, json, re, time
from PIL import Image
from unsloth import FastVisionModel
from IPython.display import FileLink, display

TEST_DIR = "/kaggle/working/ahc_test_windows"  # after unzipping
OUT_PATH = "/kaggle/working/submission.json"
assert os.path.exists(TEST_DIR), f"Attach ahc-vad-test-windows first (looked in {TEST_DIR})"

FastVisionModel.for_inference(model)

CLASSES = [
    "normal", "traffic_accident", "traffic_congestion",
    "stalled_or_broken_down_vehicle", "vehicle_blocking_traffic",
    "wrong_way_driving", "road_spill_or_debris", "waterlogging_or_flood",
    "fire", "smoke", "fighting_or_violence", "loitering_or_suspicious_presence",
]
INSTRUCTION = (
    "You are a real-time anomaly detection system analyzing frames sampled "
    "from drone, CCTV, or dashcam footage, in temporal order. Classify the "
    "scene into exactly one of these classes: " + ", ".join(CLASSES) + ". "
    "Respond ONLY with compact JSON: "
    '{"class_name": "<one of the classes>", "is_anomaly": <true|false>, "description": "<one short sentence>"}'
)
CLASS_RE = re.compile(r'"class_name"\s*:\s*"([a-z_]+)"')

windows = [json.loads(l) for l in open(os.path.join(TEST_DIR, "windows.jsonl")) if l.strip()]
levels = {}
by_video = {}
for w in windows:
    by_video.setdefault(w["video_id"], []).append(w)
    levels[w["video_id"]] = w["level"]
print(f"{len(windows)} windows across {len(by_video)} videos", flush=True)

# false alarm on a normal L2/L3 video zeroes that video -> require a run of
# >=MIN_CONSEC same-class positive windows before emitting an event
MIN_CONSEC = 2
T_START = time.time()
results = {}   # video_id -> list of classified windows


def classify(images_rel):
    imgs = [Image.open(os.path.join(TEST_DIR, p)).convert("RGB") for p in images_rel]
    messages = [{"role": "user", "content": [{"type": "image", "image": im} for im in imgs]
                  + [{"type": "text", "text": INSTRUCTION}]}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(imgs, text, add_special_tokens=False, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=60, use_cache=True, temperature=0.1)
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    m = CLASS_RE.search(gen)
    return (m.group(1) if m else "normal")


def assemble(vid):
    ws = sorted(results.get(vid, []), key=lambda x: x["window_index"])
    if not ws:
        return []
    if levels[vid] == 1:
        cls = ws[0]["pred"]
        return [] if cls == "normal" else [{
            "class_name": cls,
            "start_time_sec": ws[0]["start_sec"],
            "end_time_sec": ws[0]["end_sec"],
        }]
    events, run = [], []
    for w in ws + [None]:
        if w is not None and w["pred"] != "normal" and (not run or run[-1]["pred"] == w["pred"]):
            run.append(w)
            continue
        if len(run) >= MIN_CONSEC:
            events.append({"class_name": run[0]["pred"],
                            "start_time_sec": round(run[0]["start_sec"], 2),
                            "end_time_sec": round(run[-1]["end_sec"], 2)})
        run = [w] if (w is not None and w["pred"] != "normal") else []
    return events


def write_submission():
    preds = []
    for vid in sorted(by_video):
        ws = results.get(vid, [])
        preds.append({
            "video_id": vid,
            "events": assemble(vid),
            "runtime_metadata": {
                "frames_processed": sum(len(w["images"]) for w in ws),
                "chunks_processed": len(ws),
                "end_to_end_internal_time_ms": sum(w["ms"] for w in ws),
                "model_runtimes": [],
            },
        })
    sub = {
        "schema_version": "1.0",
        "submission_id": "akhilesh-qwen25vl3b-lora-run01",
        "model_name": "Qwen2.5-VL-3B-Instruct + LoRA (Unsloth QLoRA)",
        "run_metadata": {
            "total_wall_time_ms": int((time.time() - T_START) * 1000),
            "max_parallel_videos": 1,
            "hardware": "Kaggle Tesla T4 (fp16); deployable artifact is Q4_K_M GGUF for RTX 3050 4GB via llama.cpp",
        },
        "predictions": preds,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(sub, f, indent=2)


def process(vid):
    t0 = time.time()
    for w in sorted(by_video[vid], key=lambda x: x["window_index"]):
        tw = time.time()
        pred = classify(w["images"])
        results.setdefault(vid, []).append({**w, "pred": pred, "ms": int((time.time() - tw) * 1000)})
    evs = assemble(vid)
    print(f"{vid} L{levels[vid]}: {len(by_video[vid])} windows in {time.time()-t0:.0f}s -> {len(evs)} events {evs[:2]}", flush=True)
    write_submission()


# ---------- pass 1: L1 (25 marks, only 24 windows -> do these first) ----------
print("=== PASS 1: L1 videos ===", flush=True)
for vid in sorted([v for v in by_video if levels[v] == 1]):
    process(vid)
print(f"L1 done at {time.time()-T_START:.0f}s — submission.json is already valid", flush=True)
display(FileLink(OUT_PATH))

# ---------- pass 2: L2 ----------
print("=== PASS 2: L2 videos ===", flush=True)
for vid in sorted([v for v in by_video if levels[v] == 2]):
    process(vid)
print(f"L2 done at {time.time()-T_START:.0f}s", flush=True)

# ---------- pass 3: L3 ----------
print("=== PASS 3: L3 videos ===", flush=True)
for vid in sorted([v for v in by_video if levels[v] == 3]):
    process(vid)

print(f"ALL DONE in {time.time()-T_START:.0f}s", flush=True)
display(FileLink(OUT_PATH))
