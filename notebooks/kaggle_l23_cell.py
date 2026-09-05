import os, json, re, time, glob
from PIL import Image
from unsloth import FastVisionModel
from IPython.display import FileLink, display

# autodetect the coarse L2/L3 window set
cands = glob.glob("/kaggle/input/**/ahc_l23_coarse/windows.jsonl", recursive=True) + \
        glob.glob("/kaggle/working/**/ahc_l23_coarse/windows.jsonl", recursive=True) + \
        glob.glob("/kaggle/input/**/windows.jsonl", recursive=True)
L23_DIR = os.path.dirname([c for c in cands if "l23" in c.lower()][0]) if any("l23" in c.lower() for c in cands) else os.path.dirname(cands[0])
print("L23_DIR =", L23_DIR, flush=True)

OUT_PATH = "/kaggle/working/submission.json"
FastVisionModel.for_inference(model)

CLASSES = ["normal","traffic_accident","traffic_congestion","stalled_or_broken_down_vehicle",
           "vehicle_blocking_traffic","wrong_way_driving","road_spill_or_debris",
           "waterlogging_or_flood","fire","smoke","fighting_or_violence",
           "loitering_or_suspicious_presence"]
INSTRUCTION = ("You are a real-time anomaly detection system analyzing frames sampled "
    "from drone, CCTV, or dashcam footage, in temporal order. Classify the "
    "scene into exactly one of these classes: " + ", ".join(CLASSES) + ". "
    "Respond ONLY with compact JSON: "
    '{"class_name": "<one of the classes>", "is_anomaly": <true|false>, "description": "<one short sentence>"}')
CLASS_RE = re.compile(r'"class_name"\s*:\s*"([a-z_]+)"')

# keep the L1 predictions already computed
existing = {}
if os.path.exists(OUT_PATH):
    old = json.load(open(OUT_PATH))
    existing = {p["video_id"]: p for p in old["predictions"]}
    print(f"loaded {len(existing)} existing predictions (L1 preserved)", flush=True)

windows = [json.loads(l) for l in open(os.path.join(L23_DIR, "windows.jsonl")) if l.strip()]
by_video, levels = {}, {}
for w in windows:
    by_video.setdefault(w["video_id"], []).append(w)
    levels[w["video_id"]] = w["level"]
print(f"{len(windows)} L2/L3 windows across {len(by_video)} videos", flush=True)

MIN_CONSEC = 2      # with contiguous 10s windows this means >=20s of sustained signal
T_START = time.time()
results = {}

def classify(images_rel):
    imgs = [Image.open(os.path.join(L23_DIR, p)).convert("RGB") for p in images_rel]
    msgs = [{"role":"user","content":[{"type":"image","image":im} for im in imgs]
             + [{"type":"text","text":INSTRUCTION}]}]
    text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = tokenizer(imgs, text, add_special_tokens=False, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=60, use_cache=True, temperature=0.1)
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    m = CLASS_RE.search(gen)
    return m.group(1) if m else "normal"

def assemble(vid):
    ws = sorted(results.get(vid, []), key=lambda x: x["window_index"])
    events, run = [], []
    for w in ws + [None]:
        if w is not None and w["pred"] != "normal" and (not run or run[-1]["pred"] == w["pred"]):
            run.append(w); continue
        if len(run) >= MIN_CONSEC:
            events.append({"class_name": run[0]["pred"],
                            "start_time_sec": round(run[0]["start_sec"], 2),
                            "end_time_sec": round(run[-1]["end_sec"], 2)})
        run = [w] if (w is not None and w["pred"] != "normal") else []
    return events

ALL_VIDS = [f"T{i:03d}" for i in range(1, 35)]
L1_VIDS = {f"T{i:03d}" for i in range(1, 25)}   # T001-T024 are level 1

def write_submission():
    preds = []
    for vid in ALL_VIDS:
        if vid in results:
            ws = results[vid]
            preds.append({"video_id": vid, "events": assemble(vid),
                "runtime_metadata": {"frames_processed": sum(len(w["images"]) for w in ws),
                    "chunks_processed": len(ws),
                    "end_to_end_internal_time_ms": sum(w["ms"] for w in ws),
                    "model_runtimes": []}})
        elif vid in existing:
            e = existing[vid]
            # L1 is not temporally scored -> arena requires null time fields
            if vid in L1_VIDS:
                for ev in e.get("events", []):
                    ev["start_time_sec"] = None
                    ev["end_time_sec"] = None
            preds.append(e)
        else:
            preds.append({"video_id": vid, "events": [],
                "runtime_metadata": {"frames_processed": 0, "chunks_processed": 0,
                    "end_to_end_internal_time_ms": 0, "model_runtimes": []}})
    json.dump({"schema_version":"1.0","submission_id":"akhilesh-qwen25vl3b-lora-run01",
        "model_name":"Qwen2.5-VL-3B-Instruct + LoRA (Unsloth QLoRA)",
        "run_metadata":{"total_wall_time_ms":int((time.time()-T_START)*1000),
            "max_parallel_videos":1,
            "hardware":"Kaggle Tesla T4 (fp16); deployable artifact is Q4_K_M GGUF for RTX 3050 via llama.cpp"},
        "predictions":preds}, open(OUT_PATH,"w"), indent=2)

# make sure all 34 videos are present even before any L2/L3 work completes
write_submission()
print("baseline submission written (all 34 videos present)", flush=True)

for vid in sorted(by_video):
    t0 = time.time()
    for w in sorted(by_video[vid], key=lambda x: x["window_index"]):
        tw = time.time()
        results.setdefault(vid, []).append({**w, "pred": classify(w["images"]),
                                            "ms": int((time.time()-tw)*1000)})
    evs = assemble(vid)
    print(f"{vid} L{levels[vid]}: {len(by_video[vid])} windows in {time.time()-t0:.0f}s -> {len(evs)} events {evs[:3]}", flush=True)
    write_submission()
    # dump raw per-window predictions so assembly can be re-tuned offline
    # WITHOUT paying for inference again
    json.dump({v: [{k: w[k] for k in ("window_index","start_sec","end_sec","pred","ms")}
                    for w in ws] for v, ws in results.items()},
              open("/kaggle/working/raw_window_preds.json", "w"), indent=1)

print(f"ALL DONE in {time.time()-T_START:.0f}s", flush=True)
display(FileLink(OUT_PATH))
