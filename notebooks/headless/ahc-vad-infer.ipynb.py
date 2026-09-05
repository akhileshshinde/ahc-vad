# Headless L2/L3 inference kernel: loads base model + our LoRA adapter,
# classifies the coarse window set, writes submission.json.
# Prioritises L3 (0/40 currently, no normal videos -> pure upside) before L2.
import os, sys, json, re, time, glob, zipfile, subprocess

T_START = time.time()
def log(m): print(f"[{time.time()-T_START:7.1f}s] {m}", flush=True)

import torch
log(f"torch {torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
cap = torch.cuda.get_device_capability(0)
log(f"compute capability sm_{cap[0]}{cap[1]}")
if cap[0] < 7:
    log("FATAL: pre-Volta GPU (P100/sm_60) - modern torch has no kernels for it. Aborting.")
    sys.exit(1)

log("installing unsloth")
subprocess.run("pip install -q -U unsloth unsloth_zoo", shell=True, check=True)

from unsloth import FastVisionModel
from peft import PeftModel
from PIL import Image

log("loading base model")
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-3B-Instruct", load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

# unpack + attach our fine-tuned adapter
ad_zip = glob.glob("/kaggle/input/**/lora_adapter.zip", recursive=True)[0]
with zipfile.ZipFile(ad_zip) as z:
    z.extractall("/kaggle/working/lora_adapter")
log(f"adapter unpacked from {ad_zip}")
model = PeftModel.from_pretrained(model, "/kaggle/working/lora_adapter")
FastVisionModel.for_inference(model)
log("adapter attached")

# window set
wz = glob.glob("/kaggle/input/**/ahc_l23_coarse.zip", recursive=True)
if wz:
    with zipfile.ZipFile(wz[0]) as z:
        z.extractall("/kaggle/working/")
WDIR = os.path.dirname(glob.glob("/kaggle/working/**/ahc_l23_coarse/windows.jsonl", recursive=True)[0])
windows = [json.loads(l) for l in open(f"{WDIR}/windows.jsonl") if l.strip()]
by_video, levels = {}, {}
for w in windows:
    by_video.setdefault(w["video_id"], []).append(w)
    levels[w["video_id"]] = w["level"]
log(f"{len(windows)} windows / {len(by_video)} videos from {WDIR}")

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

def classify(rel):
    imgs = [Image.open(os.path.join(WDIR, p)).convert("RGB") for p in rel]
    msgs = [{"role":"user","content":[{"type":"image","image":i} for i in imgs]
             + [{"type":"text","text":INSTRUCTION}]}]
    txt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
    inp = tokenizer(imgs, txt, add_special_tokens=False, return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=40, use_cache=True, temperature=0.1)
    gen = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    m = CLASS_RE.search(gen)
    return m.group(1) if m else "normal"

results = {}
def dump():
    json.dump(results, open("/kaggle/working/raw_window_preds.json", "w"))

# L3 first: 40 marks fully untapped and no normal videos to false-alarm on
order = sorted(by_video, key=lambda v: (levels[v] != 3, v))
log(f"processing order: {order}")
for vid in order:
    t0 = time.time()
    rows = []
    for w in sorted(by_video[vid], key=lambda x: x["window_index"]):
        tw = time.time()
        rows.append({"window_index": w["window_index"], "start_sec": w["start_sec"],
                      "end_sec": w["end_sec"], "pred": classify(w["images"]),
                      "ms": int((time.time()-tw)*1000)})
    results[vid] = rows
    npos = sum(1 for r in rows if r["pred"] != "normal")
    log(f"{vid} L{levels[vid]}: {len(rows)} windows in {time.time()-t0:.0f}s, {npos} non-normal")
    dump()

log(f"ALL DONE in {time.time()-T_START:.0f}s")
