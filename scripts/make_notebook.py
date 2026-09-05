#!/usr/bin/env python3
"""Assemble the Kaggle training notebook as .ipynb JSON (no nbformat dep needed)."""
import json
from pathlib import Path

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}

cells = []

cells.append(md("""# AHC Visual Intelligence Hackathon — Qwen2.5-VL-3B QLoRA fine-tune

Fine-tunes `unsloth/Qwen2.5-VL-3B-Instruct` on the AHC video anomaly dataset
(12 classes) using Unsloth QLoRA, then merges the adapter and converts to
GGUF (llama.cpp) for real-time local inference on an RTX 3050 4GB laptop GPU.

Pipeline: install -> load 4bit model -> LoRA -> build dataset from attached
frames dataset -> train -> quick val accuracy -> merge to fp16 -> convert to
GGUF (text + mmproj) -> quantize text tower to Q4_K_M -> zip outputs for
download.

Every step below prints a `[LOG hh:mm:ss]` line as it starts/finishes, and
long shell commands stream their output live instead of buffering until the
cell finishes, so progress is visible the whole way through instead of the
cell just looking "stuck"."""))

cells.append(code("""import time, sys

_T0 = time.time()
def log(msg):
    elapsed = time.time() - _T0
    print(f"[LOG {time.strftime('%H:%M:%S')} +{elapsed:7.1f}s] {msg}", flush=True)

def run_streamed(cmd, cwd=None):
    \"\"\"Run a shell command, streaming stdout/stderr live line-by-line
    (a %%bash cell buffers all output until the whole cell finishes, which
    makes long steps like git clone / cmake build look hung).\"\"\"
    import subprocess
    log(f"RUN: {cmd}")
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"command failed (exit {proc.returncode}): {cmd}")
    log(f"DONE: {cmd}")
"""))

cells.append(code("""log("STEP: pip install unsloth")
run_streamed("pip install -q -U unsloth unsloth_zoo")
log("STEP DONE: pip install unsloth")
"""))

cells.append(code("""import torch, os, json, random
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
log("imports done")
"""))

cells.append(md("## Load base model (4-bit) and attach LoRA"))

cells.append(code("""log("STEP: loading base model (this downloads ~2-3GB from HF, can take a few minutes)")
from unsloth import FastVisionModel

MODEL_NAME = "unsloth/Qwen2.5-VL-3B-Instruct"

model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_NAME,
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)
log("STEP DONE: base model loaded")
"""))

cells.append(code("""log("STEP: attaching LoRA adapters")
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False,   # frozen encoder (per hackathon primer)
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    target_modules = "all-linear",
)
log("STEP DONE: LoRA attached")
"""))

cells.append(md("""## Build the dataset

Attach the `ahc-vad-frames` Kaggle Dataset (pre-extracted frames +
`train_manifest.jsonl` / `val_manifest.jsonl`, built locally from the raw
AHC videos) as a notebook input. Each manifest row is one training example:
a handful of sampled frames from one video/event, and the target is the
ground-truth class + a short JSON description.

Built with a list comprehension, not `dataset.map()` — mapping breaks on
multi-image samples (per hackathon primer tip)."""))

cells.append(code("""log("STEP: loading manifests")
DATA_DIR = "/kaggle/input/ahc-vad-frames"
assert os.path.exists(DATA_DIR), f"Attach the ahc-vad-frames dataset as a notebook input first. Looked in {DATA_DIR}"

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

def load_manifest(name):
    path = os.path.join(DATA_DIR, name)
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

train_rows = load_manifest("train_manifest.jsonl")
val_rows = load_manifest("val_manifest.jsonl")
print("train examples:", len(train_rows), " val examples:", len(val_rows))
log("STEP DONE: manifests loaded")
"""))

cells.append(code("""log("STEP: building chat-format training samples")
from PIL import Image

def make_target_json(r):
    desc = (r.get("description_summary") or "").strip()
    if not desc:
        desc = r["class_name"].replace("_", " ")
    return json.dumps({
        "class_name": r["class_name"],
        "is_anomaly": bool(r["is_anomaly"]),
        "description": desc,
    }, ensure_ascii=False)

def to_sample(r):
    images = [Image.open(os.path.join(DATA_DIR, p)).convert("RGB") for p in r["images"]]
    user_content = [{"type": "image", "image": img} for img in images]
    user_content.append({"type": "text", "text": INSTRUCTION})
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": make_target_json(r)}]},
        ]
    }

train_dataset = [to_sample(r) for r in train_rows]
val_dataset_raw = val_rows  # keep raw rows for the eval loop below
print("built", len(train_dataset), "train chat samples")
log("STEP DONE: dataset built")
"""))

cells.append(md("## Train"))

cells.append(code("""log("STEP: setting up trainer")
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
from unsloth.trainer import UnslothVisionDataCollator

FastVisionModel.for_training(model)

collator = UnslothVisionDataCollator(
    model, tokenizer,
    train_on_responses_only = True,
    instruction_part = "<|im_start|>user\\n",
    response_part = "<|im_start|>assistant\\n",
)

class HeartbeatCallback(TrainerCallback):
    \"\"\"Prints a log line on every step so a slow/compiling first step is
    visibly distinguishable from a genuinely hung one.\"\"\"
    def on_step_end(self, args, state, control, **kwargs):
        log(f"train step {state.global_step}/{state.max_steps}"
            + (f"  loss={state.log_history[-1].get('loss')}" if state.log_history else ""))
        return control

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = collator,
    train_dataset = train_dataset,
    callbacks = [HeartbeatCallback()],
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.03,
        max_steps = 100,  # time-boxed for the hackathon deadline; not full epochs
        learning_rate = 5e-5,  # lowered from 2e-4: that value diverged to NaN loss around step 30 on fp16/T4
        max_grad_norm = 0.3,   # tighter clipping, same NaN-divergence fix
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 5,
        save_strategy = "no",
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = None,  # hackathon primer: fixed max_length silently truncates image tokens
    ),
)

log(f"STEP: training starting ({len(train_dataset)} examples, first step includes compile warm-up so it will be the slowest)")
trainer_stats = trainer.train()
log("STEP DONE: training finished")
"""))

cells.append(md("## Save LoRA adapter"))

cells.append(code("""log("STEP: saving LoRA adapter")
model.save_pretrained("/kaggle/working/lora_adapter")
tokenizer.save_pretrained("/kaggle/working/lora_adapter")
print("saved LoRA adapter")
log("STEP DONE: LoRA adapter saved")
"""))

cells.append(md("""## Quick validation

Sanity-check exact-match class accuracy on the held-out val split before
spending time on merge/GGUF conversion."""))

cells.append(code("""log("STEP: running quick validation")
import re

FastVisionModel.for_inference(model)

def parse_class(text):
    m = re.search(r'"class_name"\\s*:\\s*"([a-z_]+)"', text)
    return m.group(1) if m else None

correct, total = 0, 0
confusion = {}
VAL_CAP = 30  # cut down from 200 to save time under the hackathon deadline
for i, r in enumerate(val_dataset_raw[:VAL_CAP]):
    if i % 20 == 0:
        log(f"validation {i}/{min(len(val_dataset_raw), VAL_CAP)}")
    images = [Image.open(os.path.join(DATA_DIR, p)).convert("RGB") for p in r["images"]]
    messages = [{"role": "user", "content": [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": INSTRUCTION}]}]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(images, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=100, use_cache=True, temperature=0.1)
    gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    pred = parse_class(gen_text)
    gt = r["class_name"]
    total += 1
    correct += int(pred == gt)
    confusion.setdefault(gt, {}).setdefault(pred, 0)
    confusion[gt][pred] += 1

print(f"val accuracy: {correct}/{total} = {correct/max(total,1):.3f}")
print(json.dumps(confusion, indent=2))
log("STEP DONE: validation finished")
"""))

cells.append(md("## Merge LoRA into base weights (fp16 safetensors)\n\nNote: Unsloth's built-in `save_pretrained_gguf()` is currently broken for Qwen2.5-VL (open issue upstream) so we merge to safetensors here and convert with a fresh llama.cpp checkout below instead of relying on it."))

cells.append(code("""log("STEP: merging LoRA into base weights (fp16)")
merged_dir = "/kaggle/working/merged_model"
model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
print("merged model saved to", merged_dir)
log("STEP DONE: merge finished")
"""))

cells.append(md("""## Convert to GGUF + quantize (llama.cpp)

Qwen2.5-VL needs two GGUF files: the text tower and a separate `mmproj`
(vision projector) file, produced by running `convert_hf_to_gguf.py` twice.
The text tower is then quantized to Q4_K_M for RTX 3050 (4GB) inference;
mmproj is kept at f16 (vision towers aren't well supported below 8-bit in
llama.cpp)."""))

cells.append(code("""log("STEP: cloning + building llama.cpp (CPU-only, just need llama-quantize)")
run_streamed("git clone --depth 1 https://github.com/ggml-org/llama.cpp", cwd="/kaggle/working")
run_streamed("pip install -q -r llama.cpp/requirements.txt", cwd="/kaggle/working")
run_streamed("cmake -B llama.cpp/build -S llama.cpp -DGGML_CUDA=OFF -DLLAMA_CURL=OFF", cwd="/kaggle/working")
run_streamed("cmake --build llama.cpp/build --target llama-quantize -j 4", cwd="/kaggle/working")
log("STEP DONE: llama.cpp built")
"""))

cells.append(code("""log("STEP: converting to GGUF + quantizing")
run_streamed("python llama.cpp/convert_hf_to_gguf.py merged_model --mmproj --outfile mmproj-qwen25vl3b-vad-f16.gguf --outtype f16", cwd="/kaggle/working")
run_streamed("python llama.cpp/convert_hf_to_gguf.py merged_model --outfile qwen25vl3b-vad-f16.gguf --outtype f16", cwd="/kaggle/working")
run_streamed("./llama.cpp/build/bin/llama-quantize qwen25vl3b-vad-f16.gguf qwen25vl3b-vad-Q4_K_M.gguf Q4_K_M", cwd="/kaggle/working")
run_streamed("ls -lh /kaggle/working/*.gguf")
log("STEP DONE: GGUF conversion + quantization finished")
"""))

cells.append(md("## Package outputs for download"))

cells.append(code("""log("STEP: cleaning up intermediate files")
f16_path = "/kaggle/working/qwen25vl3b-vad-f16.gguf"
if os.path.exists(f16_path):
    os.remove(f16_path)  # keep only the quantized text model + mmproj + lora adapter
run_streamed("du -sh /kaggle/working/lora_adapter /kaggle/working/mmproj-qwen25vl3b-vad-f16.gguf /kaggle/working/qwen25vl3b-vad-Q4_K_M.gguf")
log("ALL DONE")
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path("/home/akhilesh/ahc-hackathon/notebooks/ahc_vad_qwen25vl_train.ipynb")
out_path.write_text(json.dumps(nb, indent=1))
print("wrote", out_path)
