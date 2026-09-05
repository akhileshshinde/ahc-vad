# AHC Visual Intelligence Hackathon — Real-Time Video Anomaly Detection

Detecting 12 classes of anomalies (accidents, congestion, fire, flooding, fighting,
loitering, …) in drone / CCTV / dashcam footage, with a **small VLM that runs locally
on a 4GB laptop GPU**.

Hardware target: **RTX 3050 Laptop, 3.81GB usable VRAM**. That constraint drove
every architectural decision below.

---

## Results

Benchmark score progression on the public 34-video test set:

| Submission | L1 (25) | L2 (35) | L3 (40) | Total |
|---|---|---|---|---|
| Qwen2.5-VL-3B + LoRA, L1 only | 16.0 | 11.7 | 0.0 | **27.7** |
| + schema fix, full 34 videos | 15.3 | 17.5 | 0.0 | **32.8** |
| Hybrid (VLM L1 + CLIP probe L2/L3) | 16.0* | 11.7* | 6.7* | *est. ~34* |

`*` local-scorer estimates; our scorer is measurably stricter than the arena
(it scored the same L1 predictions at 13.5 where the arena gave 16.0).

Detector behaviour at 32.8: precision 50%, recall 24%, F1 32%, 11 false alarms.

---

## Approach

Two-stage by design: **a per-window classifier, plus separate temporal assembly.**
The model never learns event boundaries — all timing logic lives in post-processing,
which means it can be re-tuned offline for free without re-running inference.

```
video ──► frame sampling ──► per-window classifier ──► temporal assembly ──► events
          (ffmpeg, 384px)     (VLM or CLIP probe)      (smooth/threshold/
                                                        bridge/pad)
```

### Model 1 — Qwen2.5-VL-3B-Instruct + LoRA (Unsloth QLoRA)

- 4-bit frozen base, **vision encoder frozen**, LoRA on language layers only
- **41,084,928 / 3,795,707,904 params trainable (1.08%)**
- Trained on Kaggle T4; merged to fp16, then converted to GGUF for local serving
- Deployable artifact: **Q4_K_M 1.8GB + mmproj 1.3GB** — fits the 3.81GB budget

### Model 2 — CLIP ViT-L/14 latent probe (Alert-CLIP inspired)

Alert-CLIP (CVPR 2026) has no code release, so we implemented its transferable ideas:
frozen CLIP features + a trained linear head, class-weighted against false alarms,
with explicit tracking of the semantically-adjacent confusion pairs.

- **~22ms per window vs ~10s for the VLM (≈400x faster)** — 46 img/s on the RTX 3050
- This speed is what makes a **360-configuration assembly sweep** affordable

---

## What we learned

**1. fp16 on a T4 diverges at standard LoRA learning rates.**
`lr=2e-4` produced NaN loss at step 30 and never recovered. T4 (Turing) has no bf16
tensor cores, so fp16 is forced. Fixed with `lr=5e-5` + `max_grad_norm=0.3`.

**2. Kaggle's API always assigns P100; only the browser session gets T4.**
Modern PyTorch dropped Pascal (`sm_60`) support, so every API-pushed kernel died with
"no kernel image is available for execution on the device". Confirmed 3x. Headless
automation was impossible; the interactive session was the only viable path.

**3. The L3 failure was over-firing, not misclassification.**
On T031 the probe correctly predicted `traffic_congestion` on **109 of 120 windows** —
but the true event is only 235-360s of a 360s video. Assembly produced one giant
interval scoring **IoU 0.35**, failing the 0.5 gate. Fixing assembly alone
(min_consec + 15s gap bridging) took L3 from **0 → 6.7/40** with no model change.

**4. Validation accuracy lied — evidence of source contamination.**
The CLIP probe scored **74.6% on our held-out val split but only 12/24 on the real
L1 test set**. Our split is by video, not by *source*, so a model that latches onto
source-specific cues (watermarks, encoding artifacts) looks great on val and fails to
transfer. A source-aware split is needed to detect this.

**5. Two independent architectures fail identically on `fighting_or_violence`.**
Both the VLM and the CLIP probe collapse it into `loitering_or_suspicious_presence`
on the same videos (T021, T022). That's a property of the data, not either model —
these classes are near-identical in sampled stills and need motion features or
hard-negative training to separate.

---

## Repository layout

```
scripts/
  build_dataset.py          frame extraction + class balancing from raw videos
  build_test_windows.py     sliding-window frames for the test set
  build_l23_fine.py         finer 6s/3s grid for L2/L3 temporal localisation
  make_notebook.py          generates the Kaggle training notebook
  clip_encode.py            CLIP feature extraction (cached to .npz)
  clip_probe.py             zero-shot head + linear probe, confusion analysis
  clip_submit.py            assembly sweep -> scored submission
  tune_l1.py                L1 decision-rule sweep (pooling, normal bias)
  merge_best.py             per-level best-of merge (VLM L1 + CLIP L2/L3)
  score_submission.py       local scorer replicating the arena rules
  retune_assembly.py        offline assembly re-tuning from cached predictions
  fix_submission.py         schema normalisation (L1 requires null timestamps)
inference/
  serve_model.sh            llama-server with GGUF + mmproj, full GPU offload
  vad_pipeline.py           windowed local inference against llama-server
notebooks/
  ahc_vad_qwen25vl_train.ipynb   Unsloth QLoRA training (Kaggle)
```

## Pipeline

```bash
# 1. frames from raw videos, balanced across classes
python scripts/build_dataset.py --data_root <dataset>/train \
    --out_dir frames --max_videos_per_class 200

# 2. train on Kaggle (notebook generated by make_notebook.py)

# 3. local CLIP path — everything below runs on the RTX 3050
python scripts/clip_encode.py --manifest frames/train_manifest.jsonl \
    --root frames --out clip_train.npz
python scripts/clip_probe.py --train clip_train.npz --val clip_val.npz
python scripts/clip_submit.py --test_npz clip_test_*.npz --out submission.json

# 4. score locally before spending a submission attempt
python scripts/score_submission.py --submission submission.json
```

## Known limitations

- L2 credit still comes only from correctly-empty normal videos; the 14 real L2 events
  remain unmatched
- T034 never got frames extracted (ran out of time), so it submits empty
- Training was cut to 100 steps (~0.5 epochs of a capped 1,665-example set) for the deadline
- Per-window classification carries no cross-window context, which is the root cause of
  fragmented intervals
