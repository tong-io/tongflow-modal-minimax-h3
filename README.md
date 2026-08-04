# tongflow-modal-minimax-h3

Self-hosted [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) for
[TongFlow](https://github.com/tong-io/tongflow): 33B omni-modal video
generation with **native stereo audio** (24 fps, 768 short edge, ~5–15 s),
served from headless ComfyUI (native H3 nodes, no custom node packs) on
[Modal](https://modal.com), using the
[Comfy-Org optimized weights](https://huggingface.co/Comfy-Org/MiniMax-H3)
(pruned int8 ConvRot, ~63 GB total instead of 498 GB full precision).

## Slots

| Slot | H3 mode | Inputs |
|---|---|---|
| `text-gen-video` | FL2VA t2va | text |
| `image-gen-video` | FL2VA first frame | text + image |
| `image-image-gen-video` | FL2VA first+last frame | text + 2 images |
| `images-gen-video` | Ref2VA | text + ≤9 reference images |
| `audio-image-gen-video` | Ref2VA | image + audio (duration follows the audio) |
| `refs-gen-video` **(default)** | Ref2VA omni-reference | text + ≤9 images + ≤3 videos + ≤3 audio clips (≤12 files; audio never alone) |

Output is always a single `video/mp4` with the stereo track muxed in.

**Prompting Ref2VA:** address references as `<Picture 1>`, `<Video 1>`,
`<Audio 1>` (1-based, in connection order: images, then videos, then audio).
A reference video's own soundtrack is attached automatically when present.
MiniMax's hosted prompt rewriter (H3-Context-IR) is not open source, so prompts
pass through verbatim — write detailed, cinematic prompts including dialogue /
sound cues; `enhance_prompt` is accepted and ignored.

## Constraints

- Duration is clamped to the trained range **~5.2–15.1 s** (frame grid `17k+5`
  at 24 fps: 124–362 frames). The UI slider allows 1–30 s; out-of-range values
  are clamped, so a 3 s request produces ~5.2 s and 30 s produces ~15.1 s.
- Output canvas: short edge 768 (model native), dimensions ×32, area-capped at
  768×1344. Only the aspect ratio of the requested width×height is used.
- Reference videos: 2–15 s each, ≥5 frames, ideally ~24 fps.
- 768p only — H3-Regenerate-2K (the official 2K refiner) is not open source.

## Files / GPU

Weights land on the shared `models` Modal volume under `/models/comfyui/`:

| File | Size |
|---|---|
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 21.0 GB |
| `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 21.0 GB |
| `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27.1 GB |
| `vae/minimax_h3_video_vae_fp16.safetensors` | 5.2 GB |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 0.6 GB |

Default GPU is **H100** ($3.95/h on Modal) with the int8 text encoder.
**Duration drives the choice**: full attention scales superlinearly, so a
15 s clip costs far more than 3× a 5 s clip. Measured/estimated per clip:

| GPU | 5 s clip | 10 s clip | 15 s clip | Notes |
|---|---|---|---|---|
| A100-80GB $2.50/h | ~8–10 min | >40 min (aborted) | ⚠️ times out | cheapest, **short clips only** |
| **H100 $3.95/h (default)** | ~7 min | **20 min 25 s (measured, $1.34)** | ~35–40 min | balanced |
| B200 $6.25/h + nvfp4 | **4 min 17 s (measured, $0.45)** | ~13 min | ~22–26 min | speed ceiling; both checkpoints resident |

One checkpoint + text encoder + VAEs fit in 80 GB; on A100/H100 switching
between FL2VA and Ref2VA slots reloads ~21 GB from the volume (tens of
seconds). MiniMax's sparse-attention implementation (not yet released) is the
real fix for long clips — we'll adopt it when it lands.

Env knobs (TongFlow Settings; deploy-time — see "Applying env changes" below):
`H3_GPU` (B200), `H3_TEXT_ENCODER_VARIANT` (nvfp4|int8), `H3_SHORT_EDGE` (768;
lower it for faster drafts), `H3_STEPS` (20).

## Test runbook (run these yourself — every step below bills Modal)

Use the venv modal client (`sdk/.venv/bin/modal` from the tongflow repo) with
`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` exported, from this plugin directory.

### 1. Download weights (~63 GB, CPU-only, one-off)

```bash
modal run download.py::download
modal volume ls models comfyui/diffusion_models
modal volume ls models comfyui/text_encoders
modal volume ls models comfyui/vae
```

Expect the five files above with matching sizes. Re-running skips existing
files; a mid-download failure resumes where it stopped (per-file commits).

### 2. Deploy + boot check

```bash
modal deploy deploy.py
```

Then trigger any generation (step 3) and watch `modal app logs
tongflow-modal-minimax-h3`. Boot asserts the H3 node classes exist in
`/object_info` — a `MiniMaxH3ImageToVideo missing` error means the pinned
ComfyUI tag is wrong, not a weights problem. If B200 is unavailable / queues
badly, set `H3_GPU=H100` + `H3_TEXT_ENCODER_VARIANT=int8`, re-run download,
and re-deploy.

### 3. Timed t2v smoke test (the go/no-go gate)

From TongFlow: add a **Text → Video** node, pick the MiniMax-H3 plugin, prompt
e.g. "A corgi runs across a sunny beach, waves crashing, upbeat music",
duration 5 s, 16:9. Record:

- **wall-clock end-to-end** (first run includes ~2–4 min cold boot + model
  load; run a second generation for the steady-state number);
- **generation time** from the logs (`Prompt executed in X seconds`);
- **peak VRAM** from the Modal dashboard GPU memory graph.

Kill criteria:

- **OOM** on B200 at 5 s / 768p → stop; self-hosting is off the table for now
  (report the log tail).
- **Speed**: steady-state 5 s clip > ~10 min generation (≈ $1+/clip, 15 s
  extrapolates to ≳ 40 min) → impractical; consider the MiniMax API route
  instead. Target zone: 5 s clip ≤ 5 min.
- Also confirm the mp4 has audio: `ffprobe out.mp4` → one video + one stereo
  audio stream (or just play it).

### 4. Remaining slots, one clip each

- **I2V**: image + text (image animates from frame 1).
- **First+last frame**: two images; check both endpoints are honored.
- **Images → video**: 2–3 reference images + prompt using `<Picture 1>` tags.
- **Audio + image**: portrait + a short speech clip; duration should follow
  the audio length; check lip/voice sync.
- **Omni-reference** (`refs-gen-video` node): mix e.g. 2 images + 1 video +
  1 audio; also verify the Seedance plugin implements the same node (switch
  plugin in the picker).
- One 15 s run to check VRAM headroom at max length.

### 5. Applying env changes

`entry.py` re-deploys only when `deploy.py`'s content hash changes. After
changing `H3_*` env vars in Settings, force a re-deploy by clearing the cache
entry (`rm ~/.tongflow/modal-cache/tongflow-modal-minimax-h3.json`) or running
`modal deploy deploy.py` manually with the new env exported.

## Measured performance

- **B200 + nvfp4 TE (2026-08-03): 5 s clip @ 768p 16:9 in 4 min 17 s**
  steady-state (≈ $0.45/clip). Output quality verified comparable to
  Seedance 2.0.
- **H100 + int8 TE (2026-08-04): 10 s clip in 20 min 25 s** (≈ $1.34/clip).
- A100-80GB: a 15 s clip exceeded 40 min and was aborted — short clips only.
- Full attention scales superlinearly with duration (10 s costs ~3× a 5 s
  clip, not 2×); MiniMax's unreleased sparse attention is the long-clip fix.
- 15 s extrapolates to roughly 12–15 min (full attention scales superlinearly;
  not yet measured — see runbook step 4).
- For faster drafts lower `H3_SHORT_EDGE` (e.g. 512) — generation time scales
  roughly with pixel count.

## Known gaps / notes

- The Ref2VA graph wires references via ComfyUI autogrow inputs
  (`ref_images.ref_image_0` …) — validated against the v0.30.0 template
  serialization, but the first live run is the real test (a rejection error
  will name the offending input).
- UI thumbnails label references as "Image 1 / 图1"; the model expects
  `<Picture 1>`-style tags in the prompt. Clicking a thumbnail inserts the UI
  label — type the `<Picture n>` form for strongest adherence.
- Generation graphs mirror the official `video_minimax_h3_*` templates
  (res_multistep / simple / 20 steps / CFG-distilled, no negative prompt).
