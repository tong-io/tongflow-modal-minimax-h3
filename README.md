# tongflow-modal-minimax-h3

Self-hosted [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) for
[TongFlow](https://github.com/tong-io/tongflow): 33B omni-modal video
generation with **native stereo audio** (24 fps, 768 short edge, ~5–15 s),
served from headless ComfyUI (native H3 nodes, no custom node packs) on
[Modal](https://modal.com), using the
[Comfy-Org optimized weights](https://huggingface.co/Comfy-Org/MiniMax-H3)
(pruned int8 ConvRot, ~63 GB total instead of 498 GB full precision).

Runs entirely inside **your own** Modal account. See
[When to use this plugin](#when-to-use-this-plugin) before adopting it — for
plain speed or cost, fal's hosted H3 Max wins outright.

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

**Prompt structure** (the format LightX2V documents for H3; freeform prompts
also work, but this is what the model was tuned on):

```text
integrated_multimodal_description: [Shot 1] Visual style, subject, action,
camera, lighting, and dialogue for each shot.

overall_soundscape: Dialogue, ambient sound, and synchronized effects.

non_diegetic_music: Background score, or N/A for none.
```

Dialogue is written as `<d>[English] Line of dialogue.</d>`. For I2V,
keep first-frame identity with e.g. `at 0.00 seconds into the target video,
<Picture 1> (from [Shot 1]) is fully referenced.`

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
| `loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | 2.0 GB |
| `loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | 2.0 GB |
| `loras/minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` | 2.0 GB |
| `loras/MiniMax-H3-FL2VA-Acc-8Step_pruned_comfy.safetensors` | 1.7 GB |
| `loras/MiniMax-H3-Ref2VA-Acc-8Step_pruned_comfy.safetensors` | 1.7 GB |

### When to use this plugin

Self-hosting H3 is **not the cheap or the fast way to generate video**, and this
plugin should not be picked on either of those grounds. fal serves *H3 Max* —
their post-trained H3 — through the
[fal router plugin](https://github.com/tong-io/tongflow-router-fal) at
**$0.08/s of output** with native audio, returning a 5 s 768p clip in about
3 seconds. That is roughly the same price as a self-hosted clip and ~100×
faster, from a model that scores better than stock H3.

Reach for self-hosting when one of these is the point:

- **Data residency** — inputs and outputs never leave your own Modal account.
- **No third-party dependency** — no per-account rate limits, no vendor
  deprecating an endpoint out from under a saved workflow.
- **Modal's $30/month free credit** — a general GPU allowance shared across
  *every* official Modal plugin, not just this one. Around 35–40 ten-second
  clips a month land inside it.
- **The full Ref2VA surface** — `refs-gen-video` mixes up to 9 images, 3 videos
  and 3 audio clips in one context. fal's reference-to-video endpoint is
  announced but not shipped as of 2026-08-27.

### Picking a GPU (once you've decided to self-host)

Default is **B200** ($6.25/h) with the NVFP4 text encoder. Blackwell runs the
int8-convrot / NVFP4 kernels ~2× faster than Hopper, so Hopper cards lose on
both price and speed; the real choice is B200 vs RTX-PRO-6000.

| GPU | 10 s clip | 15 s clip | Notes |
|---|---|---|---|
| **B200 $6.25/h (default)** | **6 m 52 s (~$0.72) ✓**<br>**3 m 46 s (~$0.39) with `H3_PDD=1` ✓** | **11 m 24 s (~$1.19) ✓** | fastest; both checkpoints resident in 192 GB |
| RTX-PRO-6000 $3.03/h | 18 m 11 s ($0.92) ✓ | — | ~45% slower and, at these rates, no longer cheaper per clip; Blackwell 96 GB, nvfp4 native |
| H100 $3.95/h + int8 TE | 20 m 25 s ($1.34) ✓ | — | dominated on both axes |
| A100-80GB $2.50/h + int8 TE | >40 min (aborted) ✓ | ⚠️ times out | not recommended |

(✓ = measured on Ref2VA at 1344×768, un-distilled 20-step unless the row says
otherwise. `H3_PDD=1` was measured 2026-09-09 against the same seed, prompt and
reference image: **226 s vs 418 s, 1.85×**, with the audio track intact
(`max_volume −27.1 dB`, real dynamic range — not the constant-DC failure of
#15799). The B200 row is from
2026-09-09 on the current pin; dropping upstream's `v = v.clone()` bought ~9%
over the 2026-08-25 measurement of the same clip. The other rows are still
v0.30.0-era and should be re-measured before being trusted.) Duration scales
close to linearly here — 362 frames costs 1.66× what 243 frames does, for
1.49× the frames — so the earlier "15 s is ~2× a 10 s clip" estimate was
pessimistic by nearly half. MiniMax's sparse-attention implementation
(promised, unreleased) would still help most at the long end.

One checkpoint + text encoder + VAEs fit in 80 GB; on A100/H100, switching
between FL2VA and Ref2VA slots reloads ~21 GB from the volume (tens of
seconds).

Env knobs are read when `modal deploy` runs and **baked into the image** so the
container sees them: Modal re-imports this module inside the container, where
the deploy shell's environment does not exist, so a module-level
`os.environ.get()` would otherwise always fall back to its default. Flipping a
knob rebuilds one image layer. (TongFlow Settings; see "Applying env changes"
below):
`H3_GPU` (B200), `H3_TEXT_ENCODER_VARIANT` (nvfp4|int8), `H3_SHORT_EDGE` (768;
lower it for faster drafts), `H3_STEPS` (20), and the FL2VA turbo family:
`H3_TURBO` (off; set `1` to run the three FL2VA slots with the
[LightX2V Turbo distill LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
at `H3_TURBO_STEPS` (8) plain Euler — ~2.5× fewer sampling steps),
`H3_TURBO_LORA`, `H3_TURBO_STRENGTH` (1.0).

**Turbo scope & status:** `H3_TURBO=1` covers the FL2VA slots
(`text-gen-video`, `image-gen-video`, `image-image-gen-video`) with the 8-step
v1.0 LoRA (shipped 2026-08-11). `H3_TURBO_REF=1` separately covers the Ref2VA
slots — including the default `refs-gen-video` — with the **Ref2VA Turbo
4-step v0.1** LoRA (shipped 2026-08-13; a generation younger, and upstream's
example pairs it with the full bf16 base rather than our pruned int8, so treat
it as experimental). **A/B the same seed against the un-distilled path before
leaving either on** (known distill trade-off: quiet/sustained vocals degrade
first).

**`H3_PDD=1` is the other family** — Alibaba PAI's
[PDD Acc LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs)
(Parallel Decoding Distillation), **8 NFE for FL2VA *and* Ref2VA**. It is the
only 8-step option for the default `refs-gen-video` slot, where LightX2V still
ships just a 4-step v0.1. These are not ordinary LoRAs: a rank-64 backbone
update ships with a 32-interval bank of output heads, and each sampler step
consumes the dt-weighted mean of the heads it spans. ComfyUI detects the bank
from the `[N*out, in]` weight shape (#15908) so the stock loader reads it, but
the graph must keep H3's **native 12/3 shifts** — the bank indexes its interval
grid by them, and a wrong pair blends the wrong heads silently. Distillations
do not stack, so `H3_PDD` refuses to coexist with `H3_TURBO*` and raises at
deploy time.

The FL2VA default is the **768p-trained** 8-step build (shipped 2026-08-27):
same distillation NFE as the original 8-step LoRA but trained at 1344×768,
which is exactly this plugin's output canvas, so nothing is extrapolated from
544p. Both graphs carry an explicit `MiniMaxH3SigmaShift`, and the shift pair
is **derived from the active LoRA** — 6/3 for the 768p-trained variants, 12/3
(H3 native) for everything else, including the Ref2VA LoRA and the
un-distilled path. `H3_SHIFT_VIDEO` / `H3_SHIFT_AUDIO` override that, and are
only needed for a LoRA `deploy.py` does not know about.

## Test runbook (run these yourself — every step below bills Modal)

Use the venv modal client (`sdk/.venv/bin/modal` from the tongflow repo) with
`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` exported, from this plugin directory.

### 1. Download weights (~65 GB, CPU-only, one-off)

```bash
modal run download.py::download
modal volume ls models comfyui/diffusion_models
modal volume ls models comfyui/text_encoders
modal volume ls models comfyui/vae
modal volume ls models comfyui/loras
```

Expect the six files above with matching sizes. Re-running skips existing
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
- Also confirm the mp4 has **real** audio, not just an audio stream. A present
  but constant-DC / silent track is a known upstream failure mode (#15799),
  and it also breaks the `SaveVideo` mux on some setups:

  ```bash
  ffprobe out.mp4                      # expect 1 video + 1 stereo audio stream
  ffmpeg -i out.mp4 -af volumedetect -f null -   # mean_volume must not be -inf
  ```

  `mean_volume: -inf dB` (silence) or `max_volume: 0.0 dB` with zero dynamic
  range (full-scale DC) means the audio VAE path is broken, not the prompt.

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

- **B200 + nvfp4 TE (default): 5 s in 4 m 17 s ($0.45); 10 s in
  10 m 04 s ($1.05)** — fastest. Output quality verified comparable to
  Seedance 2.0.
- RTX-PRO-6000 + nvfp4 TE: 10 s in 18 m 11 s ($0.92) — the budget option.
- H100 + int8 TE: 10 s in 20 m 25 s ($1.34) — slower and pricier than
  RTX-PRO-6000; only a quota fallback.
- A100-80GB: a 15 s clip exceeded 40 min and was aborted — not recommended.
- Full attention scales superlinearly with duration (10 s costs ~2.3× a 5 s
  clip); MiniMax's unreleased sparse attention is the long-clip fix.
- 15 s extrapolates to roughly 12–15 min (full attention scales superlinearly;
  not yet measured — see runbook step 4).
- For faster drafts lower `H3_SHORT_EDGE` (e.g. 512) — generation time scales
  roughly with pixel count.

## Known gaps / notes

- **ComfyUI is pinned to a master commit (`15eb748b`), not a release tag.**
  The v0.33.x / v0.34.x tags are narrow backports that carry the tokenizer fix
  but none of the H3 work below, and the last published release is v0.34.0:
  - **#15808** tokenizer special tokens. H3's `tokenizer_config` declares
    `<d>`, `</d>`, `<|cutoff|>`, `<|lyrics_*|>`, `<|caption_*|>` but
    `tokenizer.json` does not, so `<d>` used to tokenize as three ordinary
    characters and dialogue markup silently did nothing.
  - **#15908** PDD acceleration LoRAs (see `H3_PDD` above).
  - **#15975 / #16020** Fun ControlNet as a model patch, and letting it
    coexist with reference conditioning. Not wired up here yet.
  - **#16065** VAE optional / text-encoder-only references.
  - **#16103** removes the `v = v.clone()` memory workaround, which cost up
    to 4× at full resolution (#15665). This build used to delete that line
    itself; now it only asserts the line stays gone, so a revert or a
    careless pin bump fails the build instead of quietly returning a 4×
    slower — and 4× more expensive — deploy.
  - Audio sampling semantics changed back at the v0.30.0 → v0.32.0 move —
    **A/B one clip's audio after upgrading**, and check the track is not
    constant-DC or silent (#15799 reports that on some setups).
  - Sol-Attn is pinned to its 2026-08-08 commit.
- The Ref2VA graph wires references via ComfyUI autogrow inputs
  (`ref_images.ref_image_0` …) — validated against the v0.30.0 template
  serialization, but the first live run is the real test (a rejection error
  will name the offending input).
- UI thumbnails label references as "Image 1 / 图1"; the model expects
  `<Picture 1>`-style tags in the prompt. Clicking a thumbnail inserts the UI
  label — type the `<Picture n>` form for strongest adherence.
- Generation graphs mirror the official `video_minimax_h3_*` templates
  (res_multistep / simple / 20 steps / CFG-distilled, no negative prompt).
