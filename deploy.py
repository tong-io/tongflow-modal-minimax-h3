"""Modal deploy entry for MiniMax-H3 (headless ComfyUI, native nodes + Sol-Attn).

MiniMax-H3 is a 33B omni-modal DiT that generates video with native stereo
audio (24 fps, 768-short-edge, ~5-15 s). ComfyUI >= v0.30.0 ships native
support; this plugin runs the Comfy-Org optimized weights (pruned int8
convrot, ~21 GB per checkpoint) and serves six video slots from the two
task checkpoints:

- FL2VA (``MiniMaxH3ImageToVideo``): ``text-gen-video`` (no image),
  ``image-gen-video`` (first frame), ``image-image-gen-video`` (first+last).
- Ref2VA (``MiniMaxH3ReferenceToVideo``): ``images-gen-video`` (<=9 images),
  ``audio-image-gen-video`` (image + voice), ``refs-gen-video``
  (omni-reference: <=9 images + <=3 videos + <=3 audio clips, <=12 files).

Inference is accelerated with Sol-Attn (kijai/Triton 3.3+) sparse attention,
yielding ~3.9× speedup on GPU with Triton support (SM75+).

The API graphs are built programmatically (not from a workflow.json): the
official t2v template wraps the pipeline in a UI subgraph, and Ref2VA needs a
variable number of loader nodes anyway. The graph mirrors the official
video_minimax_h3_* templates node-for-node (custom sampling stack with
SolAttnPatch: RandomNoise / res_multistep / simple x20 / BasicGuider /
SamplerCustomAdvanced, then VAEDecode + VAEDecodeAudio -> CreateVideo -> SaveVideo).

Prompting: Ref2VA prompts address references as <Picture i> / <Video k> /
<Audio j> (1-based, in connection order); dialogue uses <d>[Lang] ...</d>.
H3-Context-IR (MiniMax's hosted prompt rewriter) is not open source, so
prompts are passed through verbatim; ``enhance_prompt`` is accepted and
ignored.

ComfyUI is pinned to a master commit rather than a release tag, and the image
build patches out one upstream line — see COMFY_COMMIT below for why.

Env knobs (all optional, read at deploy time — re-deploy after changing):
  H3_GPU                   default "B200" (fastest: 10 s = 10m04 / $1.05).
                           "RTX-PRO-6000" is the budget option (Blackwell
                           96 GB, 10 s = 18m11 / $0.92 — 45% slower, 12%
                           cheaper). "H100"/"A100-80GB" only as quota
                           fallbacks — pair with H3_TEXT_ENCODER_VARIANT=int8;
                           A100 suits short clips only (15 s times out).
  H3_TEXT_ENCODER_VARIANT  "nvfp4" (default, Blackwell/B200 only) or "int8"
                           (works everywhere); must match download.py.
  H3_SHORT_EDGE            canvas short edge, default 768 (model native).
                           Lower (e.g. 512) for faster, cheaper drafts.
  H3_STEPS                 sampling steps, default 20 (official template).
  H3_TURBO                 "1" to enable the LightX2V FL2VA Turbo LoRA
                           (lightx2v/Minimax-h3-Turbo, 8-step v1.0, Apache-2.0)
                           on the three FL2VA slots. Default off — A/B against
                           the 20-step baseline before enabling. Ref2VA slots
                           (incl. the default refs-gen-video) always run the
                           un-distilled 20-step path: no Ref2VA distill exists
                           yet (LightX2V roadmap item 2).
  H3_TURBO_STEPS           FL2VA turbo sampling steps, default 8 (the model's
                           distillation NFE; 4 is valid but softer).
  H3_TURBO_LORA            LoRA filename under loras/, default the 8-step
                           v1.0 ComfyUI export; must match download.py.
  H3_TURBO_STRENGTH        LoRA strength, default 1.0 (tuned value); shared
                           by the Ref2VA turbo LoRA.
  H3_TURBO_REF             "1" to run the Ref2VA slots (incl. the default
                           refs-gen-video) with the LightX2V Ref2VA Turbo
                           4-step v0.1 LoRA. Younger than the FL2VA one
                           (v0.1) and upstream pairs it with the FULL bf16
                           base, not our pruned int8 — A/B before enabling.
  H3_TURBO_REF_STEPS       Ref2VA turbo steps, default 4 (distill NFE).
  H3_TURBO_REF_LORA        Ref2VA LoRA filename, must match download.py.
  H3_SHIFT_VIDEO           video sigma shift, default 12 (H3 native); the
  H3_SHIFT_AUDIO           audio shift, default 3. Set video=6 for the
                           fl2v 768p 4-step LoRA variant.

Deploy:           modal deploy deploy.py
Download weights: modal run download.py::download
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import modal
from tongflow import deploy
from tongflow.models.audio_image_gen_video import (
    AudioImageGenVideoInput,
    AudioImageGenVideoOutput,
)
from tongflow.models.image_gen_video import ImageGenVideoInput, ImageGenVideoOutput
from tongflow.models.image_image_gen_video import (
    ImageImageGenVideoInput,
    ImageImageGenVideoOutput,
)
from tongflow.models.images_gen_video import ImagesGenVideoInput, ImagesGenVideoOutput
from tongflow.models.refs_gen_video import RefsGenVideoInput, RefsGenVideoOutput
from tongflow.models.text_gen_video import TextGenVideoInput, TextGenVideoOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset, prompt_media_to_bytes
from tongflow.slots import node_slot

COMFY = "/opt/ComfyUI"
# Pinned to the master commit that lands kijai's tokenizer fix (#15808): H3's
# tokenizer_config declares 7 extra special tokens (<d>, </d>, <|cutoff|>,
# <|lyrics_*|>, <|caption_*|>) that are absent from tokenizer.json, so before
# this commit `<d>` was tokenized as three ordinary characters and dialogue
# markup silently did nothing. No release tag carries it: v0.33.2/.3/.4 are
# backports of Partner Nodes only and touch no H3 core file.
# Also included since v0.32.0: ModelSamplingAV audio protocol (#15243),
# audio-VAE offload (#15377), VAE optimization (#15446), VAEDecodeTiled crash
# (#15477), peak memory (#15486), per-token AV noise masks (#15375), taeh3.
COMFY_COMMIT = "924743af083c151296cc16f925aeab113b6484e8"
# Pin Sol-Attn: repo has no tags and moves fast; this is the 2026-08-08
# "Fixes and optimizations" commit, after ComfyUI's H3 audio protocol switch.
SOLATTN_COMMIT = "842c4eaa7d91"
COMFY_MODELS = "/models/comfyui"
COMFY_LOG = "/tmp/comfy.log"

# Slots this plugin is the default implementation of. The other five video
# slots stay with the Seedance (bytedance) plugin.
TONGFLOW_DEFAULT_SLOTS = ["refs-gen-video"]

GPU = (os.environ.get("H3_GPU") or "B200").strip()
TE_VARIANT = (os.environ.get("H3_TEXT_ENCODER_VARIANT") or "nvfp4").strip().lower()
SHORT_EDGE = int(os.environ.get("H3_SHORT_EDGE") or 768)
STEPS = int(os.environ.get("H3_STEPS") or 20)

# FL2VA Turbo (LightX2V distill LoRA). Off by default; FL2VA slots only.
TURBO = (os.environ.get("H3_TURBO") or "").strip().lower() in ("1", "true", "on")
TURBO_STEPS = int(os.environ.get("H3_TURBO_STEPS") or 8)
TURBO_LORA = (
    os.environ.get("H3_TURBO_LORA")
    or "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
).strip()
TURBO_STRENGTH = float(os.environ.get("H3_TURBO_STRENGTH") or 1.0)
# Ref2VA Turbo (LightX2V Ref2VA 4-step v0.1, 2026-08-13). Separate knob from
# H3_TURBO: it is a generation younger (v0.1 vs v1.0) and gates our DEFAULT
# slot (refs-gen-video), so it must be opt-in on its own. Caveat: the official
# example workflow pairs this LoRA with the FULL bf16 Ref2VA base, not our
# pruned int8 — compatibility with the pruned base is untested upstream.
TURBO_REF = (
    os.environ.get("H3_TURBO_REF") or ""
).strip().lower() in ("1", "true", "on")
TURBO_REF_STEPS = int(os.environ.get("H3_TURBO_REF_STEPS") or 4)
TURBO_REF_LORA = (
    os.environ.get("H3_TURBO_REF_LORA")
    or "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
).strip()
# Explicit AV sigma shifts on the sampling model, mirroring the official
# LightX2V workflows (MiniMaxH3SigmaShift 12/3 = H3 native defaults; the
# fl2v 768p 4-step variant needs H3_SHIFT_VIDEO=6).
SHIFT_VIDEO = float(os.environ.get("H3_SHIFT_VIDEO") or 12.0)
SHIFT_AUDIO = float(os.environ.get("H3_SHIFT_AUDIO") or 3.0)

FL2VA_UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
REF2VA_UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
TEXT_ENCODER = {
    "nvfp4": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "int8": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
}.get(TE_VARIANT, "qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

FPS = 24
DIM_ALIGN = 32
# Frame count lives on the model's 17k+5 grid; trained range is 124-362 frames
# (~5.2-15.1 s). Requests outside the range are clamped, then snapped up.
MIN_FRAMES = 124
MAX_FRAMES = 362
DEFAULT_FRAMES = 124

# Ref2VA reference caps (model card): <=9 images, <=3 videos, <=3 audio clips,
# <=12 files total; audio must accompany an image or video.
MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3
MAX_REF_TOTAL = 12

volume = modal.Volume.from_name("models", create_if_missing=True)

app = modal.App(Path(__file__).resolve().parent.name)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .run_commands(
        # Shallow-fetch the pinned commit (a SHA, so --branch cannot be used).
        f"git init {COMFY} && "
        f"git -C {COMFY} remote add origin "
        f"https://github.com/comfyanonymous/ComfyUI.git && "
        f"git -C {COMFY} fetch --depth 1 origin {COMFY_COMMIT} && "
        f"git -C {COMFY} checkout FETCH_HEAD",
        # Drop the defensive `v = v.clone()` added by the peak-memory fix
        # (#15486). It detaches v from the fused qkv buffer but keeps the
        # [seq, heads, dim] layout, so the attention backend receives a
        # transposed view and falls off its fast path: ~4x slower at full
        # resolution (#15665, open; the fix PR #15705 was closed unmerged).
        # Grep first so an upstream fix breaks the build loudly instead of
        # silently no-oping this patch.
        f"grep -qx '        v = v.clone()' {COMFY}/comfy/ldm/minimax/model.py && "
        f"sed -i '/^        v = v.clone()$/d' {COMFY}/comfy/ldm/minimax/model.py && "
        f"! grep -q 'v = v.clone()' {COMFY}/comfy/ldm/minimax/model.py",
        f"pip install -r {COMFY}/requirements.txt",
        "pip install --upgrade 'triton>=3.3' --no-deps",
        f"git clone https://github.com/kijai/ComfyUI-SolAttn_triton.git "
        f"{COMFY}/custom_nodes/ComfyUI-SolAttn_triton && "
        f"git -C {COMFY}/custom_nodes/ComfyUI-SolAttn_triton "
        f"checkout {SOLATTN_COMMIT}",
    )
    .pip_install("tongflow==0.2.21", "fastapi[standard]", "triton>=3.3")
    .env({"PYTHONPATH": COMFY, "HF_HOME": "/models/hf"})
)

with image.imports():
    import json
    import subprocess
    import time
    import urllib.error
    import urllib.request


def _tail_log(n: int = 3500) -> str:
    """Last n chars of the ComfyUI server stdout (per-node execution trace),
    with download progress-bar spam stripped so the node trace stays visible."""
    try:
        with open(COMFY_LOG, "rb") as f:
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return "(no server log)"
    lines = [
        ln.strip()
        for ln in text.replace("\r", "\n").split("\n")
        if ln.strip() and "MB/s" not in ln and "B/s]" not in ln
    ]
    return "\n".join(lines)[-n:]


def _maybe_bytes(val: object) -> Optional[bytes]:
    if val is None:
        return None
    try:
        return prompt_media_to_bytes(val)
    except (TypeError, ValueError):
        return None


def _frames_from_duration(duration: object) -> int:
    """Seconds -> frame count on the 17k+5 grid, clamped to the trained range."""
    try:
        seconds = float(duration) if duration is not None else 0.0
    except (TypeError, ValueError):
        seconds = 0.0
    if seconds <= 0:
        return DEFAULT_FRAMES
    n = max(MIN_FRAMES, round(seconds * FPS))
    n = min(n, MAX_FRAMES)
    while n % 17 != 5:
        n += 1
    return min(n, MAX_FRAMES)


def _canvas(width: object, height: object) -> tuple[int, int]:
    """SHORT_EDGE-short-side canvas with the model's area cap, per-axis round
    to 32 (mirrors ComfyUI's adapt_canvas). Only the aspect ratio of the
    requested width/height is used."""
    try:
        w = int(width) if width else 0
        h = int(height) if height else 0
    except (TypeError, ValueError):
        w = h = 0
    if w <= 0 or h <= 0:
        w, h = 16, 9
    ratio = w / h
    if ratio >= 1.0:
        nom_w, nom_h = SHORT_EDGE * ratio, float(SHORT_EDGE)
    else:
        nom_w, nom_h = float(SHORT_EDGE), SHORT_EDGE / ratio
    max_pixels = SHORT_EDGE * round(SHORT_EDGE * 1344 / 768)
    if nom_w * nom_h > max_pixels:
        s = (max_pixels / (nom_w * nom_h)) ** 0.5
        nom_w, nom_h = nom_w * s, nom_h * s
    return (
        max(DIM_ALIGN, round(nom_w / DIM_ALIGN) * DIM_ALIGN),
        max(DIM_ALIGN, round(nom_h / DIM_ALIGN) * DIM_ALIGN),
    )


def _seed(value: object) -> int:
    try:
        return int(value) if value is not None else 42
    except (TypeError, ValueError):
        return 42


_AUDIO_EXT = {"audio/wav": "wav", "audio/x-wav": "wav", "audio/mpeg": "mp3", "audio/flac": "flac"}


def _sampling_stack(wf: dict, cond_node: str, latent_node_slot: tuple, seed: int,
                    turbo_lora: Optional[str] = None, turbo_steps: int = 0) -> None:
    """Shared tail of both graphs: loaders are added by the callers; this wires
    the official template's custom sampling stack + AV decode + mp4 mux.
    Includes Sol-Attn sparse attention optimization (kijai/Triton) for ~3.9× speedup.

    With ``turbo_lora`` set: the LightX2V distill LoRA is applied right after
    the UNet loader and sampling drops to ``turbo_steps`` plain Euler on the
    ``simple`` schedule — the uniform-grid contract the LoRAs were distilled
    on. Model chain mirrors the official LightX2V workflows:
    UNETLoader -> [LoraLoaderModelOnly] -> MiniMaxH3SigmaShift -> SolAttn."""
    model_src = "1"
    if turbo_lora:
        wf["49"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": turbo_lora,
                "strength_model": TURBO_STRENGTH,
            },
        }
        model_src = "49"
    # Explicit AV shifts (12/3 = native defaults; kept in the graph so variants
    # like the fl2v 768p 4-step model, shift 6/3, are one env change away).
    wf["51"] = {
        "class_type": "MiniMaxH3SigmaShift",
        "inputs": {"model": [model_src, 0],
                   "shift_video": SHIFT_VIDEO, "shift_audio": SHIFT_AUDIO},
    }
    model_src = "51"
    wf["50"] = {
        "class_type": "SolAttnPatch",
        "inputs": {
            "model": [model_src, 0],
            "tau": 1.5,
            "start_percent": 0.1,
            "end_percent": 0.95,
            "min_tokens": 4096,
            "int8_qk": True,
            # Required since the 2026-08-08 V3-schema rework of the node:
            # INT8 P@V alongside INT8 QK ("the other half of the int8 win").
            "int8_pv": True,
            "sink_conditioning": "exact_kv_and_rows",
            "morton": True,
            "morton_curve": "2d_frame",
            "verbose": False,
            "use_tma": False,
            "dense_blocks": "0-2,-1",
        },
    }
    wf["6"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    wf["7"] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": "euler" if turbo_lora else "res_multistep"},
    }
    wf["8"] = {
        "class_type": "BasicScheduler",
        "inputs": {"model": ["50", 0], "scheduler": "simple",
                   "steps": turbo_steps if turbo_lora else STEPS, "denoise": 1.0},
    }
    wf["9"] = {
        "class_type": "BasicGuider",
        "inputs": {"model": ["50", 0], "conditioning": [cond_node, 0]},
    }
    wf["12"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["6", 0],
            "guider": ["9", 0],
            "sampler": ["7", 0],
            "sigmas": ["8", 0],
            "latent_image": list(latent_node_slot),
        },
    }
    wf["13"] = {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}}
    wf["14"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}}
    wf["15"] = {
        "class_type": "CreateVideo",
        "inputs": {"images": ["13", 0], "fps": float(FPS), "audio": ["14", 0]},
    }
    wf["16"] = {
        "class_type": "SaveVideo",
        "inputs": {"video": ["15", 0], "filename_prefix": "video/TongFlow_H3",
                   "format": "auto", "codec": "auto"},
    }


def _base_loaders(wf: dict, unet: str) -> None:
    wf["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}}
    wf["2"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax"}}
    wf["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}}
    wf["4"] = {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}}


def _fl2va_graph(prompt: str, width: int, height: int, frames: int, seed: int,
                 first_frame: Optional[str], last_frame: Optional[str]) -> dict:
    """t2va / first-frame / first+last-frame via MiniMaxH3ImageToVideo."""
    wf: dict[str, Any] = {}
    _base_loaders(wf, FL2VA_UNET)
    cond_inputs: dict[str, Any] = {
        "clip": ["2", 0],
        "vae": ["3", 0],
        "prompt": prompt,
        "width": width,
        "height": height,
        "length": frames,
    }
    if first_frame:
        wf["10"] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
        cond_inputs["first_frame"] = ["10", 0]
    if last_frame:
        wf["11"] = {"class_type": "LoadImage", "inputs": {"image": last_frame}}
        cond_inputs["last_frame"] = ["11", 0]
    wf["5"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": cond_inputs}
    _sampling_stack(wf, "5", ("5", 1), seed,
                    turbo_lora=TURBO_LORA if TURBO else None,
                    turbo_steps=TURBO_STEPS)
    return wf


def _ref2va_graph(prompt: str, width: int, height: int, frames: int, seed: int,
                  image_files: list[str], video_files: list[tuple[str, bool]],
                  audio_files: list[str]) -> dict:
    """Omni-reference via MiniMaxH3ReferenceToVideo. ``video_files`` entries are
    (filename, has_audio_track); a video's own soundtrack is wired as its
    paired ref_video_audio so H3 sees the clip with its sound."""
    wf: dict[str, Any] = {}
    _base_loaders(wf, REF2VA_UNET)
    cond_inputs: dict[str, Any] = {
        "clip": ["2", 0],
        "vae": ["3", 0],
        "audio_vae": ["4", 0],
        "prompt": prompt,
        "width": width,
        "height": height,
        "length": frames,
        "ref_image_size": "match",
    }
    for i, fn in enumerate(image_files):
        nid = str(100 + i)
        wf[nid] = {"class_type": "LoadImage", "inputs": {"image": fn}}
        cond_inputs[f"ref_images.ref_image_{i}"] = [nid, 0]
    for i, (fn, has_audio) in enumerate(video_files):
        load_id, comp_id = str(110 + i), str(115 + i)
        wf[load_id] = {"class_type": "LoadVideo", "inputs": {"file": fn}}
        wf[comp_id] = {"class_type": "GetVideoComponents", "inputs": {"video": [load_id, 0]}}
        cond_inputs[f"ref_videos.ref_video_{i}"] = [comp_id, 0]
        if has_audio:
            cond_inputs[f"ref_video_audios.ref_video_audio_{i}"] = [comp_id, 1]
    for i, fn in enumerate(audio_files):
        nid = str(120 + i)
        wf[nid] = {"class_type": "LoadAudio", "inputs": {"audio": fn}}
        cond_inputs[f"ref_audios.ref_audio_{i}"] = [nid, 0]
    wf["5"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": cond_inputs}
    _sampling_stack(wf, "5", ("5", 1), seed,
                    turbo_lora=TURBO_REF_LORA if TURBO_REF else None,
                    turbo_steps=TURBO_REF_STEPS)
    return wf


def _submit_graph(base, wf):
    """Submit an API workflow, poll, return (True, mp4_bytes) or (False, error).

    Prints [h3-timing] phase lines to the container stdout (visible in Modal
    logs, unlike the ComfyUI subprocess log): queue->done wall clock, plus the
    sampling-only span parsed from history execution messages when present."""
    t0 = time.monotonic()
    body = json.dumps({"prompt": wf}).encode()
    req = urllib.request.Request(
        f"{base}/prompt", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        pid = json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]
    except urllib.error.HTTPError as e:
        return False, f"workflow rejected: {e.read().decode()[:1500]}"
    out = None
    final_status = {}
    for _ in range(3600):
        time.sleep(1)
        with urllib.request.urlopen(f"{base}/history/{pid}", timeout=10) as r:
            hist = json.loads(r.read())
        if pid not in hist:
            continue
        h = hist[pid]
        status = h.get("status", {})
        final_status = status
        if status.get("status_str") == "error":
            print(f"[h3-timing] graph FAILED after {time.monotonic() - t0:.0f}s",
                  flush=True)
            return False, (
                "comfy error: " + json.dumps(status.get("messages", status))[:1500]
                + "\n[server log]\n" + _tail_log()
            )
        if h.get("outputs") and status.get("completed"):
            out = h["outputs"]
            break
    if not out:
        print(f"[h3-timing] graph TIMED OUT after {time.monotonic() - t0:.0f}s",
              flush=True)
        return False, "timed out\n[server log]\n" + _tail_log()
    # execution_start/_success carry ms timestamps -> pure execution span
    # (excludes our polling latency; on first run it includes model load).
    exec_s = None
    try:
        stamps = {m[0]: m[1].get("timestamp") for m in final_status.get("messages", [])
                  if isinstance(m, (list, tuple)) and len(m) > 1 and isinstance(m[1], dict)}
        if stamps.get("execution_start") and stamps.get("execution_success"):
            exec_s = (stamps["execution_success"] - stamps["execution_start"]) / 1000.0
    except Exception:
        pass
    print(f"[h3-timing] graph done: wall={time.monotonic() - t0:.0f}s"
          + (f" exec={exec_s:.0f}s" if exec_s else ""), flush=True)
    for node_out in out.values():
        for key in ("gifs", "videos", "images"):
            for item in node_out.get(key, []):
                fn, sub = item.get("filename"), item.get("subfolder", "")
                typ = item.get("type", "output")
                d = {"output": "output", "temp": "temp"}.get(typ, "output")
                path = os.path.join(COMFY, d, sub, fn or "")
                if fn and fn.endswith((".mp4", ".webm")) and os.path.isfile(path):
                    with open(path, "rb") as fh:
                        raw = fh.read()
                    if raw:
                        return True, raw
    summary = {}
    for nid, node_out in out.items():
        keys = {k: node_out.get(k) for k in ("gifs", "videos", "images") if node_out.get(k)}
        if keys:
            summary[nid] = keys
    return False, (
        "no video output; outputs=" + json.dumps(summary)[:600]
        + "; messages=" + json.dumps(final_status.get("messages", final_status))[:600]
        + "\n[server log]\n" + _tail_log()
    )


@deploy
@app.cls(
    image=image,
    gpu=GPU,
    volumes={"/models": volume},
    timeout=3600,
    scaledown_window=2,
)
class Inference:
    @modal.enter()
    def _boot(self) -> None:
        """Boot the ComfyUI server once; reused across calls (models stay warm)."""
        t0 = time.monotonic()
        os.makedirs(COMFY_MODELS, exist_ok=True)
        with open(os.path.join(COMFY, "extra_model_paths.yaml"), "w") as f:
            f.write(
                "h3_volume:\n"
                f"  base_path: {COMFY_MODELS}/\n"
                "  diffusion_models: diffusion_models\n"
                "  vae: vae\n"
                "  text_encoders: text_encoders\n"
                "  loras: loras\n"
            )
        for flag, name, lora in (("H3_TURBO", TURBO, TURBO_LORA),
                                 ("H3_TURBO_REF", TURBO_REF, TURBO_REF_LORA)):
            if name and not os.path.isfile(
                os.path.join(COMFY_MODELS, "loras", lora)
            ):
                raise RuntimeError(
                    f"{flag}=1 but loras/{lora} is missing from the models "
                    "volume — run `modal run download.py::download` first"
                )
        self._logfh = open(COMFY_LOG, "wb")
        self.proc = subprocess.Popen(
            [
                "python",
                "main.py",
                "--listen",
                "127.0.0.1",
                "--port",
                "8188",
                "--disable-auto-launch",
            ],
            cwd=COMFY,
            stdout=self._logfh,
            stderr=subprocess.STDOUT,
        )
        self.base = "http://127.0.0.1:8188"
        info = None
        for _ in range(600):
            if self.proc.poll() is not None:
                raise RuntimeError(f"ComfyUI exited early: {self.proc.returncode}")
            try:
                with urllib.request.urlopen(f"{self.base}/object_info", timeout=2) as r:
                    if r.status == 200:
                        info = json.loads(r.read())
                        break
            except Exception:
                time.sleep(1)
        if info is None:
            raise RuntimeError("ComfyUI server did not become ready")
        # Fail fast if this ComfyUI build predates the H3 nodes.
        for cls in ("MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo"):
            if cls not in info:
                raise RuntimeError(
                    f"{cls} missing from ComfyUI {COMFY_COMMIT[:12]} — bump COMFY_COMMIT"
                )
        print(
            f"[h3-timing] comfy {COMFY_COMMIT[:12]} ready in "
            f"{time.monotonic() - t0:.0f}s "
            f"(gpu={GPU} te={TE_VARIANT} steps={STEPS} "
            f"shift={SHIFT_VIDEO:g}/{SHIFT_AUDIO:g} turbo="
            + (f"on/{TURBO_STEPS}step" if TURBO else "off")
            + " turbo_ref="
            + (f"on/{TURBO_REF_STEPS}step" if TURBO_REF else "off")
            + ") — weights load lazily on the first graph",
            flush=True,
        )

    @modal.exit()
    def _shutdown(self) -> None:
        try:
            self.proc.terminate()
        except Exception:
            pass

    # ── input staging ──────────────────────────────────────────────────

    def _write_input(self, name: str, data: bytes) -> str:
        os.makedirs(f"{COMFY}/input", exist_ok=True)
        with open(f"{COMFY}/input/{name}", "wb") as f:
            f.write(data)
        return name

    def _has_audio_stream(self, name: str) -> bool:
        try:
            res = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a",
                 "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                 f"{COMFY}/input/{name}"],
                capture_output=True, text=True, timeout=30,
            )
            return "audio" in (res.stdout or "")
        except Exception:
            return False

    def _media_duration(self, name: str) -> float:
        try:
            res = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", f"{COMFY}/input/{name}"],
                capture_output=True, text=True, timeout=30,
            )
            return float((res.stdout or "0").strip())
        except Exception:
            return 0.0

    def _stage_audio(self, idx: int, aud) -> str:
        ext = _AUDIO_EXT.get((getattr(aud, "mime", None) or "").lower(), "mp3")
        return self._write_input(f"ref_aud_{idx}.{ext}", prompt_media_to_bytes(aud))

    # ── shared slot bodies ─────────────────────────────────────────────

    def _fl2va(self, text, width, height, duration, seed,
               first: Optional[bytes], last: Optional[bytes]):
        w, h = _canvas(width, height)
        frames = _frames_from_duration(duration)
        print(f"[h3-timing] request fl2va {w}x{h} {frames}f "
              f"steps={TURBO_STEPS if TURBO else STEPS} turbo={TURBO}", flush=True)
        first_fn = self._write_input("first.png", first) if first else None
        last_fn = self._write_input("last.png", last) if last else None
        wf = _fl2va_graph((text or "").strip(), w, h, frames, _seed(seed),
                          first_fn, last_fn)
        return _submit_graph(self.base, wf)

    def _ref2va(self, text, width, height, frames, seed,
                images: list[bytes], videos: list[bytes], audios: list) -> tuple:
        w, h = _canvas(width, height)
        print(f"[h3-timing] request ref2va {w}x{h} {frames}f "
              f"steps={TURBO_REF_STEPS if TURBO_REF else STEPS} "
              f"turbo_ref={TURBO_REF} "
              f"refs={len(images)}i/{len(videos)}v/{len(audios)}a", flush=True)
        image_files = [
            self._write_input(f"ref_img_{i}.png", b) for i, b in enumerate(images)
        ]
        video_files = []
        for i, b in enumerate(videos):
            fn = self._write_input(f"ref_vid_{i}.mp4", b)
            video_files.append((fn, self._has_audio_stream(fn)))
        audio_files = [self._stage_audio(i, a) for i, a in enumerate(audios)]
        wf = _ref2va_graph((text or "").strip(), w, h, frames, _seed(seed),
                           image_files, video_files, audio_files)
        return _submit_graph(self.base, wf)

    # ── FL2VA slots ────────────────────────────────────────────────────

    @modal.method()
    @node_slot(NodeSlots.TEXT_GEN_VIDEO)
    def text_gen_video(self, input: TextGenVideoInput) -> TextGenVideoOutput:
        if not (input.text or "").strip():
            return TextGenVideoOutput(success=False, error="Missing text prompt")
        ok, res = self._fl2va(input.text, input.width, input.height,
                              input.duration, input.seed, None, None)
        if ok:
            return TextGenVideoOutput(success=True, video=asset(res, mime="video/mp4"))
        return TextGenVideoOutput(success=False, error=str(res))

    @modal.method()
    @node_slot(NodeSlots.IMAGE_GEN_VIDEO)
    def image_gen_video(self, input: ImageGenVideoInput) -> ImageGenVideoOutput:
        if not (input.text or "").strip():
            return ImageGenVideoOutput(success=False, error="Missing text prompt")
        img = _maybe_bytes(input.image)
        if not img:
            return ImageGenVideoOutput(success=False, error="Missing image")
        ok, res = self._fl2va(input.text, input.width, input.height,
                              input.duration, input.seed, img, None)
        if ok:
            return ImageGenVideoOutput(success=True, video=asset(res, mime="video/mp4"))
        return ImageGenVideoOutput(success=False, error=str(res))

    @modal.method()
    @node_slot(NodeSlots.IMAGE_IMAGE_GEN_VIDEO)
    def image_image_gen_video(
        self, input: ImageImageGenVideoInput
    ) -> ImageImageGenVideoOutput:
        if not (input.text or "").strip():
            return ImageImageGenVideoOutput(success=False, error="Missing text prompt")
        first = _maybe_bytes(input.image)
        last = _maybe_bytes(input.end_image)
        if not first or not last:
            return ImageImageGenVideoOutput(
                success=False, error="Missing first or last frame image"
            )
        ok, res = self._fl2va(input.text, input.width, input.height,
                              input.duration, input.seed, first, last)
        if ok:
            return ImageImageGenVideoOutput(
                success=True, video=asset(res, mime="video/mp4")
            )
        return ImageImageGenVideoOutput(success=False, error=str(res))

    # ── Ref2VA slots ───────────────────────────────────────────────────

    @modal.method()
    @node_slot(NodeSlots.IMAGES_GEN_VIDEO)
    def images_gen_video(self, input: ImagesGenVideoInput) -> ImagesGenVideoOutput:
        if not (input.text or "").strip():
            return ImagesGenVideoOutput(success=False, error="Missing text prompt")
        images = [b for b in (_maybe_bytes(a) for a in input.images or []) if b]
        if not images:
            return ImagesGenVideoOutput(success=False, error="Missing reference images")
        if len(images) > MAX_REF_IMAGES:
            return ImagesGenVideoOutput(
                success=False, error=f"Too many reference images (max {MAX_REF_IMAGES})"
            )
        ok, res = self._ref2va(input.text, input.width, input.height,
                               _frames_from_duration(input.duration), input.seed,
                               images, [], [])
        if ok:
            return ImagesGenVideoOutput(success=True, video=asset(res, mime="video/mp4"))
        return ImagesGenVideoOutput(success=False, error=str(res))

    @modal.method()
    @node_slot(NodeSlots.AUDIO_IMAGE_GEN_VIDEO)
    def audio_image_gen_video(
        self, input: AudioImageGenVideoInput
    ) -> AudioImageGenVideoOutput:
        img = _maybe_bytes(input.image)
        if not img:
            return AudioImageGenVideoOutput(success=False, error="Missing image")
        if input.audio is None or not _maybe_bytes(input.audio):
            return AudioImageGenVideoOutput(success=False, error="Missing audio")
        # This slot has no duration field: follow the audio clip's length.
        aud_fn = self._stage_audio(0, input.audio)
        frames = _frames_from_duration(self._media_duration(aud_fn))
        text = (input.text or "").strip() or (
            "The character in <Picture 1> speaks with the voice from <Audio 1>."
        )
        w, h = _canvas(input.width, input.height)
        img_fn = self._write_input("ref_img_0.png", img)
        print(f"[h3-timing] request ref2va(audio-image) {w}x{h} {frames}f "
              f"steps={STEPS}", flush=True)
        wf = _ref2va_graph(text, w, h, frames, 42, [img_fn], [], [aud_fn])
        ok, res = _submit_graph(self.base, wf)
        if ok:
            return AudioImageGenVideoOutput(
                success=True, video=asset(res, mime="video/mp4")
            )
        return AudioImageGenVideoOutput(success=False, error=str(res))

    @modal.method()
    @node_slot(NodeSlots.REFS_GEN_VIDEO)
    def refs_gen_video(self, input: RefsGenVideoInput) -> RefsGenVideoOutput:
        if not (input.text or "").strip():
            return RefsGenVideoOutput(success=False, error="Missing text prompt")
        images = [b for b in (_maybe_bytes(a) for a in input.images or []) if b]
        videos = [b for b in (_maybe_bytes(a) for a in input.videos or []) if b]
        audios = [a for a in (input.audios or []) if _maybe_bytes(a)]
        if not images and not videos:
            return RefsGenVideoOutput(
                success=False,
                error="At least one reference image or video is required; "
                      "audio cannot be the only reference",
            )
        if (len(images) > MAX_REF_IMAGES or len(videos) > MAX_REF_VIDEOS
                or len(audios) > MAX_REF_AUDIOS
                or len(images) + len(videos) + len(audios) > MAX_REF_TOTAL):
            return RefsGenVideoOutput(
                success=False,
                error=f"Too many references: up to {MAX_REF_IMAGES} images, "
                      f"{MAX_REF_VIDEOS} videos and {MAX_REF_AUDIOS} audio clips "
                      f"({MAX_REF_TOTAL} files total)",
            )
        ok, res = self._ref2va(input.text, input.width, input.height,
                               _frames_from_duration(input.duration), input.seed,
                               images, videos, audios)
        if ok:
            return RefsGenVideoOutput(success=True, video=asset(res, mime="video/mp4"))
        return RefsGenVideoOutput(success=False, error=str(res))

    @modal.fastapi_endpoint(method="GET", label=f"{Path(__file__).resolve().parent.name}-serve")
    def serve(self, taskId: str = "", token: str = "", origin: str = ""):
        from fastapi.responses import StreamingResponse
        from tongflow import serve_stream_from_spec

        return StreamingResponse(
            serve_stream_from_spec(
                origin, taskId, token, __file__,
                invoke=lambda m, inp: getattr(self, m).local(inp),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
        )
