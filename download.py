"""Modal download entry for MiniMax-H3 (ComfyUI native path).

Run:
  modal run download.py::download

Fetches the Comfy-Org optimized MiniMax-H3 weights onto the shared ``models``
volume, laid out as ComfyUI model dirs under /models/comfyui:

  diffusion_models/  H3 FL2VA + Ref2VA packed DiT (pruned int8 convrot, 21 GB each)
  text_encoders/     Qwen3-VL-32B H3 encoder (nvfp4 awq 15.7 GB or int8 27.1 GB)
  vae/               H3 video VAE (fp16, 5.2 GB) + audio VAE (fp32, 0.6 GB)

Comfy-Org/MiniMax-H3 is public; HF_TOKEN is only needed if it becomes gated.
Set H3_TEXT_ENCODER_VARIANT=nvfp4 (default int8) to fetch the NVFP4 text
encoder instead — Blackwell-only, pair with H3_GPU=B200.
"""

from __future__ import annotations

import os
from typing import Any

import modal

_cfg: dict[str, Any] = {}

COMFY_MODELS = "/models/comfyui"

_TE_FILES = {
    "nvfp4": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "int8": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
}
_TE_VARIANT = (os.environ.get("H3_TEXT_ENCODER_VARIANT") or "int8").strip().lower()
_TE_FILE = _TE_FILES.get(_TE_VARIANT, _TE_FILES["int8"])

# (repo_id, path-in-repo, comfyui-subdir, flat-dest-name)
# Filenames must match the graph builder in deploy.py exactly.
MODELS = [
    (
        "Comfy-Org/MiniMax-H3",
        "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "diffusion_models",
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    ),
    (
        "Comfy-Org/MiniMax-H3",
        "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "diffusion_models",
        "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    ),
    (
        "Comfy-Org/MiniMax-H3",
        f"text_encoders/{_TE_FILE}",
        "text_encoders",
        _TE_FILE,
    ),
    (
        "Comfy-Org/MiniMax-H3",
        "vae/minimax_h3_video_vae_fp16.safetensors",
        "vae",
        "minimax_h3_video_vae_fp16.safetensors",
    ),
    (
        "Comfy-Org/MiniMax-H3",
        "vae/minimax_h3_audio_vae_fp32.safetensors",
        "vae",
        "minimax_h3_audio_vae_fp32.safetensors",
    ),
]

volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(volume_name, create_if_missing=True)

model_downloader = modal.App("model_downloader")

_download_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface_hub>=0.34.0,<1.0"
)


@model_downloader.function(
    image=_download_image,
    volumes={"/models": volume},
    timeout=7200,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def _download(models: list[tuple[str, str, str, str]]) -> None:
    import shutil

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    token = os.environ.get("HF_TOKEN") or None

    for repo, path, subdir, name in models:
        dest_dir = os.path.join(COMFY_MODELS, subdir)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, name)
        if os.path.isfile(dest) and os.path.getsize(dest) > 1_000_000:
            print(f"skip (exists): {subdir}/{name}")
            continue
        print(f"Downloading {repo}/{path} ...")
        try:
            src = hf_hub_download(repo_id=repo, filename=path, token=token)
        except GatedRepoError as e:
            raise RuntimeError(
                f"{repo} is gated and this HF_TOKEN is not authorized. Open "
                f"https://huggingface.co/{repo}, accept the license, then re-run."
            ) from e
        shutil.copyfile(src, dest)
        print(f"  got {subdir}/{name} ({os.path.getsize(dest) // (1024 * 1024)} MB)")
        # Commit after each large file so a later failure doesn't re-download it.
        volume.commit()

    print("Done.")


@model_downloader.local_entrypoint()
def download() -> None:
    # The MODELS list is resolved locally (H3_TEXT_ENCODER_VARIANT from the
    # caller's env) and passed in, so the remote function needs no env mirroring.
    _download.remote(MODELS)
