"""A/B attention backends and distill families on the deployed app.

Every run bills Modal on the deployed GPU (B200 by default). Runs go to a
private container pool (max 1 container, 30 s scaledown) so they share one
warm boot without touching the production pool's scaledown setting. The first
run pays the cold boot + weight load: keep a throwaway baseline first.

Each run is "<attn>/<distill>", attn in solattn | core | vsa | dense, distill
in pdd | turbo | base. vsa needs the gate on the volume
(`H3_VSA=1 modal run download.py::download`). Same seed throughout, so the
clips are directly comparable.

  python bench.py --image ref.png --prompt "<Picture 1> ..."          # plan
  python bench.py --image ref.png --prompt "<Picture 1> ..." --yes    # run
  python bench.py --slot t2v --prompt "A corgi ..." --yes             # FL2VA

Writes <out>/<n>-<attn>-<distill>.mp4 plus results.json with client wall
time and the decoded audio levels (mean -inf dB = silent track).
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import shutil
import subprocess
import time
from pathlib import Path

APP = Path(__file__).resolve().parent.name
DEFAULT_RUNS = "solattn/pdd,solattn/pdd,core/pdd,vsa/pdd,vsa/turbo,dense/pdd"
SLOTS = {"refs": "refs_gen_video", "t2v": "text_gen_video"}


def _audio_levels(path: Path) -> dict:
    if not shutil.which("ffmpeg"):
        return {}
    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af", "volumedetect",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    levels = {}
    for key in ("mean_volume", "max_volume"):
        m = re.search(rf"{key}: (-?[\d.]+|-inf) dB", res.stderr)
        levels[key] = m.group(1) if m else None
    return levels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--slot", choices=sorted(SLOTS), default="refs")
    ap.add_argument("--image", type=Path, help="reference image (refs slot)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs", default=DEFAULT_RUNS)
    ap.add_argument("--vsa-sparsity", type=float)
    ap.add_argument("--out", type=Path, default=Path("bench-out"))
    ap.add_argument("--yes", action="store_true", help="actually run (bills Modal)")
    args = ap.parse_args()

    runs = [tuple(r.strip().split("/")) for r in args.runs.split(",") if r.strip()]
    for r in runs:
        if len(r) != 2:
            ap.error(f"run {'/'.join(r)!r} is not <attn>/<distill>")
    payload: dict = {"text": args.prompt, "duration": args.duration, "seed": args.seed}
    if args.slot == "refs":
        if not args.image:
            ap.error("--image is required for the refs slot")
        mime = mimetypes.guess_type(args.image.name)[0] or "image/png"
        payload["images"] = [{
            "bytesBase64": base64.b64encode(args.image.read_bytes()).decode(),
            "mime": mime, "filename": args.image.name,
        }]

    print(f"{APP}.{SLOTS[args.slot]}  {args.duration:g}s  seed={args.seed}")
    for i, (attn, distill) in enumerate(runs):
        print(f"  {i}: attn={attn} distill={distill}"
              + ("  (cold boot, timing not comparable)" if i == 0 else ""))
    if not args.yes:
        print("Dry run. Re-run with --yes to execute; every run bills Modal.")
        return

    import modal

    cls = modal.Cls.from_name(APP, "Inference").with_options(
        max_containers=1, scaledown_window=30
    )
    method = getattr(cls(), SLOTS[args.slot])
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for i, (attn, distill) in enumerate(runs):
        params: dict = {"attn": attn, "distill": distill}
        if args.vsa_sparsity is not None:
            params["vsa_sparsity"] = args.vsa_sparsity
        t0 = time.monotonic()
        try:
            out = method.remote({**payload, "_params": params})
        except Exception as e:  # keep the remaining runs going
            out = {"success": False, "error": f"{type(e).__name__}: {e}"}
        wall = time.monotonic() - t0
        row = {"run": i, "attn": attn, "distill": distill, "wall_s": round(wall, 1),
               "cold": i == 0, "success": bool(out.get("success"))}
        if row["success"]:
            path = args.out / f"{i}-{attn}-{distill}.mp4"
            path.write_bytes(base64.b64decode(out["video"]["bytesBase64"]))
            row.update(file=str(path), **_audio_levels(path))
        else:
            row["error"] = str(out.get("error"))[:800]
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        (args.out / "results.json").write_text(json.dumps(results, indent=2))

    print("\nrun  attn     distill  wall_s  audio(mean/max dB)")
    for r in results:
        audio = f"{r.get('mean_volume')}/{r.get('max_volume')}" if r["success"] else "FAILED"
        print(f"{r['run']:>3}  {r['attn']:<8} {r['distill']:<8} {r['wall_s']:>6}"
              f"{'*' if r['cold'] else ' '} {audio}")
    print("* includes cold boot + weight load")


if __name__ == "__main__":
    main()
