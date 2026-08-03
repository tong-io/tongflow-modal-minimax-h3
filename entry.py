"""Local bridge for a @deploy (Modal-backed) plugin.

The platform spawns ``python entry.py`` with cwd set to this plugin directory,
writes ``{"nodeSlot","prompt","taskId"}`` to stdin, and reads one ABI-JSON
object from stdout. This file owns everything backend-specific: discovering which
class/method serves the requested slot, ensuring the app is deployed, invoking
the deployed method remotely, streaming progress, and cancelling on SIGTERM.

This file is IDENTICAL across all @deploy (Modal) plugins:
- slot -> (class, method) is AST-parsed from this plugin's own deploy.py via the
  SDK's backend-neutral tongflow.parse_deploy (it matches @deploy, not @app.cls);
  no hand-maintained map, no reflection through Modal wrappers.
- ``modal`` is imported lazily only at remote-invoke time, supplied by this
  plugin's requirements.txt — the SDK itself never depends on modal.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from tongflow.parse_deploy import _slot_to_ident, parse_deploy_py  # AST only, no modal
from tongflow.progress import progress

_HERE = Path(__file__).resolve().parent
DEPLOY_PATH = _HERE / "deploy.py"
DOWNLOAD_PATH = _HERE / "download.py"
APP_NAME = _HERE.name  # matches modal.App(Path(__file__).parent.name) in deploy.py

# Hard cap on waiting for remote work; align with the heaviest plugin's own
# Modal function timeout. Overridable via env.
_CALL_TIMEOUT_S = float(os.environ.get("TONGFLOW_MODAL_CALL_TIMEOUT_S", 40 * 60))


def _discover(node_slot: str) -> tuple[str, str]:
    """(cls_name, method_name) for node_slot — AST-parsed from deploy.py.

    No import of deploy.py, no modal: the SDK parser reads the @deploy class and
    its @node_slot methods statically. deploy.py is the single source of truth.
    """
    scan, err = parse_deploy_py(DEPLOY_PATH)
    if err or scan is None:
        raise RuntimeError(err or f"failed to parse {DEPLOY_PATH.name}")
    # parse_deploy keys methods_by_slot / cls_by_slot by NodeSlots ident
    # (e.g. "TRANSCRIBE"), so map the wire slot string to its ident first.
    ident = _slot_to_ident(node_slot)
    method = scan.methods_by_slot.get(ident)
    if not method:
        raise RuntimeError(f"deploy.py does not implement nodeSlot={node_slot!r}")
    cls_name = scan.cls_by_slot.get(ident, scan.cls_name)
    return cls_name, method


def _file_hash(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _cache_dir() -> Path:
    base = os.environ.get("TONGFLOW_MODAL_CACHE_DIR")
    d = Path(base) if base else (Path.home() / ".tongflow" / "modal-cache")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path() -> Path:
    return _cache_dir() / f"{APP_NAME}.json"


def _load_cache() -> dict[str, Any]:
    try:
        return json.loads(_cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_cache(data: dict[str, Any]) -> None:
    try:
        _cache_path().write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass  # cache is best-effort; a write failure just re-runs next time


def _run_modal_cli(args: list[str], label: str) -> None:
    progress(f"Modal: {label}")
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "modal", *args],
        cwd=str(_HERE),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
        raise RuntimeError(f"{label} failed (exit {proc.returncode}): {tail}")


def _ensure_weights() -> None:
    """Run `modal run download::download` once per download.py content."""
    if not DOWNLOAD_PATH.is_file():
        return
    h = _file_hash(DOWNLOAD_PATH)
    cache = _load_cache()
    if cache.get("downloadHash") == h:
        return
    _run_modal_cli(["run", f"{DOWNLOAD_PATH.name}::download"], "downloading weights")
    cache["downloadHash"] = h
    _save_cache(cache)


def _ensure_deployed() -> None:
    """Deploy once per deploy.py content (picks up deploy.py edits proactively)."""
    h = _file_hash(DEPLOY_PATH)
    cache = _load_cache()
    if cache.get("deployHash") == h:
        return
    _deploy()


def _deploy() -> None:
    _run_modal_cli(["deploy", DEPLOY_PATH.name], "deploying")
    cache = _load_cache()
    cache["deployHash"] = _file_hash(DEPLOY_PATH)
    _save_cache(cache)


def _invoke(cls_name: str, method_name: str, prompt: dict[str, Any]) -> Any:
    import modal  # lazy: supplied by this plugin's requirements.txt

    progress(f"Modal: invoking {method_name}()")
    cls = modal.Cls.from_name(APP_NAME, cls_name)
    instance = cls()
    fn = getattr(instance, method_name)

    # Spawn so we hold a handle we can cancel on SIGTERM (platform sends it on
    # task cancellation), then block on the result with a hard timeout.
    call = fn.spawn(prompt)

    def _on_term(_signo: int, _frame: Any) -> None:
        try:
            call.cancel()
        finally:
            raise SystemExit(130)

    signal.signal(signal.SIGTERM, _on_term)
    return call.get(timeout=_CALL_TIMEOUT_S)


def _looks_like_not_deployed(err: Exception) -> bool:
    msg = str(err).lower()
    return "not found" in msg or "deploy" in msg or "no such" in msg


def run(node_slot: str, prompt: dict[str, Any]) -> Any:
    cls_name, method_name = _discover(node_slot)
    _ensure_weights()
    _ensure_deployed()
    try:
        return _invoke(cls_name, method_name, prompt)
    except Exception as e:
        # The deploy cache claimed the app was live but the remote disagrees
        # (fresh Modal account, or a method added since last deploy). Force a
        # redeploy once and retry.
        if _looks_like_not_deployed(e):
            _deploy()
            return _invoke(cls_name, method_name, prompt)
        raise


def main() -> int:
    try:
        raw = sys.stdin.read()
        req = json.loads(raw) if raw.strip() else {}
        node_slot = str(req.get("nodeSlot") or "")
        prompt = req.get("prompt") if isinstance(req.get("prompt"), dict) else {}
        if not node_slot:
            raise RuntimeError("missing nodeSlot")
        out = run(node_slot, prompt)
    except SystemExit:
        raise
    except Exception as e:  # surfaced to the UI as an ABI failure
        sys.stdout.write(json.dumps({"success": False, "error": str(e)}))
        sys.stdout.flush()
        return 1

    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
