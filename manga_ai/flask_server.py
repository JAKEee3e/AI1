from __future__ import annotations

import atexit
import os
import threading
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from flask import Flask, jsonify, request, send_file

from manga_ai.scripts.cloudflare_tunnel import TunnelProcess, start_tunnel

if TYPE_CHECKING:
    from manga_ai.scripts.pipeline import Backends


class MangaService:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._lock = threading.Lock()
        self._backends: Optional["Backends"] = None

    def init_models(self) -> None:
        try:
            from manga_ai.scripts.pipeline import default_model_paths, init_backends
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Missing Python dependency. Install requirements (pip install -r manga_ai/requirements.txt). "
                "If the error mentions segment_anything, ensure Segment Anything is installed."
            ) from e

        with self._lock:
            if self._backends is not None:
                return
            paths = default_model_paths(self.root)
            self._backends = init_backends(paths)

    def generate(self, prompt: str, pages: int, page_index: int = 0) -> Path:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt is empty")

        try:
            from manga_ai.scripts.pipeline import run_one_page
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Missing Python dependency. Install requirements (pip install -r manga_ai/requirements.txt). "
                "If the error mentions segment_anything, ensure Segment Anything is installed."
            ) from e

        self.init_models()

        with self._lock:
            return run_one_page(
                root=self.root,
                story_prompt=prompt,
                pages=int(pages),
                page_index=int(page_index),
                backends=self._backends,
            )


def start_public_tunnel(app: Flask, *, local_port: int, host: str) -> TunnelProcess:
    tunnel = app.extensions["manga_tunnel"]
    if tunnel["proc"] is not None:
        return tunnel["proc"]

    root: Path = app.extensions["manga_root"]
    cache_dir = root / "outputs" / "cloudflared"
    cache_dir.mkdir(parents=True, exist_ok=True)

    tp = start_tunnel(local_port=local_port, cache_dir=cache_dir, host=host)
    tunnel["proc"] = tp
    return tp


def create_app(root: str | Path) -> Flask:
    app = Flask(__name__)
    service = MangaService(root)

    tunnel: dict[str, Optional[TunnelProcess]] = {"proc": None}

    app.extensions["manga_root"] = Path(root)
    app.extensions["manga_service"] = service
    app.extensions["manga_tunnel"] = tunnel

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/init")
    def init():
        try:
            service.init_models()
            return jsonify({"status": "ready"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/generate")
    def generate():
        payload = request.get_json(silent=True) or {}
        prompt = payload.get("prompt", "")
        pages = payload.get("pages", 1)
        page_index = payload.get("page_index", 0)

        try:
            out_path = service.generate(prompt=prompt, pages=int(pages), page_index=int(page_index))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        if request.args.get("download") == "1":
            return send_file(out_path, mimetype="image/png")

        return jsonify({"out_path": str(out_path)})

    @app.post("/tunnel/start")
    def tunnel_start():
        if tunnel["proc"] is not None:
            return jsonify({"public_url": tunnel["proc"].public_url})

        payload = request.get_json(silent=True) or {}
        local_port = int(payload.get("local_port", int(os.environ.get("MANGA_AI_PORT", "7860"))))
        host = str(payload.get("host", os.environ.get("MANGA_AI_HOST", "127.0.0.1")))

        tp = start_public_tunnel(app, local_port=local_port, host=host)
        return jsonify({"public_url": tp.public_url})

    @app.get("/tunnel")
    def tunnel_info():
        if tunnel["proc"] is None:
            return jsonify({"public_url": None})
        return jsonify({"public_url": tunnel["proc"].public_url})

    def _cleanup_tunnel() -> None:
        tp = tunnel.get("proc")
        if tp is None:
            return
        try:
            tp.process.terminate()
        except Exception:
            pass

    atexit.register(_cleanup_tunnel)
    return app


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    default_root = Path(__file__).resolve().parent
    root = Path(os.environ.get("MANGA_AI_ROOT", str(default_root)))

    host = os.environ.get("MANGA_AI_HOST", "127.0.0.1")
    port = int(os.environ.get("MANGA_AI_PORT", "7860"))

    preload = _env_bool("MANGA_AI_PRELOAD_MODELS", default=False)
    auto_tunnel = _env_bool("MANGA_AI_START_TUNNEL", default=False)

    app = create_app(root)

    service: MangaService = app.extensions["manga_service"]

    if preload:
        threading.Thread(target=service.init_models, daemon=True).start()

    if auto_tunnel:
        tp = start_public_tunnel(app, local_port=port, host=host)
        print("Public URL:", tp.public_url)

    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
