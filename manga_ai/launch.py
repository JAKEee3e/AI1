from __future__ import annotations

from pathlib import Path
import time

from manga_ai.scripts.cloudflare_tunnel import start_tunnel
from manga_ai.scripts.gradio_app import launch_gradio


def main():
    root = Path(__file__).resolve().parent
    port = 7860

    _, port = launch_gradio(root=root, host="127.0.0.1", port=port)

    tunnel = start_tunnel(local_port=port, cache_dir=root / "outputs" / "cloudflared")
    print("Public URL:", tunnel.public_url)

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        try:
            tunnel.process.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()
