from __future__ import annotations

from pathlib import Path
from typing import Dict

from PIL import Image, ImageDraw

from .storyboard import PageSpec


def compose_page(
    page: PageSpec,
    panel_images: Dict[str, Image.Image],
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    page_img = Image.new("RGB", (page.page_width, page.page_height), (255, 255, 255))
    draw = ImageDraw.Draw(page_img)

    for panel in page.panels:
        x, y, w, h = panel.bbox_norm
        px = int(x * page.page_width)
        py = int(y * page.page_height)
        pw = int(w * page.page_width)
        ph = int(h * page.page_height)

        img = panel_images[panel.panel_id]
        img = img.resize((max(8, pw), max(8, ph)), resample=Image.LANCZOS)
        page_img.paste(img, (px, py))

        bx0 = px
        by0 = py
        bx1 = px + pw
        by1 = py + ph
        for i in range(int(page.border_px)):
            draw.rectangle((bx0 + i, by0 + i, bx1 - i, by1 - i), outline=(0, 0, 0))

    page_img.save(out_path)
    return out_path
