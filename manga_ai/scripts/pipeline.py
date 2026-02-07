from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

from PIL import Image

from .model_downloader import ensure_models_downloaded

if TYPE_CHECKING:
    from . import panel_generator, storyboard


@dataclass
class ModelPaths:
    qwen_dir: Path
    sdxl_dir: Path


@dataclass
class Backends:
    qwen: "storyboard.QwenBackend"
    sdxl_panel: "panel_generator.SDXLPanelBackend"


def init_backends(paths: ModelPaths) -> Backends:
    ensure_models_downloaded(
        qwen_dir=paths.qwen_dir,
        sdxl_dir=paths.sdxl_dir,
    )

    from . import panel_generator, storyboard

    qwen = storyboard.load_qwen_backend(paths.qwen_dir)
    sdxl_panel = panel_generator.load_sdxl_panel_pipeline(paths.sdxl_dir)
    return Backends(qwen=qwen, sdxl_panel=sdxl_panel)


def default_model_paths(root: str | Path) -> ModelPaths:
    root = Path(root)
    return ModelPaths(
        qwen_dir=root / "models" / "qwen2.5",
        sdxl_dir=root / "models" / "sdxl",
    )


def run_one_page(
    root: str | Path,
    story_prompt: str,
    page_index: int = 0,
    pages: int = 1,
    backends: Optional[Backends] = None,
) -> Path:
    from . import page_composer, panel_generator, storyboard

    root = Path(root)
    paths = default_model_paths(root)

    if backends is None:
        backends = init_backends(paths)

    sb = storyboard.generate_storyboard_from_backend(
        story_prompt=story_prompt,
        backend=backends.qwen,
        pages=pages,
    )

    page = sb.pages[page_index]

    storyboard.save_storyboard(sb, root / "outputs" / "storyboards" / "storyboard.json")

    panel_dir = root / "outputs" / "panels" / f"page_{page.page_index:03d}"
    panel_paths = panel_generator.generate_page_panels(
        backend=backends.sdxl_panel,
        page=page,
        out_dir=panel_dir,
        global_negative_prompt=sb.global_negative_prompt,
    )

    final_panels: Dict[str, Image.Image] = {}

    for panel in page.panels:
        img = Image.open(panel_paths[panel.panel_id]).convert("RGB")

        final_panels[panel.panel_id] = img

    out_page = root / "outputs" / "pages" / f"page_{page.page_index:03d}.png"
    return page_composer.compose_page(
        page=page,
        panel_images=final_panels,
        out_path=out_page,
    )
