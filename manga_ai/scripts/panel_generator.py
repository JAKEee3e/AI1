from __future__ import annotations

from dataclasses import dataclass
import os
import inspect
from pathlib import Path
from typing import Dict, Optional

import torch
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from PIL import Image

from .storyboard import PageSpec, PanelSpec


@dataclass
class SDXLPanelBackend:
    pipe: StableDiffusionXLPipeline
    img2img: StableDiffusionXLImg2ImgPipeline


def _from_pretrained_with_dtype(cls, model_dir: Path, dtype: torch.dtype, **kwargs):
    try:
        sig = inspect.signature(cls.from_pretrained)
        if "dtype" in sig.parameters:
            kwargs["dtype"] = dtype
        elif "torch_dtype" in sig.parameters:
            kwargs["torch_dtype"] = dtype
        else:
            kwargs["torch_dtype"] = dtype
    except Exception:
        kwargs["torch_dtype"] = dtype

    return cls.from_pretrained(model_dir, **kwargs)


def load_sdxl_panel_pipeline(
    model_dir: str | Path,
    dtype: torch.dtype = torch.float16,
) -> SDXLPanelBackend:
    model_dir = Path(model_dir)

    vae = None
    vae_path = model_dir / "vae"
    if vae_path.exists():
        vae = _from_pretrained_with_dtype(AutoencoderKL, vae_path, dtype)

    variant = "fp16" if dtype == torch.float16 else None
    variant_used = variant
    try:
        pipe = _from_pretrained_with_dtype(
            StableDiffusionXLPipeline,
            model_dir,
            dtype,
            variant=variant,
            use_safetensors=True,
            vae=vae,
        )
    except ValueError as e:
        msg = str(e)
        if variant == "fp16" and "variant=fp16" in msg and "no such modeling files" in msg:
            pipe = _from_pretrained_with_dtype(
                StableDiffusionXLPipeline,
                model_dir,
                dtype,
                variant=None,
                use_safetensors=True,
                vae=vae,
            )
            variant_used = None
        else:
            raise

    try:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    except TypeError:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    if torch.cuda.is_available():
        pipe.to("cuda")

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    try:
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        else:
            pipe.enable_vae_tiling()
    except Exception:
        pass

    try:
        img2img = _from_pretrained_with_dtype(
            StableDiffusionXLImg2ImgPipeline,
            model_dir,
            dtype,
            variant=variant_used,
            use_safetensors=True,
            vae=vae,
        )
    except ValueError as e:
        msg = str(e)
        if variant_used == "fp16" and "variant=fp16" in msg and "no such modeling files" in msg:
            img2img = _from_pretrained_with_dtype(
                StableDiffusionXLImg2ImgPipeline,
                model_dir,
                dtype,
                variant=None,
                use_safetensors=True,
                vae=vae,
            )
        else:
            raise
    img2img.scheduler = pipe.scheduler
    img2img.set_progress_bar_config(disable=True)
    if torch.cuda.is_available():
        img2img.to("cuda")
    try:
        img2img.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    try:
        if hasattr(img2img, "vae") and hasattr(img2img.vae, "enable_tiling"):
            img2img.vae.enable_tiling()
        else:
            img2img.enable_vae_tiling()
    except Exception:
        pass

    return SDXLPanelBackend(pipe=pipe, img2img=img2img)


def _panel_size_px(page: PageSpec, panel: PanelSpec) -> tuple[int, int]:
    _, _, w, h = panel.bbox_norm
    width = max(1024, int(page.page_width * w))
    height = max(1024, int(page.page_height * h))

    width = (width // 8) * 8
    height = (height // 8) * 8
    return max(64, width), max(64, height)


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in {"0", "false", "no", "off"}


def _round8(x: int) -> int:
    return max(64, (int(x) // 8) * 8)


@torch.inference_mode()
def generate_page_panels(
    backend: SDXLPanelBackend,
    page: PageSpec,
    out_dir: str | Path,
    global_negative_prompt: str,
    default_seed: int = 1234,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Path] = {}

    hires = _env_flag("MANGA_AI_HIRES", True)
    hires_strength = float(os.environ.get("MANGA_AI_HIRES_STRENGTH", "0.28"))
    hires_steps = int(os.environ.get("MANGA_AI_HIRES_STEPS", "28"))
    hires_upscale = float(os.environ.get("MANGA_AI_HIRES_UPSCALE", "2.0"))

    for panel in page.panels:
        w, h = _panel_size_px(page, panel)
        seed = panel.seed if panel.seed is not None else default_seed
        gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))

        prompt = panel.sdxl_prompt
        neg = " , ".join([panel.sdxl_negative_prompt, global_negative_prompt]).strip(" ,")

        if hires:
            base_w = _round8(min(w, int(w / hires_upscale)))
            base_h = _round8(min(h, int(h / hires_upscale)))
            base_w = max(768, base_w)
            base_h = max(768, base_h)

            image = backend.pipe(
                prompt=prompt,
                negative_prompt=neg,
                width=base_w,
                height=base_h,
                num_inference_steps=int(panel.steps),
                guidance_scale=float(panel.cfg_scale),
                generator=gen,
            ).images[0]

            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)

            image = image.resize((w, h), resample=Image.LANCZOS)

            image = backend.img2img(
                prompt=prompt,
                negative_prompt=neg,
                image=image,
                strength=hires_strength,
                num_inference_steps=hires_steps,
                guidance_scale=float(panel.cfg_scale),
                generator=gen,
            ).images[0]
        else:
            image = backend.pipe(
                prompt=prompt,
                negative_prompt=neg,
                width=w,
                height=h,
                num_inference_steps=int(panel.steps),
                guidance_scale=float(panel.cfg_scale),
                generator=gen,
            ).images[0]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        path = out_dir / f"{panel.panel_id}.png"
        image.save(path)
        results[panel.panel_id] = path

    return results
