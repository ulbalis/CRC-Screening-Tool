import io
from pathlib import Path

import numpy as np
from PIL import Image
import openslide


# ──────────────────────────────────────────────
# Core viewing functions
# ──────────────────────────────────────────────

def open_slide(slide_path: str) -> openslide.OpenSlide:
    """
    Open a whole slide image.

    Args:
        slide_path: path to .svs, .ndpi, .tiff, etc.

    Returns:
        openslide.OpenSlide object
    """
    return openslide.OpenSlide(str(slide_path))


def get_slide_summary(slide: openslide.OpenSlide) -> dict:
    """
    Get a human-readable summary of the slide.

    Returns dict with: dimensions, mpp, objective, level_count,
                       level_dims, downsamples
    """
    props = slide.properties
    mpp_x = props.get(openslide.PROPERTY_NAME_MPP_X)
    mpp_y = props.get(openslide.PROPERTY_NAME_MPP_Y)
    mpp = None
    if mpp_x and mpp_y:
        mpp = (float(mpp_x) + float(mpp_y)) / 2.0

    objective = props.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)

    return {
        "dimensions": slide.dimensions,
        "mpp": mpp,
        "objective": float(objective) if objective else None,
        "level_count": slide.level_count,
        "level_dims": slide.level_dimensions,
        "downsamples": slide.level_downsamples,
    }


def get_thumbnail(slide: openslide.OpenSlide, max_size: int = 800) -> Image.Image:
    """
    Get a thumbnail of the entire slide.

    Args:
        slide:    open OpenSlide object
        max_size: max pixels for the longest edge

    Returns:
        PIL Image (RGB)
    """
    full_w, full_h = slide.dimensions
    if full_w >= full_h:
        thumb_w = max_size
        thumb_h = int(round(full_h * max_size / full_w))
    else:
        thumb_h = max_size
        thumb_w = int(round(full_w * max_size / full_h))

    return slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")


def get_view_region(
    slide: openslide.OpenSlide,
    center_x: int,
    center_y: int,
    downsample: float,
    view_width: int = 800,
    view_height: int = 600,
) -> Image.Image:
    """
    Read a region from the slide centered on (center_x, center_y)
    at the given downsample level.

    This is the core function that powers the interactive viewer.
    It reads from the best available pyramid level for efficiency.

    Args:
        slide:       open OpenSlide object
        center_x:    center x in level-0 (full resolution) coordinates
        center_y:    center y in level-0 (full resolution) coordinates
        downsample:  zoom level (1.0 = full resolution, 4.0 = 4x zoomed out)
        view_width:  viewport width in pixels to display
        view_height: viewport height in pixels to display

    Returns:
        PIL Image (RGB) of size (view_width, view_height)
    """
    slide_w, slide_h = slide.dimensions

    # How many level-0 pixels are visible in the viewport
    region_w = int(round(view_width * downsample))
    region_h = int(round(view_height * downsample))

    # Top-left corner in level-0 coordinates (clamped to slide bounds)
    x0 = max(0, int(center_x - region_w // 2))
    y0 = max(0, int(center_y - region_h // 2))

    # Don't go past the right/bottom edge
    if x0 + region_w > slide_w:
        x0 = max(0, slide_w - region_w)
    if y0 + region_h > slide_h:
        y0 = max(0, slide_h - region_h)

    # Find best pyramid level
    level = slide.get_best_level_for_downsample(downsample)
    level_downsample = slide.level_downsamples[level]

    # How many pixels to read at this level
    read_w = int(round(region_w / level_downsample))
    read_h = int(round(region_h / level_downsample))
    read_w = max(read_w, 1)
    read_h = max(read_h, 1)

    # Read the region
    region = slide.read_region((x0, y0), level, (read_w, read_h))
    region = region.convert("RGB")

    # Resize to exact viewport size
    if region.size != (view_width, view_height):
        region = region.resize((view_width, view_height), Image.LANCZOS)

    return region


def pil_to_png_bytes(img: Image.Image) -> bytes:
    """
    Convert a PIL Image to PNG bytes for display in an ipywidgets Image widget.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()