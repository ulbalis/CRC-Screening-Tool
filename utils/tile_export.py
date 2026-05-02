from pathlib import Path
import numpy as np
from PIL import Image
import openslide
import random



# ──────────────────────────────────────────────
# Slide metadata helpers
# ──────────────────────────────────────────────

def get_slide_info(slide: openslide.OpenSlide) -> dict:
    """
    Extract key metadata from a whole slide image.

    Returns a dict with:
        - dimensions:   (width, height) at full resolution
        - mpp:          microns per pixel (averaged x/y), or None
        - objective:    objective power (e.g. 40.0), or None
        - level_count:  number of resolution levels
        - level_dims:   list of (w, h) for each level
        - downsamples:  list of downsample factors for each level
    """
    props = slide.properties

    mpp_x = props.get(openslide.PROPERTY_NAME_MPP_X)
    mpp_y = props.get(openslide.PROPERTY_NAME_MPP_Y)
    if mpp_x is not None and mpp_y is not None:
        mpp = (float(mpp_x) + float(mpp_y)) / 2.0
    else:
        mpp = None

    objective = props.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    if objective is not None:
        objective = float(objective)

    return {
        "dimensions": slide.dimensions,
        "mpp": mpp,
        "objective": objective,
        "level_count": slide.level_count,
        "level_dims": slide.level_dimensions,
        "downsamples": slide.level_downsamples,
    }


def calculate_downsample(
    slide: openslide.OpenSlide,
    target_pixel_size: float = 1.0,
    fallback_downsample: float = 4.0,
) -> float:
    """
    Calculate the downsample factor to achieve a target pixel size.

    A target of 1.0 um/pixel corresponds to approximately 10x magnification.

    Args:
        slide:              an open OpenSlide object
        target_pixel_size:  desired um per pixel (default 1.0 for ~10x)
        fallback_downsample: used when pixel calibration is unavailable

    Returns:
        downsample factor (float)
    """
    info = get_slide_info(slide)
    mpp = info["mpp"]

    if mpp is None or mpp <= 0:
        print(f"  Pixel calibration not available, using fallback downsample = {fallback_downsample}")
        return fallback_downsample

    downsample = target_pixel_size / mpp
    return downsample


def get_best_level(slide: openslide.OpenSlide, downsample: float) -> int:
    """
    Find the slide pyramid level whose native downsample is closest
    to (but not greater than) the requested downsample.
    """
    return slide.get_best_level_for_downsample(downsample)


# ──────────────────────────────────────────────
# Tile grid generation
# ──────────────────────────────────────────────

def generate_tile_grid(
    slide_width: int,
    slide_height: int,
    downsample: float,
    tile_size: int = 512,
    overlap: int = 0,
    include_partial: bool = True,
) -> list[dict]:
    """
    Generate a list of tile coordinates that cover the entire slide.

    Args:
        slide_width:     slide width in full-resolution pixels
        slide_height:    slide height in full-resolution pixels
        downsample:      downsample factor
        tile_size:       tile width/height at the target resolution
        overlap:         tile overlap in pixels at the target resolution
        include_partial: whether to include edge tiles smaller than tile_size

    Returns:
        list of dicts, each with:
            - x, y:             top-left in full-resolution coordinates
            - read_w, read_h:   region size in full-resolution coordinates
            - tile_w, tile_h:   expected output size at target resolution
            - is_partial:       True if this is an edge tile
    """
    target_w = int(round(slide_width / downsample))
    target_h = int(round(slide_height / downsample))

    step = tile_size - overlap
    tiles = []

    for ty in range(0, target_h, step):
        for tx in range(0, target_w, step):
            tw = min(tile_size, target_w - tx)
            th = min(tile_size, target_h - ty)

            is_partial = (tw < tile_size) or (th < tile_size)

            if not include_partial and is_partial:
                continue

            x_full = int(round(tx * downsample))
            y_full = int(round(ty * downsample))
            w_full = int(round(tw * downsample))
            h_full = int(round(th * downsample))

            tiles.append({
                "x": x_full,
                "y": y_full,
                "read_w": w_full,
                "read_h": h_full,
                "tile_w": tw,
                "tile_h": th,
                "is_partial": is_partial,
            })

    return tiles


# ──────────────────────────────────────────────
# Read a single tile from the slide
# ──────────────────────────────────────────────

def read_tile(
    slide: openslide.OpenSlide,
    x: int,
    y: int,
    read_w: int,
    read_h: int,
    tile_w: int,
    tile_h: int,
    level: int = 0,
    downsample: float = 1.0,
) -> np.ndarray:
    """
    Read a tile from the slide and resize to the target dimensions.

    Args:
        slide:      an open OpenSlide object
        x, y:       top-left corner in full-resolution (level 0) coordinates
        read_w:     width to read in full-resolution pixels
        read_h:     height to read in full-resolution pixels
        tile_w:     desired output width
        tile_h:     desired output height
        level:      pyramid level to read from
        downsample: the downsample factor

    Returns:
        RGB numpy array of shape (tile_h, tile_w, 3)
    """
    level_downsample = slide.level_downsamples[level]
    level_w = max(int(round(read_w / level_downsample)), 1)
    level_h = max(int(round(read_h / level_downsample)), 1)

    region = slide.read_region((x, y), level, (level_w, level_h))
    region = region.convert("RGB")

    if region.size != (tile_w, tile_h):
        region = region.resize((tile_w, tile_h), Image.LANCZOS)

    return np.array(region)


# ──────────────────────────────────────────────
# Content filter (replicates tileHasInformation)
# ──────────────────────────────────────────────

def tile_has_information(
    tile: np.ndarray,
    min_informative_fraction: float = 0.05,
    min_std: float = 10.0,
    gradient_threshold: float = 30.0,
    sample_step: int = 8,
) -> bool:
    """
    Check whether a tile contains meaningful tissue content.

    Stage 1 - Variance check:
        Reject tiles where grayscale standard deviation is too low
        (blank background, empty glass).

    Stage 2 - Gradient texture check:
        Reject tiles where too few pixels have strong intensity gradients
        (blurry, out-of-focus, or featureless areas).

    Args:
        tile:                       RGB numpy array, shape (H, W, 3)
        min_informative_fraction:   minimum fraction of pixels with strong gradients
        min_std:                    minimum grayscale standard deviation
        gradient_threshold:         minimum gradient magnitude to count as informative
        sample_step:                spacing between sampled pixels

    Returns:
        True if the tile has enough texture to be worth keeping
    """
    gray = tile[::sample_step, ::sample_step].mean(axis=2).astype(np.float64)

    if gray.size < 10:
        return False

    if gray.std() < min_std:
        return False

    g_right = np.abs(gray[:, 1:] - gray[:, :-1])
    g_down = np.abs(gray[1:, :] - gray[:-1, :])

    min_rows = min(g_right.shape[0], g_down.shape[0])
    min_cols = min(g_right.shape[1], g_down.shape[1])
    g_max = np.maximum(g_right[:min_rows, :min_cols], g_down[:min_rows, :min_cols])

    if g_max.size == 0:
        return False

    informative_fraction = (g_max > gradient_threshold).sum() / g_max.size
    return informative_fraction >= min_informative_fraction


# ──────────────────────────────────────────────
# Main export function
# ──────────────────────────────────────────────

def export_tiles(
    svs_path: str,
    output_dir: str,
    tile_size: int = 512,
    target_pixel_size: float = 1.0,
    overlap: int = 0,
    include_partial: bool = True,
    jpeg_quality: int = 100,
    start_serial: int = 1,
    min_informative_fraction: float = 0.05,
    min_std: float = 10.0,
    gradient_threshold: float = 30.0,
    sample_step: int = 8,
    max_tiles: int = None,
    progress_callback=None,
) -> dict:
    """
    Export tiles from a whole slide image.

    Tile filenames include the level-0 coordinates so that each tile's
    position on the slide can be recovered later for visualization:

        {base_name}_x{x}_y{y}__{serial}.jpg

    Returns:
        dict with keys:
            - kept, discarded, total
            - base_name, downsample
            - tile_info: list of dicts for each kept tile
    """
    svs_path = Path(svs_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = svs_path.stem

    slide = openslide.OpenSlide(str(svs_path))
    info = get_slide_info(slide)
    print(f"  Slide: {svs_path.name}")
    print(f"  Dimensions: {info['dimensions']}")
    print(f"  Microns/pixel: {info['mpp']}")
    print(f"  Objective: {info['objective']}")

    downsample = calculate_downsample(slide, target_pixel_size)
    print(f"  Downsample factor: {downsample:.4f}")

    level = get_best_level(slide, downsample)
    print(f"  Reading from level: {level} (native downsample: {slide.level_downsamples[level]:.2f})")

    slide_w, slide_h = slide.dimensions
    tiles = generate_tile_grid(
        slide_w, slide_h, downsample,
        tile_size=tile_size,
        overlap=overlap,
        include_partial=include_partial,
    )
    total = len(tiles)
    random.shuffle(tiles)
    print(f"  Total candidate tiles: {total}")

    serial = start_serial
    kept = 0
    discarded = 0
    kept_tiles = []

    for i, t in enumerate(tiles):
        tile_rgb = read_tile(
            slide,
            t["x"], t["y"],
            t["read_w"], t["read_h"],
            t["tile_w"], t["tile_h"],
            level=level,
            downsample=downsample,
        )

        if tile_has_information(
            tile_rgb,
            min_informative_fraction=min_informative_fraction,
            min_std=min_std,
            gradient_threshold=gradient_threshold,
            sample_step=sample_step,
        ):
            serial_str = f"{serial:05d}"
            filename = f"{base_name}_x{t['x']}_y{t['y']}__{serial_str}.jpg"
            filepath = output_dir / filename

            img = Image.fromarray(tile_rgb)
            img.save(str(filepath), "JPEG", quality=jpeg_quality)

            kept_tiles.append({
                "x": t["x"],
                "y": t["y"],
                "read_w": t["read_w"],
                "read_h": t["read_h"],
                "tile_w": t["tile_w"],
                "tile_h": t["tile_h"],
                "filename": filename,
                "serial": serial,
            })

            serial += 1
            kept += 1
        else:
            discarded += 1

        if progress_callback:
            progress_callback(kept, discarded, total)

        if max_tiles is not None and kept >= max_tiles:
            break

    slide.close()

    print(f"  Kept {kept} tiles, discarded {discarded} low-information tiles.")
    print(f"  Done.")

    return {
        "kept": kept,
        "discarded": discarded,
        "total": total,
        "base_name": base_name,
        "downsample": downsample,
        "tile_info": kept_tiles,
    }


# ──────────────────────────────────────────────
# Visualization: show tile locations on thumbnail
# ──────────────────────────────────────────────

def get_slide_thumbnail(slide_path: str, max_size: int = 2000) -> tuple[np.ndarray, float]:
    """
    Get a thumbnail of the whole slide image.

    Reads from the lowest-resolution pyramid level and resizes
    so the longest edge is at most max_size pixels.

    Args:
        slide_path: path to the whole slide image
        max_size:   max pixels for the longest edge

    Returns:
        (thumbnail_rgb, scale_factor)
        - thumbnail_rgb:  numpy array (H, W, 3)
        - scale_factor:   multiply level-0 coords by this to get thumbnail coords
    """
    slide = openslide.OpenSlide(str(slide_path))
    full_w, full_h = slide.dimensions

    if full_w >= full_h:
        thumb_w = max_size
        thumb_h = int(round(full_h * max_size / full_w))
    else:
        thumb_h = max_size
        thumb_w = int(round(full_w * max_size / full_h))

    thumbnail = slide.get_thumbnail((thumb_w, thumb_h))
    thumbnail = thumbnail.convert("RGB")
    slide.close()

    scale_factor = thumb_w / full_w
    return np.array(thumbnail), scale_factor


def visualize_tile_locations(
    slide_path: str,
    tile_info: list[dict],
    max_size: int = 2000,
    box_color: tuple = (0, 0, 0),
    box_width: int = 2,
    figsize: tuple = (14, 10),
    title: str = "Tile Locations on Whole Slide Image",
):
    """
    Draw tile bounding boxes on top of a whole slide thumbnail.

    Args:
        slide_path:  path to the whole slide image
        tile_info:   list of tile dicts from export_tiles()["tile_info"]
                     each dict must have: x, y, read_w, read_h
        max_size:    max thumbnail dimension in pixels
        box_color:   RGB tuple for box outlines (default black)
        box_width:   line width for box outlines
        figsize:     matplotlib figure size
        title:       plot title

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    thumbnail, scale = get_slide_thumbnail(slide_path, max_size)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(thumbnail)

    for t in tile_info:
        rect_x = t["x"] * scale
        rect_y = t["y"] * scale
        rect_w = t["read_w"] * scale
        rect_h = t["read_h"] * scale

        rect = patches.Rectangle(
            (rect_x, rect_y),
            rect_w, rect_h,
            linewidth=box_width,
            edgecolor=[c / 255 for c in box_color],
            facecolor="none",
        )
        ax.add_patch(rect)

    ax.set_title(f"{title}  ({len(tile_info)} tiles)", fontsize=14)
    ax.axis("off")
    plt.tight_layout()

    return fig
