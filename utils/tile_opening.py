import re
import numpy as np
import pandas as pd
import openslide
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from scipy import ndimage


def build_tile_grid(df, score_column="p_Positive", threshold=0.5):
    """Convert tile coordinates and scores into a 2D binary grid.
    Each tile becomes one cell. Foreground (1) = malignant, background (0) = benign."""

    coords = df['file'].apply(lambda f: re.search(r'_x(\d+)_y(\d+)__', Path(f).name))
    df = df.copy()
    df['x'] = coords.apply(lambda m: int(m.group(1)) if m else None)
    df['y'] = coords.apply(lambda m: int(m.group(2)) if m else None)
    df = df.dropna(subset=['x', 'y'])
    df['x'] = df['x'].astype(int)
    df['y'] = df['y'].astype(int)

    xs = sorted(df['x'].unique())
    ys = sorted(df['y'].unique())
    tile_ext = int(np.median([xs[i+1] - xs[i] for i in range(len(xs)-1)])) if len(xs) > 1 else 512

    x_to_col = {x: i for i, x in enumerate(xs)}
    y_to_row = {y: i for i, y in enumerate(ys)}

    n_rows = len(ys)
    n_cols = len(xs)

    binary_grid = np.zeros((n_rows, n_cols), dtype=np.int32)
    score_grid = np.full((n_rows, n_cols), np.nan)
    coord_map = {}

    for _, row in df.iterrows():
        c = x_to_col[int(row['x'])]
        r = y_to_row[int(row['y'])]
        score_grid[r, c] = row[score_column]
        binary_grid[r, c] = 1 if row[score_column] >= threshold else 0
        coord_map[(r, c)] = (int(row['x']), int(row['y']))

    return binary_grid, score_grid, coord_map, tile_ext, xs, ys


def morphological_opening(binary_grid, min_neighbors=4, iterations=2):
    """Apply morphological opening to a binary tile grid.

    Erosion (applied twice): a positive tile is removed if it has fewer than
    min_neighbors positive neighbors (8-connected).

    Dilation (applied twice): after erosion, tiles that were positive in the
    ORIGINAL grid and are adjacent to a surviving positive tile are restored.
    Dilation never creates new positives that weren't in the original grid.
    """

    rows, cols = binary_grid.shape
    original = binary_grid.copy()

    def erode_once(grid):
        result = np.zeros_like(grid)
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 0:
                    continue
                count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 1:
                            count += 1
                if count >= min_neighbors:
                    result[r, c] = 1
        return result

    def dilate_once(grid):
        result = grid.copy()
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 0:
                    continue
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if original[nr, nc] == 1:
                                result[nr, nc] = 1
        return result

    # Erode twice
    current = binary_grid.copy()
    for _ in range(iterations):
        current = erode_once(current)
    eroded = current.copy()

    # Dilate twice
    for _ in range(iterations):
        current = dilate_once(current)
    opened = current

    return eroded, opened


def create_overlay(ax, thumbnail, df, scale, tile_ext, mask_grid, coord_map,
                   xs, ys, score_column="p_Positive", threshold=0.5, alpha=0.4):
    """Draw heatmap overlay on a slide thumbnail using a filtered binary mask."""

    ax.imshow(thumbnail)
    tw = tile_ext * scale

    x_to_col = {x: i for i, x in enumerate(xs)}
    y_to_row = {y: i for i, y in enumerate(ys)}

    n_mal, n_ben, n_removed = 0, 0, 0

    coords = df['file'].apply(lambda f: re.search(r'_x(\d+)_y(\d+)__', Path(f).name))
    df = df.copy()
    df['x'] = coords.apply(lambda m: int(m.group(1)) if m else None)
    df['y'] = coords.apply(lambda m: int(m.group(2)) if m else None)
    df = df.dropna(subset=['x', 'y'])

    for _, row in df.iterrows():
        x_px, y_px = int(row['x']), int(row['y'])
        s = row[score_column]
        tx, ty = x_px * scale, y_px * scale

        c = x_to_col.get(x_px)
        r = y_to_row.get(y_px)
        if r is None or c is None:
            continue

        is_positive_in_mask = mask_grid[r, c] == 1

        if s >= threshold and is_positive_in_mask:
            intensity = (s - threshold) / (1.0 - threshold + 1e-9)
            color = (1.0, 0.0, 0.0, alpha * (0.3 + 0.7 * intensity))
            n_mal += 1
        elif s >= threshold and not is_positive_in_mask:
            color = (0.0, 0.6, 0.0, alpha * 0.5)
            n_removed += 1
        else:
            intensity = (threshold - s) / (threshold + 1e-9)
            color = (0.0, 0.6, 0.0, alpha * (0.3 + 0.7 * intensity))
            n_ben += 1

        ax.add_patch(patches.Rectangle((tx, ty), tw, tw, facecolor=color, linewidth=0))

    ax.set_xlim(0, thumbnail.shape[1])
    ax.set_ylim(thumbnail.shape[0], 0)
    ax.axis("off")

    return n_mal, n_ben, n_removed


if __name__ == "__main__":

    SLIDE_PATH = "../sample_data/wsi/positive/8130/8130.svs"
    SCORES_CSV = "../sample_data/wsi/positive/8130/batch_tile_scores.csv"
    OUTPUT_PNG = "../sample_data/wsi/opening_before_after.png"
    THRESHOLD  = 0.5
    ALPHA      = 0.4
    MIN_NEIGHBORS = 4

    # Load slide thumbnail
    slide = openslide.OpenSlide(SLIDE_PATH)
    w, h = slide.dimensions
    scale = 2000 / max(w, h)
    thumb = np.array(slide.get_thumbnail((int(w * scale), int(h * scale))).convert("RGB"))
    slide.close()

    # Load scores
    df = pd.read_csv(SCORES_CSV)

    # Build tile grid
    binary_grid, score_grid, coord_map, tile_ext, xs, ys = build_tile_grid(
        df, score_column="p_Positive", threshold=THRESHOLD
    )

    n_pos_before = binary_grid.sum()
    print(f"Tile grid: {binary_grid.shape[0]} rows x {binary_grid.shape[1]} cols")
    print(f"Positive tiles before opening: {n_pos_before}")

    # Apply morphological opening
    eroded, opened = morphological_opening(binary_grid, min_neighbors=MIN_NEIGHBORS)

    n_pos_eroded = eroded.sum()
    n_pos_opened = opened.sum()
    n_removed = n_pos_before - n_pos_opened
    print(f"Positive tiles after erosion:   {n_pos_eroded}")
    print(f"Positive tiles after opening:   {n_pos_opened}")
    print(f"Tiles removed by opening:       {n_removed}")

    # Before/after comparison
    fig, axes = plt.subplots(1, 2, figsize=(28, 12))

    # Before (original binary grid as mask)
    n_mal_b, n_ben_b, _ = create_overlay(
        axes[0], thumb, df, scale, tile_ext, binary_grid, coord_map, xs, ys,
        threshold=THRESHOLD, alpha=ALPHA
    )
    axes[0].set_title(
        f"Malignancy Heatmap Overlay (Before Opening)\n{n_mal_b} malignant (red), {n_ben_b} benign (green)",
        fontsize=14
    )

    # After (opened grid as mask)
    n_mal_a, n_ben_a, n_rem = create_overlay(
        axes[1], thumb, df, scale, tile_ext, opened, coord_map, xs, ys,
        threshold=THRESHOLD, alpha=ALPHA
    )
    axes[1].set_title(
        f"After Opening (min {MIN_NEIGHBORS} neighbors, 2× erosion + 2× dilation)\n"
        f"{n_mal_a} malignant (red), {n_ben_a + n_rem} benign (green) — {n_rem} reclassified by opening",
        fontsize=14
    )

    plt.suptitle(
        f"Morphological Opening — Tile-Level Spatial Filtering\n"
        f"2× erosion (tiles with < {MIN_NEIGHBORS} neighbors removed) "
        f"followed by 2× dilation (restores edges of surviving clusters)",
        fontsize=15
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_PNG}")