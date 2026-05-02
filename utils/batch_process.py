"""
batch_process.py

Run tile export, scoring, and heatmap generation for every
slide in sample_data/wsi/.

Expected folder structure (created by the bash command):
    wsi/
    ├── 8130/
    │   ├── 8130.svs
    │   └── tiles/
    ├── 2085139779/
    │   ├── 2085139779.svs
    │   └── tiles/
    ...

Usage (in notebook cell or terminal):
    %run batch_process.py
"""

import os
import time
from pathlib import Path

# Suppress TensorFlow noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils.tile_export import export_tiles
from utils.tile_score import score_tiles, create_heatmap_overlay

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
WSI_DIR = Path("sample_data/wsi")
MODEL_PATH = Path("model/model.h5")
PREFIX = "batch_"

# ──────────────────────────────────────────────
# Find all slide folders
# ──────────────────────────────────────────────
slide_folders = sorted([
    d for d in WSI_DIR.iterdir()
    if d.is_dir() and any(d.glob("*.svs"))
])

print(f"Found {len(slide_folders)} slides to process.")
print("=" * 60)

# ──────────────────────────────────────────────
# Process each slide
# ──────────────────────────────────────────────
for i, folder in enumerate(slide_folders):
    # Find the SVS file inside the folder
    svs_files = list(folder.glob("*.svs"))
    if not svs_files:
        print(f"\n⚠️  No SVS file found in {folder.name}, skipping.")
        continue

    svs_path = svs_files[0]
    tiles_dir = folder / "tiles"
    tiles_dir.mkdir(exist_ok=True)

    slide_name = folder.name
    print(f"\n{'=' * 60}")
    print(f"[{i+1}/{len(slide_folders)}] {slide_name}")
    print(f"{'=' * 60}")

    start_time = time.time()

    # ── Step 1: Export tiles ──
    print(f"\n  Step 1: Exporting tiles...")
    try:
        result = export_tiles(
            svs_path=str(svs_path),
            output_dir=str(tiles_dir),
        )
        n_kept = result["kept"]
        n_discarded = result["discarded"]
        print(f"  → Kept {n_kept} tiles, discarded {n_discarded}")
    except Exception as e:
        print(f"  ❌ Tile export failed: {e}")
        continue

    if n_kept == 0:
        print(f"  ⚠️  No informative tiles found, skipping scoring.")
        continue

    # ── Step 2: Score tiles ──
    print(f"\n  Step 2: Scoring tiles...")
    csv_path = folder / f"{PREFIX}tile_scores.csv"
    try:
        df = score_tiles(
            image_dir=str(tiles_dir),
            model_path=str(MODEL_PATH),
            output_csv=str(csv_path),
        )
        print(f"  → Scored {len(df)} tiles, saved to {csv_path.name}")
    except Exception as e:
        print(f"  ❌ Scoring failed: {e}")
        continue

    # ── Step 3: Generate heatmap overlay ──
    print(f"\n  Step 3: Generating heatmap overlay...")
    try:
        fig = create_heatmap_overlay(
            slide_path=str(svs_path),
            scores_df=df,
            threshold=0.5,
            alpha=0.4,
            title=f"Heatmap: {slide_name}",
        )

        if fig is not None:
            heatmap_path = folder / f"{PREFIX}heatmap.png"
            fig.savefig(str(heatmap_path), dpi=200, bbox_inches="tight")

            # Close figure to free memory
            import matplotlib.pyplot as plt
            plt.close(fig)

            print(f"  → Saved heatmap to {heatmap_path.name}")
        else:
            print(f"  ⚠️  Heatmap generation returned None (no coordinate data in filenames?)")
    except Exception as e:
        print(f"  ❌ Heatmap generation failed: {e}")

    elapsed = time.time() - start_time
    print(f"\n  Completed in {elapsed:.1f} seconds.")

    # ── Free memory ──
    del df

# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"All done! Processed {len(slide_folders)} slides.")
print(f"{'=' * 60}")
print(f"\nOutput files in each slide folder:")
print(f"  tiles/                  — exported tile images")
print(f"  {PREFIX}tile_scores.csv — model predictions for each tile")
print(f"  {PREFIX}heatmap.png     — color-coded malignancy overlay")