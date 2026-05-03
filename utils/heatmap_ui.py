"""
Step 6: Heatmap Overlay — Interactive tile-level heatmap on WSI
================================================================
Renders a color-coded malignancy heatmap on a whole slide image
using pre-computed tile scores. Optionally applies morphological
opening to remove isolated false positive tiles.

Usage (in a notebook cell):
    from utils.heatmap_ui import create_heatmap_ui
    create_heatmap_ui()
"""

import re
import numpy as np
import pandas as pd
import openslide
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from ipyfilechooser import FileChooser
from pathlib import Path
from utils.tile_opening import build_tile_grid, morphological_opening


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def validate_csv_for_slide(csv_path, slide_path):
    """Check if a tile scores CSV corresponds to the given slide.
    Extracts slide name from the SVS filename and checks if tile
    filenames in the CSV contain that slide name."""

    slide_name = Path(slide_path).stem
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, f"Could not read CSV: {e}", None

    if 'file' not in df.columns:
        return False, "CSV is missing the 'file' column.", None

    if 'p_Positive' not in df.columns:
        return False, "CSV is missing the 'p_Positive' column.", None

    if len(df) == 0:
        return False, "CSV has no rows.", None

    sample_files = df['file'].head(10).apply(lambda f: Path(f).name).tolist()
    matches = sum(1 for f in sample_files if slide_name in f)

    if matches == 0:
        first_file = sample_files[0] if sample_files else "N/A"
        return False, (
            f"Slide name '{slide_name}' not found in tile filenames.\n"
            f"Example tile filename: {first_file}\n"
            f"Make sure the CSV was generated from this slide."
        ), None

    n_pos = (df['p_Positive'] >= 0.5).sum()
    n_neg = (df['p_Positive'] < 0.5).sum()

    return True, f"Matched. {len(df)} tiles ({n_pos} malignant, {n_neg} benign).", df


def render_heatmap(slide_path, df, apply_opening, min_neighbors, iterations,
                   threshold, alpha, output_widget):
    """Generate and display the heatmap overlay."""

    with output_widget:
        clear_output(wait=True)
        print("Loading slide thumbnail...")

        slide = openslide.OpenSlide(str(slide_path))
        w, h = slide.dimensions
        scale = 2000 / max(w, h)
        thumb = np.array(
            slide.get_thumbnail((int(w * scale), int(h * scale))).convert("RGB")
        )
        slide.close()

        print("Building tile grid...")
        binary_grid, score_grid, coord_map, tile_ext, xs, ys = build_tile_grid(
            df, score_column="p_Positive", threshold=threshold
        )

        n_pos_before = int(binary_grid.sum())

        if apply_opening:
            print(f"Applying morphological opening "
                  f"(min_neighbors={min_neighbors}, iterations={iterations})...")
            eroded, opened = morphological_opening(
                binary_grid, min_neighbors=min_neighbors, iterations=iterations
            )
            n_pos_after = int(opened.sum())
            n_removed = n_pos_before - n_pos_after
            print(f"  Before: {n_pos_before} positive tiles")
            print(f"  After:  {n_pos_after} positive tiles")
            print(f"  Removed: {n_removed} isolated tiles")

            # Side-by-side
            fig, axes = plt.subplots(1, 2, figsize=(24, 10))

            _draw_overlay(axes[0], thumb, df, scale, tile_ext, binary_grid,
                          xs, ys, threshold, alpha)
            n_mal_b = int(binary_grid.sum())
            n_ben_b = len(df) - n_mal_b
            axes[0].set_title(
                f"Before Opening\n{n_mal_b} malignant (red), {n_ben_b} benign (green)",
                fontsize=13
            )

            _draw_overlay(axes[1], thumb, df, scale, tile_ext, opened,
                          xs, ys, threshold, alpha)
            axes[1].set_title(
                f"After Opening\n{n_pos_after} malignant (red), "
                f"{len(df) - n_pos_after} benign (green) — "
                f"{n_removed} reclassified",
                fontsize=13
            )

            plt.suptitle("Malignancy Heatmap Overlay — Before vs After Opening",
                         fontsize=15)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            _draw_overlay(ax, thumb, df, scale, tile_ext, binary_grid,
                          xs, ys, threshold, alpha)
            n_mal = int(binary_grid.sum())
            n_ben = len(df) - n_mal
            ax.set_title(
                f"Malignancy Heatmap Overlay\n"
                f"{n_mal} malignant (red), {n_ben} benign (green) — "
                f"threshold={threshold}",
                fontsize=14
            )

        plt.tight_layout()
        plt.show()

def _draw_overlay(ax, thumbnail, df, scale, tile_ext, mask_grid,
                  xs, ys, threshold=0.5, alpha=0.4):
    """Draw colored rectangles on a slide thumbnail."""

    ax.imshow(thumbnail)
    tw = tile_ext * scale

    x_to_col = {x: i for i, x in enumerate(xs)}
    y_to_row = {y: i for i, y in enumerate(ys)}

    coords = df['file'].apply(lambda f: re.search(r'_x(\d+)_y(\d+)__', Path(f).name))
    df_tmp = df.copy()
    df_tmp['x'] = coords.apply(lambda m: int(m.group(1)) if m else None)
    df_tmp['y'] = coords.apply(lambda m: int(m.group(2)) if m else None)
    df_tmp = df_tmp.dropna(subset=['x', 'y'])

    for _, row in df_tmp.iterrows():
        x_px, y_px = int(row['x']), int(row['y'])
        s = row['p_Positive']
        tx, ty = x_px * scale, y_px * scale

        c = x_to_col.get(x_px)
        r = y_to_row.get(y_px)
        if r is None or c is None:
            continue

        is_positive_in_mask = mask_grid[r, c] == 1

        if s >= threshold and is_positive_in_mask:
            intensity = (s - threshold) / (1.0 - threshold + 1e-9)
            color = (1.0, 0.0, 0.0, alpha * (0.3 + 0.7 * intensity))
        elif s >= threshold and not is_positive_in_mask:
            color = (0.0, 0.6, 0.0, alpha * 0.5)
        else:
            intensity = (threshold - s) / (threshold + 1e-9)
            color = (0.0, 0.6, 0.0, alpha * (0.3 + 0.7 * intensity))

        ax.add_patch(patches.Rectangle((tx, ty), tw, tw, facecolor=color, linewidth=0))

    ax.set_xlim(0, thumbnail.shape[1])
    ax.set_ylim(thumbnail.shape[0], 0)
    ax.axis("off")


def create_heatmap_ui():
    """Create the Step 6: Heatmap Overlay interactive panel."""

    project_dir = Path.cwd()

    # ── UI Elements ──

    slide_chooser = FileChooser(str(project_dir / "sample_data" / "wsi"/ "positive" / "8130" ),
                                filter_pattern='*.svs',
                                title='Select WSI file (.svs):')
    slide_chooser.show_only_dirs = False

    csv_chooser = FileChooser(str(project_dir / "sample_data" / "wsi" / "positive" / "8130" ),
                              filter_pattern='*.csv',
                              title='Select tile scores CSV:')
    csv_chooser.show_only_dirs = False

    validation_output = widgets.Output()

    apply_opening_cb = widgets.Checkbox(
        value=False,
        description='Apply morphological opening',
        style={'description_width': 'auto'},
        layout=widgets.Layout(width='300px')
    )

    # Opening options (visible only when checkbox is checked)
    neighbors_slider = widgets.IntSlider(
        value=4, min=2, max=8, step=1,
        description='Min neighbors:',
        style={'description_width': '110px'},
        layout=widgets.Layout(width='300px')
    )

    iterations_slider = widgets.IntSlider(
        value=2, min=1, max=4, step=1,
        description='Iterations:',
        style={'description_width': '110px'},
        layout=widgets.Layout(width='300px')
    )

    opening_options = widgets.VBox([neighbors_slider, iterations_slider])
    opening_options.layout.display = 'none'

    def on_opening_toggle(change):
        opening_options.layout.display = '' if change['new'] else 'none'

    apply_opening_cb.observe(on_opening_toggle, names='value')

    threshold_slider = widgets.FloatSlider(
        value=0.5, min=0.05, max=0.95, step=0.05,
        description='Threshold:',
        style={'description_width': '110px'},
        layout=widgets.Layout(width='300px')
    )

    alpha_slider = widgets.FloatSlider(
        value=0.4, min=0.1, max=0.8, step=0.05,
        description='Overlay alpha:',
        style={'description_width': '110px'},
        layout=widgets.Layout(width='300px')
    )

    render_btn = widgets.Button(
        description='Render Heatmap',
        button_style='success',
        icon='play',
        layout=widgets.Layout(width='180px'),
        disabled=True
    )

    plot_output = widgets.Output()

    # ── State ──
    state = {'df': None}

    # ── Auto-validation ──
    def check_ready(*args):
        render_btn.disabled = True
        state['df'] = None
        validation_output.clear_output()

        slide_path = slide_chooser.selected
        csv_path = csv_chooser.selected

        if not slide_path or not Path(slide_path).exists():
            with validation_output:
                display(HTML(
                    '<span style="color: #888;">Select a slide and a tile scores CSV.</span>'
                ))
            return

        if not csv_path or not Path(csv_path).exists():
            with validation_output:
                display(HTML(
                    '<span style="color: #888;">Select a tile scores CSV to continue.</span>'
                ))
            return

        valid, msg, df = validate_csv_for_slide(csv_path, slide_path)

        if not valid:
            with validation_output:
                display(HTML(
                    f'<div style="color: #c62828; background: #ffebee; padding: 10px; '
                    f'border-radius: 4px; margin: 5px 0;">'
                    f'<b>⚠ CSV does not match this slide.</b><br>'
                    f'{msg}</div>'
                ))
            return

        state['df'] = df
        with validation_output:
            display(HTML(
                f'<div style="color: #2e7d32; background: #e8f5e9; padding: 10px; '
                f'border-radius: 4px; margin: 5px 0;">'
                f'<b>✓ {msg}</b></div>'
            ))
        render_btn.disabled = False

    slide_chooser.register_callback(check_ready)
    csv_chooser.register_callback(check_ready)

    # ── Render Handler ──
    def on_render(btn):
        render_btn.disabled = True
        with plot_output:
            clear_output()

        try:
            render_heatmap(
                slide_path=slide_chooser.selected,
                df=state['df'],
                apply_opening=apply_opening_cb.value,
                min_neighbors=neighbors_slider.value,
                iterations=iterations_slider.value,
                threshold=threshold_slider.value,
                alpha=alpha_slider.value,
                output_widget=plot_output
            )
        except Exception as e:
            with plot_output:
                print(f"Error: {e}")
            raise
        finally:
            render_btn.disabled = False

    render_btn.on_click(on_render)

    # ── Layout ──
    header = widgets.HTML(
        '<h3 style="margin-bottom: 5px;">Step 6: Heatmap Overlay</h3>'
        '<p style="color: #666; font-size: 13px; margin-top: 0;">'
        'Visualize tile-level malignancy predictions on a whole slide image. '
        'Optionally apply morphological opening to remove isolated false positive tiles.</p>'
    )

    config_box = widgets.VBox([
        widgets.HBox([
            widgets.VBox([
                widgets.HTML('<b>1. Select WSI slide:</b>'),
                slide_chooser,
                widgets.HTML('<br><b>2. Select tile scores CSV:</b>'),
                csv_chooser,
            ], layout=widgets.Layout(width='50%')),
            widgets.VBox([
                widgets.HTML('<b>3. Options:</b>'),
                threshold_slider,
                alpha_slider,
                apply_opening_cb,
                opening_options,
                render_btn,
            ], layout=widgets.Layout(width='50%', padding='0 0 0 20px')),
        ]),
        validation_output,
    ])

    results_box = widgets.VBox([
        widgets.HTML('<hr>'),
        plot_output,
    ])

    ui = widgets.VBox([header, config_box, results_box])
    display(ui)