import os
from pathlib import Path
from PIL import Image
import ipywidgets as widgets
from IPython.display import display, clear_output
from ipyfilechooser import FileChooser
from utils.tile_export import export_tiles, visualize_tile_locations

# ──────────────────────────────────────────────
# Tile Extraction UI
# ──────────────────────────────────────────────
def create_extraction_ui():
    default_path = os.path.join(os.getcwd(), 'sample_data', 'wsi','positive','2138813337')
    
    # 1. Choosers (Expanded width for long paths)
    svs_chooser = FileChooser(default_path)
    svs_chooser.filter_pattern = ['*.svs', '*.ndpi', '*.tiff']
    svs_chooser.title = '<b>1. Select WSI File:</b>'
    svs_chooser.layout.width = '80%'

    out_chooser = FileChooser(os.path.join(os.getcwd(), 'sample_data', 'wsi','positive','2138813337','tiles'))
    out_chooser.show_only_dirs = True
    out_chooser.title = '<b>2. Select Output Folder:</b>'
    out_chooser.layout.width = '80%'

    # ── 3. Tile Limit ──
    limit_title = widgets.HTML(
        value='<b>3. Tile Limit:</b>',
        layout=widgets.Layout(margin='10px 0px 0px 0px')
    )

    limit_toggle = widgets.RadioButtons(
        options=['Extract all tiles', 'Set custom limit'],
        value='Extract all tiles',
        layout=widgets.Layout(width='80%'),
    )

    limit_input = widgets.IntText(
        value=100,
        description='Max tiles:',
        layout=widgets.Layout(width='200px', display='none'),
        style={'description_width': '70px'},
    )

    limit_help = widgets.HTML(
        value="<i style='color:gray; font-size:12px;'>Total candidate tiles will be shown when extraction starts.</i>",
        layout=widgets.Layout(display='none'),
    )

    def on_toggle_change(change):
        if change['new'] == 'Set custom limit':
            limit_input.layout.display = ''
            limit_help.layout.display = ''
        else:
            limit_input.layout.display = 'none'
            limit_help.layout.display = 'none'

    limit_toggle.observe(on_toggle_change, names='value')       

    # 3. Add an optional checkbox for the visualization map
    map_title = widgets.HTML(
        value='<b>3. Post-Extraction:</b>',
        layout=widgets.Layout(margin='0px 0px 0px 0px')
    )
    
    show_map_checkbox = widgets.Checkbox(
        value=True,
        description='Generate and show tile map after extraction',
        indent=False,  
        layout=widgets.Layout(
            width='80%', 
            margin='0px',
        )
    )

    # 4. Button and Progress
    run_button = widgets.Button(
        description="Run Extraction", 
        button_style='success', 
        icon='play',
        layout=widgets.Layout(margin='20px 0px 10px 0px')
    )
    progress_bar = widgets.IntProgress(value=0, min=0, max=100, bar_style='info')
    progress_label = widgets.Label(value="Ready.")
    progress_ui = widgets.HBox([widgets.Label("Progress: "), progress_bar, progress_label])

    # 5. Output Sandbox
    output_area = widgets.Output()

    def on_button_click(b):
        # 1. LOCK THE BUTTON IMMEDIATELY to prevent double-clicks
        run_button.disabled = True
        run_button.description = "Running..."
        run_button.button_style = 'warning' # Turns yellow while busy
        
        with output_area:
            clear_output(wait=True)
            
            svs_path = svs_chooser.selected
            out_path = out_chooser.selected
            
            if not svs_path or not out_path:
                print("⚠️ Please select both an SVS file and an Output folder!")
                # If they made a mistake, unlock the button so they can try again
                run_button.disabled = False 
                run_button.description = "Run Extraction"
                run_button.button_style = 'success'
                return
            
            print(f"Starting pipeline for: {os.path.basename(svs_path)}")
            
            def update_progress(kept, discarded, total):
                current = kept + discarded
                progress_bar.max = total
                progress_bar.value = current
                percent = int((current / total) * 100) if total > 0 else 0
                progress_label.value = f"{current} / {total} tiles ({percent}%)"
                if current == total:
                    progress_bar.bar_style = 'success'
                    progress_label.value = f"Done! Kept {kept}, Discarded {discarded}."

            # Run extraction
            max_tiles = None
            if limit_toggle.value == 'Set custom limit':
                max_tiles = limit_input.value
                print(f"Tile limit set to: {max_tiles}")

            result = export_tiles(
                svs_path=svs_path,
                output_dir=out_path,
                max_tiles=max_tiles,
                progress_callback=update_progress
            )
            
            # ONLY show visualization if the user checked the box
            if show_map_checkbox.value:
                print("\nGenerating visualization map...")
                fig = visualize_tile_locations(slide_path=svs_path, tile_info=result["tile_info"])
                display(fig)
            else:
                print("\nExtraction complete! (Visualization skipped).")
                
        # 2. UNLOCK THE BUTTON WHEN FINISHED
        run_button.disabled = False
        run_button.description = "Run Extraction"
        run_button.button_style = 'success' # Turns back to green

    run_button.on_click(on_button_click)

    # Display everything on screen
    display(
        svs_chooser, 
        out_chooser, 
        limit_title,
        limit_toggle,
        limit_input,
        limit_help,
        map_title,
        show_map_checkbox,
        run_button, 
        progress_ui, 
        output_area
    )


# ──────────────────────────────────────────────
# Tile Scoring UI
# ──────────────────────────────────────────────
def create_scoring_ui():
    """
    Interactive UI for scoring tile images with the Keras model.
 
    The user selects:
        1. The Keras .h5 model file
        2. The folder containing tile images
        3. An output prefix for the CSV filename
 
    Output:
        - {prefix}tile_scores.csv saved in the tile folder
        - DataFrame stored in last_scoring_result["df"] for downstream use
    """
    from utils.tile_score import score_tiles
 
    default_model_path = os.path.join(os.getcwd(), 'model')
    default_tile_path = os.path.join(os.getcwd(), 'sample_data', 'tiles')
 
    # ── 1. Model file chooser ──
    model_chooser = FileChooser(default_model_path)
    model_chooser.filter_pattern = ['*.h5']
    model_chooser.title = '<b>1. Select Model File (.h5):</b>'
    model_chooser.layout.width = '80%'
 
    # ── 2. Tile folder chooser ──
    tile_chooser = FileChooser(default_tile_path)
    tile_chooser.show_only_dirs = True
    tile_chooser.title = '<b>2. Select Tile Folder:</b>'
    tile_chooser.layout.width = '80%'
 
     # ── 3. Output folder chooser ──
    output_chooser = FileChooser(default_tile_path)
    output_chooser.show_only_dirs = True
    output_chooser.title = '<b>3. Select Output Folder:</b>'
    output_chooser.layout.width = '80%'



    # ── 4. Output prefix ──
    prefix_label = widgets.HTML(value='<b>4. Output Prefix:</b>')
    prefix_input = widgets.Text(
        value='Version1.0_',
        placeholder='e.g., MyRun_',
        description='',
        layout=widgets.Layout(width='40%')
    )
    prefix_help = widgets.HTML(
        value='<i style="color: gray;">Output file will be saved as: {prefix}tile_scores.csv inside the tile folder.</i>'
    )
 
    # ── Run button ──
    run_button = widgets.Button(
        description="Run Scoring",
        button_style='success',
        icon='play',
        layout=widgets.Layout(margin='20px 0px 10px 0px')
    )
 
    # ── Progress bar ──
    progress_bar = widgets.IntProgress(value=0, min=0, max=100, bar_style='info')
    progress_label = widgets.Label(value="Ready.")
    progress_ui = widgets.HBox([widgets.Label("Progress: "), progress_bar, progress_label])
 
    # ── Output area ──
    output_area = widgets.Output()
 
    def on_button_click(b):
        # Lock button
        run_button.disabled = True
        run_button.description = "Running..."
        run_button.button_style = 'warning'
 
        # Reset progress
        progress_bar.value = 0
        progress_bar.bar_style = 'info'
        progress_label.value = "Starting..."
 
        with output_area:
            clear_output(wait=True)
 
            model_path = model_chooser.selected
            tile_path = tile_chooser.selected
            output_dir = output_chooser.selected
            prefix = prefix_input.value.strip()
 
            # ── Validate inputs ──
            if not model_path:
                print("⚠️ Please select a model file (.h5)!")
                run_button.disabled = False
                run_button.description = "Run Scoring"
                run_button.button_style = 'success'
                progress_label.value = "Ready."
                return
 
            if not tile_path:
                print("⚠️ Please select a tile folder!")
                run_button.disabled = False
                run_button.description = "Run Scoring"
                run_button.button_style = 'success'
                progress_label.value = "Ready."
                return
            
            if not output_dir:
                print("⚠️ Please select an output folder!")
                run_button.disabled = False
                run_button.description = "Run Scoring"
                run_button.button_style = 'success'
                progress_label.value = "Ready."
                return
 
            # ── Build output path ──
            csv_filename = f"{prefix}tile_scores.csv"
            output_csv = os.path.join(output_dir, csv_filename)
 
            print(f"Model:   {os.path.basename(model_path)}")
            print(f"Tiles:   {tile_path}")
            print(f"Output:  {csv_filename}")
            print()

            # Check for positive/negative subfolder structure
            tile_subdirs = [d.name.lower() for d in Path(tile_path).iterdir() if d.is_dir()]
            has_pos = any(d in ("positive", "pos", "malignant", "cancer", "adenoca", "adeno", "tumor", "tumour") for d in tile_subdirs)
            has_neg = any(d in ("negative", "neg", "benign", "normal", "nonneoplastic") for d in tile_subdirs)

            if has_pos and has_neg:
                print("   Detected positive/negative subfolders.")
                print("   Ground truth will be inferred from folder names.")
                print("   The 'truth' column in the output CSV will be populated.")
                print()
            elif has_pos or has_neg:
                print(f"⚠️ Only {'positive' if has_pos else 'negative'} subfolder detected.")
                print("   Both positive/ and negative/ are needed for full ground truth.")
                print("   The 'truth' column will be partially populated.")
                print()
            else:
                print("ℹ️ No positive/negative subfolders detected.")
                print("   All tiles will be scored, but the 'truth' column will be empty.")
                print("   (This is normal for unlabeled data — ROC evaluation will be skipped later.)")
                print()
 
            # ── Progress callback ──
            def update_progress(scored, total):
                progress_bar.max = total
                progress_bar.value = scored
                percent = int((scored / total) * 100) if total > 0 else 0
                progress_label.value = f"{scored} / {total} tiles ({percent}%)"
 
            # ── Run scoring ──
            print("Loading model...")
            try:
                df = score_tiles(
                    image_dir=tile_path,
                    model_path=model_path,
                    output_csv=output_csv,
                    progress_callback=update_progress,
                )
            except Exception as e:
                print(f"\n❌ Scoring failed: {e}")
                run_button.disabled = False
                run_button.description = "Run Scoring"
                run_button.button_style = 'success'
                progress_bar.bar_style = 'danger'
                progress_label.value = "Failed."
                return
 
            # ── Update progress to complete ──
            progress_bar.bar_style = 'success'
            progress_label.value = f"Done! Scored {len(df)} tiles."
 
            # ── Print summary ──
            print(f"\n    Scoring complete!")
            print(f"   Scored {len(df)} tiles")
            print(f"   Saved to: {csv_filename}")
 
            # Quick preview of first few rows
            n_pos = (df["truth"] == "positive").sum()
            n_neg = (df["truth"] == "negative").sum()
            n_unlabeled = (df["truth"] == "").sum()
 
            if n_pos > 0 or n_neg > 0:
                print(f"   Ground truth found: {n_pos} positive, {n_neg} negative")
            if n_unlabeled > 0:
                print(f"   Unlabeled tiles: {n_unlabeled}")
 
        # Unlock button
        run_button.disabled = False
        run_button.description = "Run Scoring"
        run_button.button_style = 'success'
 
    run_button.on_click(on_button_click)
 
    # ── Display everything ──
    display(
        model_chooser,
        tile_chooser,
        output_chooser,
        prefix_label,
        prefix_input,
        prefix_help,
        run_button,
        progress_ui,
        output_area,
    )

# ──────────────────────────────────────────────
# Evaluation UI (ROC, Statistics)
# ──────────────────────────────────────────────
def create_evaluation_ui():
    """
    Interactive UI for evaluating tile scores.

    The user selects:
        1. A tile_scores.csv file (from the scoring step)
        2. An output prefix

    Output files (saved in the same folder as the CSV):
        - {prefix}tile_stats.txt    summary statistics
        - {prefix}tile_roc.csv      ROC data points
        - {prefix}roc_curve.png     static ROC figure

    Also displays an interactive ROC curve with hover tooltips.
    """
    from utils.tile_score import (
        identify_positive_label,
        compute_statistics,
        compute_roc,
        compute_auc,
        plot_roc,
    )
    import plotly.graph_objects as go

    default_path = os.path.join(os.getcwd(), 'sample_data', 'tiles')

    # ── 1. CSV file chooser ──
    csv_chooser = FileChooser(default_path)
    csv_chooser.filter_pattern = ['*.csv']
    csv_chooser.title = '<b>1. Select tile_scores.csv:</b>'
    csv_chooser.layout.width = '80%'
    
    # ── 2. Output folder chooser ──
    output_chooser = FileChooser(default_path)
    output_chooser.show_only_dirs = True
    output_chooser.title = '<b>2. Select Output Folder:</b>'
    output_chooser.layout.width = '80%'


    # ── 3. Output prefix ──
    prefix_label = widgets.HTML(value='<b>3. Output Prefix:</b>')
    prefix_input = widgets.Text(
        value='Version1.0_',
        placeholder='e.g., MyRun_',
        description='',
        layout=widgets.Layout(width='40%')
    )
    prefix_help = widgets.HTML(
        value='<i style="color: gray;">Output files: {prefix}tile_stats.txt, {prefix}tile_roc.csv, {prefix}roc_curve.png</i>'
    )

    # ── Run button ──
    run_button = widgets.Button(
        description="Run Evaluation",
        button_style='success',
        icon='play',
        layout=widgets.Layout(margin='20px 0px 10px 0px')
    )

    # ── Output area ──
    output_area = widgets.Output()

    def on_button_click(b):
        run_button.disabled = True
        run_button.description = "Running..."
        run_button.button_style = 'warning'

        with output_area:
            clear_output(wait=True)

            csv_path = csv_chooser.selected
            prefix = prefix_input.value.strip()
            output_dir = output_chooser.selected


            # ── Validate CSV selection ──
            if not csv_path:
                print("⚠️ Please select a tile_scores.csv file!")
                run_button.disabled = False
                run_button.description = "Run Evaluation"
                run_button.button_style = 'success'
                return
            
            if not output_dir:
                print("⚠️ Please select an output folder!")
                run_button.disabled = False
                run_button.description = "Run Evaluation"
                run_button.button_style = 'success'
                return

            # ── Load CSV ──
            import pandas as pd
            print(f"Loading: {os.path.basename(csv_path)}")
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"❌ Failed to read CSV: {e}")
                run_button.disabled = False
                run_button.description = "Run Evaluation"
                run_button.button_style = 'success'
                return

            # ── Validate columns ──
            required_cols = {"truth"}
            missing = required_cols - set(df.columns)
            if missing:
                print(f"❌ CSV is missing required columns: {missing}")
                run_button.disabled = False
                run_button.description = "Run Evaluation"
                run_button.button_style = 'success'
                return

            prob_cols = [c for c in df.columns if c.startswith("p_")]
            if not prob_cols:
                print("❌ CSV has no probability columns (expected columns starting with 'p_').")
                run_button.disabled = False
                run_button.description = "Run Evaluation"
                run_button.button_style = 'success'
                return

            print(f"  Found {len(df)} rows, probability columns: {prob_cols}")

            # ── Compute statistics ──
            print("\nComputing statistics...")
            stats = compute_statistics(df)
            print(f"  Tiles:     {stats['n_tiles']}")
            print(f"  Score min: {stats['score_min']:.6f}")
            print(f"  Score mean:{stats['score_mean']:.6f}")
            print(f"  Score max: {stats['score_max']:.6f}")

            if stats['n_positive'] > 0 or stats['n_negative'] > 0:
                print(f"  Positive:  {stats['n_positive']}")
                print(f"  Negative:  {stats['n_negative']}")

            # ── Save stats file ──
            stats_path = os.path.join(output_dir, f"{prefix}tile_stats.txt")

            with open(stats_path, "w", encoding="utf-8") as f:
                f.write("Tile Evaluation Summary\n")
                f.write(f"Source: {os.path.basename(csv_path)}\n")
                f.write(f"Positive label: {stats['positive_label']}\n")
                f.write(f"Score column: {stats['score_column']}\n\n")
                f.write(f"N tiles:    {stats['n_tiles']}\n")
                f.write(f"Score min:  {stats['score_min']:.6f}\n")
                f.write(f"Score mean: {stats['score_mean']:.6f}\n")
                f.write(f"Score max:  {stats['score_max']:.6f}\n")
                if stats['has_truth']:
                    f.write(f"\nPositive tiles: {stats['n_positive']}\n")
                    f.write(f"Negative tiles: {stats['n_negative']}\n")

            print(f"\n  Saved: {os.path.basename(stats_path)}")

            # ── Check for truth labels before ROC ──
            if not stats['has_truth']:
                print("\n⚠️ No truth labels found (no positive/negative subfolders).")
                print("   Statistics saved. ROC skipped.")

                # Write AUC into stats file
                with open(stats_path, "a", encoding="utf-8") as f:
                    f.write("\nROC: skipped (no truth labels)\n")

                run_button.disabled = False
                run_button.description = "Run Evaluation"
                run_button.button_style = 'success'
                return

            # ── Compute ROC ──
            print("\nComputing ROC curve...")
            roc_df = compute_roc(df)
            auc = compute_auc(roc_df)
            print(f"  AUC = {auc:.3f}")

            # Append AUC to stats file
            with open(stats_path, "a", encoding="utf-8") as f:
                f.write(f"\nAUC: {auc:.6f}\n")

            # ── Save ROC CSV ──
            roc_csv_path = os.path.join(output_dir, f"{prefix}tile_roc.csv")
            roc_df.to_csv(roc_csv_path, index=False)
            print(f"  Saved: {os.path.basename(roc_csv_path)}")

            # ── Save static ROC PNG ──
            roc_png_path = os.path.join(output_dir, f"{prefix}roc_curve.png")
            fig_static = plot_roc(roc_df, auc)
            fig_static.savefig(roc_png_path, dpi=200, bbox_inches="tight")
            print(f"  Saved: {os.path.basename(roc_png_path)}")

            # ── Display interactive ROC (plotly) ──
            print("\n── Interactive ROC Curve ──")

            roc_sorted = roc_df.sort_values("fpr").drop_duplicates(subset=["fpr", "tpr"])

            fig_interactive = go.Figure()

            # ROC curve with hover data
            fig_interactive.add_trace(go.Scatter(
                x=roc_sorted["fpr"],
                y=roc_sorted["tpr"],
                mode="lines",
                name=f"ROC (AUC = {auc:.3f})",
                line=dict(color="royalblue", width=2),
                customdata=roc_sorted[["threshold", "precision", "recall", "f1"]].values,
                hovertemplate=(
                    "<b>Threshold:</b> %{customdata[0]:.4f}<br>"
                    "<b>TPR:</b> %{y:.4f}<br>"
                    "<b>FPR:</b> %{x:.4f}<br>"
                    "<b>Precision:</b> %{customdata[1]:.4f}<br>"
                    "<b>Recall:</b> %{customdata[2]:.4f}<br>"
                    "<b>F1:</b> %{customdata[3]:.4f}"
                    "<extra></extra>"
                ),
            ))

            # Random classifier line
            fig_interactive.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random classifier",
                line=dict(color="orange", dash="dash", width=1),
                hoverinfo="skip",
            ))

            fig_interactive.update_layout(
                title=f"ROC Curve (AUC = {auc:.3f})",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                xaxis=dict(range=[-0.1, 1.1], constrain="domain"),
                yaxis=dict(range=[-0.1, 1.1], scaleanchor="x"),
                width=800,
                height=800,
                legend=dict(x=0.55, y=0.05),
                template="plotly_white",
            )

            fig_interactive.show(renderer="iframe")

            # ── Final summary ──
            print(f"\n    Evaluation complete!")
            print(f"   AUC: {auc:.3f}")
            print(f"   Files saved in: {output_dir}")

        # Unlock button
        run_button.disabled = False
        run_button.description = "Run Evaluation"
        run_button.button_style = 'success'

    run_button.on_click(on_button_click)

    # ── Display everything ──
    display(
        csv_chooser,
        output_chooser,
        prefix_label,
        prefix_input,
        prefix_help,
        run_button,
        output_area,
    )


# ──────────────────────────────────────────────
# WSI Viewer UI
# ──────────────────────────────────────────────
def create_viewer_ui():
    """
    Interactive whole slide image viewer.
 
    The user selects an SVS file, clicks Load, then:
        - Clicks on the thumbnail to navigate to any region
        - Drags on the thumbnail to pan around
        - Uses the zoom slider to zoom in/out
 
    Layout:
        - Left:   thumbnail (preserves slide aspect ratio)
                  with red rectangle showing current viewport
        - Right:  square zoomed view of the current region
        - Above:  zoom slider
    """
    from utils.viewer import (
        open_slide,
        get_slide_summary,
        get_thumbnail,
        get_view_region,
        pil_to_png_bytes,
    )
    from PIL import ImageDraw
    from ipyevents import Event
 
    THUMB_MAX = 400
    VIEW_SIZE = 600
 
    default_path = os.path.join(os.getcwd(), 'sample_data', 'wsi','positive','2138813337')
 
    # ── State ──
    state = {
        "slide": None,
        "slide_w": 0,
        "slide_h": 0,
        "thumbnail": None,
        "thumb_w": 0,
        "thumb_h": 0,
        "thumb_scale": 1.0,
        "center_x": 0,
        "center_y": 0,
        "is_dragging": False,
    }
 
    # ── File chooser ──
    svs_chooser = FileChooser(default_path)
    svs_chooser.filter_pattern = ['*.svs', '*.ndpi', '*.tiff', '*.tif']
    svs_chooser.title = '<b>Select WSI File:</b>'
    svs_chooser.layout.width = '80%'
 
    # ── Load button ──
    load_button = widgets.Button(
        description="Load Slide",
        button_style='success',
        icon='eye',
        layout=widgets.Layout(margin='10px 0px')
    )
 
    # ── Info label ──
    info_label = widgets.HTML(value="<i>No slide loaded.</i>")
 
    # ── Thumbnail (clickable/draggable navigator) ──
    thumb_widget = widgets.Image(
        format='png',
        layout=widgets.Layout(
            border='1px solid #ccc',
            cursor='crosshair',
        )
    )
    thumb_label = widgets.HTML(
        value="<i style='color:gray; font-size:12px;'>Click or drag on the thumbnail to navigate</i>"
    )
    thumb_column = widgets.VBox([thumb_widget, thumb_label])
 
    # ── Viewport (square) ──
    view_widget = widgets.Image(
        format='png',
        width=VIEW_SIZE,
        height=VIEW_SIZE,
        layout=widgets.Layout(border='1px solid #ccc')
    )
 
    # ── Image row: thumbnail left, viewport right ──
    image_row = widgets.HBox(
        [thumb_column, view_widget],
        layout=widgets.Layout(margin='10px 0px', gap='20px')
    )
 
    # ── Zoom slider ──
    zoom_slider = widgets.FloatSlider(
        value=32.0,
        min=1.0,
        max=64.0,
        step=0.5,
        description='Zoom out:',
        readout_format='.1f',
        layout=widgets.Layout(width='80%'),
        style={'description_width': '80px'},
        continuous_update=True,
    )
 
    # ── Controls (hidden until slide is loaded) ──
    controls = widgets.VBox(
        [zoom_slider, image_row],
        layout=widgets.Layout(display='none')
    )
 
    # ── Output area ──
    output_area = widgets.Output()
 
    # ── Mouse events on thumbnail ──
    thumb_event = Event(
        source=thumb_widget,
        watched_events=['click', 'mousedown', 'mouseup', 'mousemove'],
    )
    thumb_event.prevent_default_action = True
 
    # ──────────────────────────────────────────
    # Rendering
    # ──────────────────────────────────────────
 
    def render_view():
        """Re-render the zoomed viewport and thumbnail overlay."""
        if state["slide"] is None:
            return
 
        slide = state["slide"]
        downsample = zoom_slider.value
        cx = state["center_x"]
        cy = state["center_y"]
        slide_w = state["slide_w"]
        slide_h = state["slide_h"]
        scale = state["thumb_scale"]
 
        # Update zoomed viewport (square)
        view_img = get_view_region(
            slide, cx, cy,
            downsample=downsample,
            view_width=VIEW_SIZE,
            view_height=VIEW_SIZE,
        )
        view_widget.value = pil_to_png_bytes(view_img)
 
        # Update thumbnail with red rectangle
        thumb_copy = state["thumbnail"].copy()
        draw = ImageDraw.Draw(thumb_copy)
 
        # Viewport extent in level-0 coordinates
        region_size = int(round(VIEW_SIZE * downsample))
 
        # Clamp top-left (same logic as get_view_region)
        x0 = max(0, cx - region_size // 2)
        y0 = max(0, cy - region_size // 2)
        if x0 + region_size > slide_w:
            x0 = max(0, slide_w - region_size)
        if y0 + region_size > slide_h:
            y0 = max(0, slide_h - region_size)
 
        # Convert to thumbnail pixel coordinates
        rx0 = x0 * scale
        ry0 = y0 * scale
        rx1 = (x0 + region_size) * scale
        ry1 = (y0 + region_size) * scale
 
        draw.rectangle([rx0, ry0, rx1, ry1], outline="red", width=2)
        thumb_widget.value = pil_to_png_bytes(thumb_copy)
 
    # ──────────────────────────────────────────
    # Mouse interaction on thumbnail
    # ──────────────────────────────────────────
 
    def set_position_from_mouse(event):
        """Convert thumbnail click coordinates to level-0 slide coordinates."""
        ox = event.get('offsetX', 0)
        oy = event.get('offsetY', 0)
        scale = state["thumb_scale"]
 
        if scale <= 0:
            return
 
        new_cx = int(ox / scale)
        new_cy = int(oy / scale)
 
        new_cx = max(0, min(new_cx, state["slide_w"]))
        new_cy = max(0, min(new_cy, state["slide_h"]))
 
        state["center_x"] = new_cx
        state["center_y"] = new_cy
        render_view()
 
    def handle_mouse(event):
        """Handle click and drag on the thumbnail."""
        etype = event.get('type', '')
 
        if etype == 'mousedown':
            state["is_dragging"] = True
            set_position_from_mouse(event)
        elif etype == 'mouseup':
            state["is_dragging"] = False
        elif etype == 'mousemove' and state["is_dragging"]:
            set_position_from_mouse(event)
        elif etype == 'click':
            set_position_from_mouse(event)
 
    thumb_event.on_dom_event(handle_mouse)
 
    # ──────────────────────────────────────────
    # Zoom slider change
    # ──────────────────────────────────────────
 
    def on_zoom_change(change):
        render_view()
 
    zoom_slider.observe(on_zoom_change, names='value')
 
    # ──────────────────────────────────────────
    # Load button
    # ──────────────────────────────────────────
 
    def on_load_click(b):
        load_button.disabled = True
        load_button.description = "Loading..."
        load_button.button_style = 'warning'
 
        with output_area:
            clear_output(wait=True)
 
            svs_path = svs_chooser.selected
            if not svs_path:
                print("⚠️ Please select a WSI file!")
                load_button.disabled = False
                load_button.description = "Load Slide"
                load_button.button_style = 'success'
                return
 
            # Close previous slide
            if state["slide"] is not None:
                state["slide"].close()
 
            print(f"Loading: {os.path.basename(svs_path)}")
 
            try:
                slide = open_slide(svs_path)
            except Exception as e:
                print(f"❌ Failed to open slide: {e}")
                load_button.disabled = False
                load_button.description = "Load Slide"
                load_button.button_style = 'success'
                return
 
            summary = get_slide_summary(slide)
            state["slide"] = slide
            state["slide_w"] = summary["dimensions"][0]
            state["slide_h"] = summary["dimensions"][1]
 
            # Generate thumbnail (aspect ratio preserved)
            thumb = get_thumbnail(slide, max_size=THUMB_MAX)
            state["thumbnail"] = thumb
            state["thumb_w"] = thumb.size[0]
            state["thumb_h"] = thumb.size[1]
            state["thumb_scale"] = thumb.size[0] / state["slide_w"]
 
            # Set widget size to match actual thumbnail pixels
            # (ensures click coordinates map 1:1 to image pixels)
            thumb_widget.width = state["thumb_w"]
            thumb_widget.height = state["thumb_h"]
 
            # Start centered
            state["center_x"] = state["slide_w"] // 2
            state["center_y"] = state["slide_h"] // 2
 
            # Show slide info
            mpp_str = f"{summary['mpp']:.4f} µm/px" if summary['mpp'] else "N/A"
            obj_str = f"{summary['objective']}×" if summary['objective'] else "N/A"
 
            info_label.value = (
                f"<b>{os.path.basename(svs_path)}</b> — "
                f"{summary['dimensions'][0]:,} × {summary['dimensions'][1]:,} px, "
                f"Objective: {obj_str}, "
                f"MPP: {mpp_str}, "
                f"{summary['level_count']} levels"
            )
 
            print("   Slide loaded.")
            print("   Click on the thumbnail to navigate.")
            print("   Drag on the thumbnail to pan.")
            print("   Use the zoom slider to zoom in/out.")
 
            # Show controls and render
            controls.layout.display = ''
            render_view()
 
        load_button.disabled = False
        load_button.description = "Load Slide"
        load_button.button_style = 'success'
 
    load_button.on_click(on_load_click)
 
    # ── Display ──
    display(
        svs_chooser,
        load_button,
        info_label,
        output_area,
        controls,
    )

# ──────────────────────────────────────────────
# Click-to-Predict UI
# ──────────────────────────────────────────────

def create_predict_ui():
    """
    Interactive click-to-predict tool.

    The user selects a model and a slide, then clicks anywhere
    on the slide thumbnail. At the click location, a 512x512
    tile is extracted at ~10x magnification, displayed in a
    viewport, and run through the model. The prediction updates
    each time the user clicks a new location.
    """
    from utils.viewer import (
        open_slide,
        get_slide_summary,
        get_thumbnail,
        pil_to_png_bytes,
    )
    from utils.tile_export import (
        calculate_downsample,
        get_best_level,
        read_tile,
    )
    from utils.tile_score import (
        load_model,
        get_image_size_from_model,
    )
    from PIL import Image as PILImage, ImageDraw
    from ipyevents import Event
    import numpy as np

    THUMB_MAX = 500
    TILE_SIZE = 512

    default_model_path = os.path.join(os.getcwd(), 'model')
    default_wsi_path = os.path.join(os.getcwd(), 'sample_data', 'wsi','positive','2138813337')

    # ── State ──
    state = {
        "slide": None,
        "slide_w": 0,
        "slide_h": 0,
        "thumbnail": None,
        "thumb_scale": 1.0,
        "downsample": 1.0,
        "level": 0,
        "model": None,
        "image_size": 224,
        "tile_display_size": 300,
    }

    # ── 1. Model file chooser ──
    model_chooser = FileChooser(default_model_path)
    model_chooser.filter_pattern = ['*.h5']
    model_chooser.title = '<b>1. Select Model File (.h5):</b>'
    model_chooser.layout.width = '80%'

    # ── 2. Slide file chooser ──
    svs_chooser = FileChooser(default_wsi_path)
    svs_chooser.filter_pattern = ['*.svs', '*.ndpi', '*.tiff', '*.tif']
    svs_chooser.title = '<b>2. Select WSI File:</b>'
    svs_chooser.layout.width = '80%'

    # ── Load button ──
    load_button = widgets.Button(
        description="Load Model & Slide",
        button_style='success',
        icon='eye',
        layout=widgets.Layout(margin='10px 0px')
    )

    # ── Info label ──
    info_label = widgets.HTML(value="<i>No slide loaded.</i>")

    # ── Thumbnail (clickable) ──
    thumb_widget = widgets.Image(
        format='png',
        layout=widgets.Layout(
            border='1px solid #ccc',
            cursor='crosshair',
        )
    )
    thumb_label = widgets.HTML(
        value="<i style='color:gray; font-size:12px;'>Click anywhere on the slide to predict that region</i>"
    )
    thumb_column = widgets.VBox([thumb_widget, thumb_label])

    # ── Tile viewport (square) ──
    view_widget = widgets.Image(
        format='png',
        layout=widgets.Layout(border='1px solid #ccc')
    )
    view_label = widgets.HTML(
        value="<i style='color:gray; font-size:12px;'>Extracted tile will appear here</i>"
    )

    # ── Prediction display ──
    prediction_label = widgets.HTML(
        value="",
        layout=widgets.Layout(margin='10px 0px')
    )

    view_column = widgets.VBox([view_widget, view_label, prediction_label])

    # ── Image row ──
    image_row = widgets.HBox(
        [thumb_column, view_column],
        layout=widgets.Layout(margin='10px 0px', gap='20px')
    )

    # ── Controls (hidden until loaded) ──
    controls = widgets.VBox(
        [image_row],
        layout=widgets.Layout(display='none')
    )

    # ── Output area ──
    output_area = widgets.Output()

    # ── Mouse events on thumbnail ──
    thumb_event = Event(
        source=thumb_widget,
        watched_events=['click'],
    )
    thumb_event.prevent_default_action = True

    # ──────────────────────────────────────────
    # Click handler
    # ──────────────────────────────────────────

    def on_thumb_click(event):
        """Handle click on thumbnail: extract tile, predict, display."""
        if state["slide"] is None or state["model"] is None:
            return

        etype = event.get('type', '')
        if etype != 'click':
            return

        # ── Convert click to level-0 coordinates ──
        ox = event.get('offsetX', 0)
        oy = event.get('offsetY', 0)
        scale = state["thumb_scale"]

        if scale <= 0:
            return

        center_x = int(ox / scale)
        center_y = int(oy / scale)

        center_x = max(0, min(center_x, state["slide_w"]))
        center_y = max(0, min(center_y, state["slide_h"]))

        slide = state["slide"]
        downsample = state["downsample"]
        level = state["level"]

        # ── Compute tile region in level-0 coordinates ──
        tile_extent = int(round(TILE_SIZE * downsample))
        x0 = max(0, center_x - tile_extent // 2)
        y0 = max(0, center_y - tile_extent // 2)

        if x0 + tile_extent > state["slide_w"]:
            x0 = max(0, state["slide_w"] - tile_extent)
        if y0 + tile_extent > state["slide_h"]:
            y0 = max(0, state["slide_h"] - tile_extent)

        # ── Draw red dot and tile box on thumbnail ──
        thumb_copy = state["thumbnail"].copy()
        draw = ImageDraw.Draw(thumb_copy)

        dot_r = 5
        draw.ellipse(
            [ox - dot_r, oy - dot_r, ox + dot_r, oy + dot_r],
            fill="red",
            outline="darkred",
            width=1,
        )

        rx0 = x0 * scale
        ry0 = y0 * scale
        rx1 = (x0 + tile_extent) * scale
        ry1 = (y0 + tile_extent) * scale
        draw.rectangle([rx0, ry0, rx1, ry1], outline="red", width=2)

        thumb_widget.value = pil_to_png_bytes(thumb_copy)

        # ── Extract tile ──
        tile_rgb = read_tile(
            slide,
            x0, y0,
            tile_extent, tile_extent,
            TILE_SIZE, TILE_SIZE,
            level=level,
            downsample=downsample,
        )

        # ── Display tile in viewport ──
        tile_pil = PILImage.fromarray(tile_rgb)
        ds = state["tile_display_size"]
        tile_display = tile_pil.resize((ds, ds), PILImage.LANCZOS)
        view_widget.value = pil_to_png_bytes(tile_display)

        # ── Run prediction ──
        model = state["model"]
        image_size = state["image_size"]

        tile_input = tile_pil.resize((image_size, image_size))
        arr = np.asarray(tile_input).astype(np.float32)
        arr = (arr / 127.5) - 1.0
        arr = np.expand_dims(arr, axis=0)

        probs = model.predict(arr, verbose=0)[0]
        p_positive = float(probs[0])
        p_negative = float(probs[1])

        pred_label = "Positive (Malignant)" if p_positive > p_negative else "Negative (Benign)"

        if p_positive > 0.8:
            color = "#d32f2f"
        elif p_positive > 0.5:
            color = "#f57c00"
        elif p_positive > 0.2:
            color = "#fbc02d"
        else:
            color = "#388e3c"

        prediction_label.value = (
            f"<div style='padding:10px; border-radius:8px; border:2px solid {color}; "
            f"background-color:{color}10; max-width:{ds}px;'>"
            f"<b style='font-size:16px; color:{color};'>{pred_label}</b><br>"
            f"<span style='font-size:13px;'>"
            f"p(Positive) = <b>{p_positive:.4f}</b> &nbsp;&nbsp; "
            f"p(Negative) = <b>{p_negative:.4f}</b>"
            f"</span><br>"
            f"<span style='font-size:11px; color:gray;'>"
            f"Location: ({center_x:,}, {center_y:,})"
            f"</span>"
            f"</div>"
        )

        view_label.value = (
            f"<i style='color:gray; font-size:12px;'>"
            f"512×512 tile at ~10× from ({center_x:,}, {center_y:,})"
            f"</i>"
        )

    thumb_event.on_dom_event(on_thumb_click)

    # ──────────────────────────────────────────
    # Load button
    # ──────────────────────────────────────────

    def on_load_click(b):
        load_button.disabled = True
        load_button.description = "Loading..."
        load_button.button_style = 'warning'

        with output_area:
            clear_output(wait=True)

            model_path = model_chooser.selected
            svs_path = svs_chooser.selected

            if not model_path:
                print("⚠️ Please select a model file (.h5)!")
                load_button.disabled = False
                load_button.description = "Load Model & Slide"
                load_button.button_style = 'success'
                return

            if not svs_path:
                print("⚠️ Please select a WSI file!")
                load_button.disabled = False
                load_button.description = "Load Model & Slide"
                load_button.button_style = 'success'
                return

            # ── Load model ──
            print(f"Loading model: {os.path.basename(model_path)}")
            try:
                model = load_model(model_path)
                image_size = get_image_size_from_model(model)
                state["model"] = model
                state["image_size"] = image_size
                print(f"  Model input size: {image_size}×{image_size}")
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                load_button.disabled = False
                load_button.description = "Load Model & Slide"
                load_button.button_style = 'success'
                return

            # ── Load slide ──
            if state["slide"] is not None:
                state["slide"].close()

            print(f"Loading slide: {os.path.basename(svs_path)}")
            try:
                slide = open_slide(svs_path)
            except Exception as e:
                print(f"❌ Failed to open slide: {e}")
                load_button.disabled = False
                load_button.description = "Load Model & Slide"
                load_button.button_style = 'success'
                return

            summary = get_slide_summary(slide)
            state["slide"] = slide
            state["slide_w"] = summary["dimensions"][0]
            state["slide_h"] = summary["dimensions"][1]

            downsample = calculate_downsample(slide)
            level = get_best_level(slide, downsample)
            state["downsample"] = downsample
            state["level"] = level

            # Generate thumbnail
            thumb = get_thumbnail(slide, max_size=THUMB_MAX)
            state["thumbnail"] = thumb
            state["thumb_scale"] = thumb.size[0] / state["slide_w"]

            # Set thumbnail widget size
            thumb_w = thumb.size[0]
            thumb_h = thumb.size[1]
            thumb_widget.width = thumb_w
            thumb_widget.height = thumb_h
            thumb_widget.value = pil_to_png_bytes(thumb)

            # Set tile viewport to match thumbnail height (square)
            state["tile_display_size"] = thumb_h
            view_widget.width = thumb_h
            view_widget.height = thumb_h

            # Show info
            mpp_str = f"{summary['mpp']:.4f} µm/px" if summary['mpp'] else "N/A"
            obj_str = f"{summary['objective']}×" if summary['objective'] else "N/A"

            info_label.value = (
                f"<b>{os.path.basename(svs_path)}</b> — "
                f"{summary['dimensions'][0]:,} × {summary['dimensions'][1]:,} px, "
                f"Objective: {obj_str}, "
                f"MPP: {mpp_str}, "
                f"Downsample: {downsample:.2f}"
            )

            print("   Model and slide loaded.")
            print("   Click anywhere on the slide thumbnail to predict.")

            controls.layout.display = ''

            # Trigger initial prediction at default location
            initial_x = 54605
            initial_y = 17309
            fake_event = {
                'type': 'click',
                'offsetX': int(initial_x * state["thumb_scale"]),
                'offsetY': int(initial_y * state["thumb_scale"]),
            }
            on_thumb_click(fake_event)

        load_button.disabled = False
        load_button.description = "Load Model & Slide"
        load_button.button_style = 'success'

    load_button.on_click(on_load_click)

    # ── Display ──
    display(
        model_chooser,
        svs_chooser,
        load_button,
        info_label,
        output_area,
        controls,
    )


# ──────────────────────────────────────────────
# Case-Level Analysis UI (Top-K Threshold)
# ──────────────────────────────────────────────

def create_case_analysis_ui():
    """
    Interactive case-level classification using top-K tile thresholding.

    The user selects a WSI directory containing positive/ and negative/
    subfolders. Each subfolder contains slide folders with tile_scores.csv.

    Two sliders control the classification:
        - Tile threshold: what probability counts as a "malignant tile"
        - K value: how many malignant tiles needed to call a case positive

    A confusion matrix updates live as the user adjusts the sliders.
    """
    from utils.case_score import load_case_data, classify_cases, compute_confusion_matrix, find_optimal_k

    default_path = os.path.join(os.getcwd(), 'sample_data', 'wsi')

    # ── State ──
    state = {
        "cases": None,
        "max_tiles": 0,
    }

    # ── 1. Folder chooser ──
    folder_chooser = FileChooser(default_path)
    folder_chooser.show_only_dirs = True
    folder_chooser.title = '<b>1. Select WSI Directory (must contain positive/ and negative/ subfolders):</b>'
    folder_chooser.layout.width = '80%'

    # ── 2. CSV filename ──
    csv_label = widgets.HTML(value='<b>2. Scores CSV Filename:</b>')
    csv_input = widgets.Text(
        value='batch_tile_scores.csv',
        placeholder='e.g., batch_tile_scores.csv',
        layout=widgets.Layout(width='40%'),
    )

    # ── Load button ──
    load_button = widgets.Button(
        description="Load Cases",
        button_style='success',
        icon='folder-open',
        layout=widgets.Layout(margin='10px 0px'),
    )

    # ── Info label ──
    info_label = widgets.HTML(value="<i>No cases loaded.</i>")

    # ── Output area for load messages ──
    load_output = widgets.Output()

    # ── K mode: count vs percent ──
    k_mode_label = widgets.HTML(value='<b>3. Top-K Mode:</b>')
    k_mode_radio = widgets.RadioButtons(
        options=['Absolute count (K tiles)', 'Percentage (K% of total tiles)'],
        value='Absolute count (K tiles)',
        layout=widgets.Layout(width='80%'),
    )

    # ── Tile threshold slider ──
    tile_threshold_slider = widgets.FloatSlider(
        value=0.5,
        min=0.0,
        max=1.0,
        step=0.01,
        description='Tile threshold:',
        readout_format='.2f',
        layout=widgets.Layout(width='80%'),
        style={'description_width': '120px'},
        continuous_update=True,
    )
    tile_threshold_help = widgets.HTML(
        value="<i style='color:gray; font-size:12px;'>Tiles with p_Positive ≥ this value are counted as malignant</i>"
    )

    # ── K slider (count mode) ──
    k_count_slider = widgets.IntSlider(
        value=1,
        min=1,
        max=100,
        step=1,
        description='K (tiles):',
        layout=widgets.Layout(width='80%'),
        style={'description_width': '120px'},
        continuous_update=True,
    )

    # ── K slider (percent mode) ──
    k_percent_slider = widgets.FloatSlider(
        value=5.0,
        min=0.1,
        max=100.0,
        step=0.1,
        description='K (%):',
        readout_format='.1f',
        layout=widgets.Layout(width='80%', display='none'),
        style={'description_width': '120px'},
        continuous_update=True,
    )

    # ── Optimal K info ──
    optimal_k_label = widgets.HTML(value="")

    # ── Confusion matrix display ──
    confusion_display = widgets.HTML(value="")

    # ── Per-case detail table ──
    detail_display = widgets.HTML(value="")

    # ── Controls (hidden until loaded) ──
    controls = widgets.VBox(
        [
            k_mode_label,
            k_mode_radio,
            tile_threshold_slider,
            tile_threshold_help,
            k_count_slider,
            k_percent_slider,
            optimal_k_label,
            confusion_display,
            detail_display,
        ],
        layout=widgets.Layout(display='none'),
    )

    # ──────────────────────────────────────────
    # Toggle K slider mode
    # ──────────────────────────────────────────

    def on_mode_change(change):
        if change['new'] == 'Absolute count (K tiles)':
            k_count_slider.layout.display = ''
            k_percent_slider.layout.display = 'none'
        else:
            k_count_slider.layout.display = 'none'
            k_percent_slider.layout.display = ''
        update_analysis()

    k_mode_radio.observe(on_mode_change, names='value')

    # ──────────────────────────────────────────
    # Update confusion matrix
    # ──────────────────────────────────────────

    def update_analysis(change=None):
        if state["cases"] is None:
            return

        tile_threshold = tile_threshold_slider.value

        if k_mode_radio.value == 'Absolute count (K tiles)':
            k_value = k_count_slider.value
            k_mode = "count"
        else:
            k_value = k_percent_slider.value
            k_mode = "percent"

        results = classify_cases(
            state["cases"],
            tile_threshold=tile_threshold,
            k_value=k_value,
            k_mode=k_mode,
        )

        cm = compute_confusion_matrix(results)

        # Color code the confusion matrix cells
        tp_color = "#c8e6c9" if cm["tp"] > 0 else "#ffffff"
        tn_color = "#c8e6c9" if cm["tn"] > 0 else "#ffffff"
        fp_color = "#ffcdd2" if cm["fp"] > 0 else "#ffffff"
        fn_color = "#ffcdd2" if cm["fn"] > 0 else "#ffffff"

        # Perfect classification highlight
        if cm["fp"] == 0 and cm["fn"] == 0 and cm["total"] > 0:
            border_style = "3px solid #388e3c"
            status_text = "<b style='color:#388e3c; font-size:16px;'>     Perfect Classification!</b>"
        else:
            border_style = "1px solid #ccc"
            status_text = ""

        confusion_display.value = f"""
        <div style='margin:15px 0px;'>
            {status_text}
            <table style='border-collapse:collapse; margin:10px 0px; font-size:14px; border:{border_style};'>
                <tr>
                    <td style='padding:8px; border:1px solid #ccc;'></td>
                    <td style='padding:8px; border:1px solid #ccc; font-weight:bold; text-align:center;'>Predicted Positive</td>
                    <td style='padding:8px; border:1px solid #ccc; font-weight:bold; text-align:center;'>Predicted Negative</td>
                </tr>
                <tr>
                    <td style='padding:8px; border:1px solid #ccc; font-weight:bold;'>Actual Positive</td>
                    <td style='padding:12px 20px; border:1px solid #ccc; text-align:center; background-color:{tp_color};'>
                        <b>TP = {cm["tp"]}</b>
                    </td>
                    <td style='padding:12px 20px; border:1px solid #ccc; text-align:center; background-color:{fn_color};'>
                        <b>FN = {cm["fn"]}</b>
                    </td>
                </tr>
                <tr>
                    <td style='padding:8px; border:1px solid #ccc; font-weight:bold;'>Actual Negative</td>
                    <td style='padding:12px 20px; border:1px solid #ccc; text-align:center; background-color:{fp_color};'>
                        <b>FP = {cm["fp"]}</b>
                    </td>
                    <td style='padding:12px 20px; border:1px solid #ccc; text-align:center; background-color:{tn_color};'>
                        <b>TN = {cm["tn"]}</b>
                    </td>
                </tr>
            </table>
            <div style='font-size:13px; margin-top:5px;'>
                Sensitivity: <b>{cm["sensitivity"]:.3f}</b> &nbsp;&nbsp;
                Specificity: <b>{cm["specificity"]:.3f}</b> &nbsp;&nbsp;
                Accuracy: <b>{cm["accuracy"]:.3f}</b> &nbsp;&nbsp;
                ({cm["n_correct"]}/{cm["total"]} correct)
            </div>
        </div>
        """

        # Per-case detail table
        rows_html = ""
        for r in results:
            if r["correct"]:
                row_color = "#f1f8e9"
                icon = "     "
            else:
                row_color = "#fce4ec"
                icon = "❌"

            rows_html += f"""
            <tr style='background-color:{row_color};'>
                <td style='padding:5px 10px; border:1px solid #ddd;'>{r["name"]}</td>
                <td style='padding:5px 10px; border:1px solid #ddd; text-align:center;'>{r["truth"]}</td>
                <td style='padding:5px 10px; border:1px solid #ddd; text-align:center;'>{r["n_tiles"]}</td>
                <td style='padding:5px 10px; border:1px solid #ddd; text-align:center;'>{r["n_malignant"]}</td>
                <td style='padding:5px 10px; border:1px solid #ddd; text-align:center;'>{r["k_threshold"]}</td>
                <td style='padding:5px 10px; border:1px solid #ddd; text-align:center;'>{r["pred"]}</td>
                <td style='padding:5px 10px; border:1px solid #ddd; text-align:center;'>{icon}</td>
            </tr>
            """

        detail_display.value = f"""
        <div style='margin:10px 0px;'>
            <b>Per-Case Breakdown:</b>
            <table style='border-collapse:collapse; margin:5px 0px; font-size:12px; width:100%;'>
                <tr style='background-color:#e0e0e0;'>
                    <th style='padding:6px 10px; border:1px solid #ccc; text-align:left;'>Slide</th>
                    <th style='padding:6px 10px; border:1px solid #ccc; text-align:center;'>Truth</th>
                    <th style='padding:6px 10px; border:1px solid #ccc; text-align:center;'>Total Tiles</th>
                    <th style='padding:6px 10px; border:1px solid #ccc; text-align:center;'>Malignant Tiles</th>
                    <th style='padding:6px 10px; border:1px solid #ccc; text-align:center;'>K Threshold</th>
                    <th style='padding:6px 10px; border:1px solid #ccc; text-align:center;'>Prediction</th>
                    <th style='padding:6px 10px; border:1px solid #ccc; text-align:center;'>Correct</th>
                </tr>
                {rows_html}
            </table>
        </div>
        """

    # Wire sliders to update
    tile_threshold_slider.observe(update_analysis, names='value')
    k_count_slider.observe(update_analysis, names='value')
    k_percent_slider.observe(update_analysis, names='value')

    # ──────────────────────────────────────────
    # Load button
    # ──────────────────────────────────────────

    def on_load_click(b):
        load_button.disabled = True
        load_button.description = "Loading..."
        load_button.button_style = 'warning'

        with load_output:
            clear_output(wait=True)

            wsi_dir = folder_chooser.selected
            csv_filename = csv_input.value.strip()

            if not wsi_dir:
                print("⚠️ Please select a WSI directory!")
                load_button.disabled = False
                load_button.description = "Load Cases"
                load_button.button_style = 'success'
                return

            # Check for positive/negative subfolders
            pos_dir = Path(wsi_dir) / "positive"
            neg_dir = Path(wsi_dir) / "negative"

            if not pos_dir.exists() and not neg_dir.exists():
                print("❌ Directory must contain 'positive' and/or 'negative' subfolders.")
                print(f"   Looked in: {wsi_dir}")
                load_button.disabled = False
                load_button.description = "Load Cases"
                load_button.button_style = 'success'
                return

            print(f"Loading cases from: {wsi_dir}")
            print(f"Looking for: {csv_filename}")

            cases = load_case_data(
                wsi_dir=wsi_dir,
                csv_filename=csv_filename,
            )

            if not cases:
                print("❌ No scored slides found. Make sure each slide folder has a tile_scores.csv.")
                load_button.disabled = False
                load_button.description = "Load Cases"
                load_button.button_style = 'success'
                return

            state["cases"] = cases

            n_pos = sum(1 for c in cases if c["truth"] == "positive")
            n_neg = sum(1 for c in cases if c["truth"] == "negative")
            max_tiles = max(c["n_tiles"] for c in cases)
            state["max_tiles"] = max_tiles

            # Update K count slider range
            k_count_slider.max = max_tiles
            k_count_slider.value = min(1, max_tiles)

            print(f"\n    Loaded {len(cases)} cases: {n_pos} positive, {n_neg} negative")
            print(f"   Tile counts range: {min(c['n_tiles'] for c in cases)} – {max_tiles}")

            # Show summary
            info_label.value = (
                f"<b>{len(cases)} cases loaded</b> — "
                f"{n_pos} positive, {n_neg} negative"
            )

            # Show controls and run initial analysis
# Find optimal K and set as default
            if k_mode_radio.value == 'Absolute count (K tiles)':
                k_mode = "count"
            else:
                k_mode = "percent"

            result = find_optimal_k(
                cases,
                tile_threshold=tile_threshold_slider.value,
                k_mode=k_mode,
            )

            if k_mode == "count":
                k_count_slider.value = int(result["k_optimal"])
            else:
                k_percent_slider.value = float(result["k_optimal"])

            if result["found_perfect"]:
                optimal_k_label.value = (
                    f"<span style='color:#388e3c; font-size:13px;'>"
                    f"    Optimal K = <b>{result['k_optimal']}</b> "
                    f"(perfect classification: sensitivity=1.0, specificity=1.0)</span>"
                )
            else:
                pass

            print(f"   Optimal K: {result['k_optimal']} ({k_mode} mode)")

            # Show controls and run initial analysis
            controls.layout.display = ''
            update_analysis()

        load_button.disabled = False
        load_button.description = "Load Cases"
        load_button.button_style = 'success'

    load_button.on_click(on_load_click)

    # ── Display ──
    display(
        folder_chooser,
        csv_label,
        csv_input,
        load_button,
        info_label,
        load_output,
        controls,
    )