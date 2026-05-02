"""
Fine-Tuning UI for the CRC Screening Tool Notebook
===================================================
Adds Step 7: Fine-Tune Model to the notebook.

Usage (in a notebook cell):
    from utils.finetune_ui import create_finetuning_ui
    create_finetuning_ui()

Features:
    - Folder selection with validation (positive/negative subfolders)
    - Option to combine with existing tiles or use new tiles only
    - Frozen backbone (head-only training)
    - Configurable epochs
    - Live updating training curves (accuracy, AUC, loss, weighted F1)
    - Before/after comparison table with confusion matrices
    - Save model button (saves to model/ with timestamp)
"""

import os
import shutil
import random
import datetime
import numpy as np
from pathlib import Path
from collections import Counter
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import F1Score
from keras.models import load_model
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
    f1_score, precision_score, recall_score, accuracy_score
)
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from ipyfilechooser import FileChooser

# ── Constants ────────────────────────────────────────────────────────────────

IMG_SIZE         = 224
BATCH_SIZE       = 32
POSITIVE_INDEX   = 0        # model convention: index 0 = Positive
AUTOTUNE         = tf.data.AUTOTUNE
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
SEED             = 42


# ── Helper Functions ─────────────────────────────────────────────────────────

def validate_tile_folder(folder_path):
    """Check if a folder has positive/ and negative/ subfolders with images.
    Returns (is_valid, pos_count, neg_count, message)."""
    folder = Path(folder_path)
    if not folder.exists():
        return False, 0, 0, f"Folder does not exist: {folder}"

    pos_dir = folder / "positive"
    neg_dir = folder / "negative"

    if not pos_dir.exists() or not neg_dir.exists():
        return False, 0, 0, "Folder must contain 'positive/' and 'negative/' subfolders."

    pos_count = sum(1 for f in pos_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
    neg_count = sum(1 for f in neg_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)

    if pos_count == 0 or neg_count == 0:
        return False, pos_count, neg_count, f"Both subfolders need images. Found: {pos_count} positive, {neg_count} negative."

    return True, pos_count, neg_count, f"Found {pos_count} positive and {neg_count} negative tiles."


def load_tiles_for_eval(tiles_dir, img_size=224):
    """Load tiles from positive/ and negative/ subfolders for evaluation."""
    images, truths = [], []
    for subfolder in sorted(Path(tiles_dir).iterdir()):
        if not subfolder.is_dir():
            continue
        label = subfolder.name.lower()
        for img_path in sorted(subfolder.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            img = Image.open(img_path).convert("RGB")
            img = img.resize((img_size, img_size))
            arr = np.array(img, dtype=np.float32)
            arr = (arr / 127.5) - 1.0
            images.append(arr)
            truths.append(label)
    return np.stack(images, axis=0), truths


def evaluate_model(model, images, truths):
    """Run inference and return metrics dict."""
    predictions = model.predict(images, batch_size=BATCH_SIZE, verbose=0)
    p_positive = predictions[:, POSITIVE_INDEX]
    pred_labels = ["positive" if p >= 0.5 else "negative" for p in p_positive]

    y_true = np.array([1 if t == "positive" else 0 for t in truths])
    y_pred = np.array([1 if p == "positive" else 0 for p in pred_labels])
    cm = confusion_matrix(y_true, y_pred)

    return {
        'y_true': y_true, 'y_pred': y_pred, 'y_prob': p_positive, 'cm': cm,
        'acc':      accuracy_score(y_true, y_pred),
        'auc':      roc_auc_score(y_true, p_positive),
        'f1_w':     f1_score(y_true, y_pred, average='weighted'),
        'prec_pos': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'prec_neg': precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        'rec_pos':  recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'rec_neg':  recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        'n_total':  len(y_true),
        'n_pos':    int(y_true.sum()),
        'n_neg':    int((1 - y_true).sum()),
    }


def copy_tiles_to_dir(source_dir, dest_dir, prefix):
    """Copy tile images from source positive/negative into dest positive/negative."""
    count = {"positive": 0, "negative": 0}
    for label in ["positive", "negative"]:
        src = Path(source_dir) / label
        if not src.exists():
            continue
        for img_path in sorted(src.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                dest = Path(dest_dir) / label / f"{prefix}_{img_path.name}"
                shutil.copy2(img_path, dest)
                count[label] += 1
    return count


# ── Live Plot Callback ───────────────────────────────────────────────────────

class LivePlotCallback(keras.callbacks.Callback):
    """Keras callback that updates training curves in an Output widget after each epoch.
    Also checks a stop flag to allow the user to cancel training between epochs."""

    def __init__(self, output_widget, progress_label, stop_flag):
        super().__init__()
        self.output = output_widget
        self.progress_label = progress_label
        self.stop_flag = stop_flag        # mutable list: [False]. Set [True] to stop.
        self.history = {
            'accuracy': [], 'val_accuracy': [],
            'auc': [], 'val_auc': [],
            'loss': [], 'val_loss': [],
            'f1_score_weighted': [], 'val_f1_score_weighted': [],
        }
        self.total_epochs = 0

    def set_total_epochs(self, n):
        self.total_epochs = n

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key in self.history:
            if key in logs:
                self.history[key].append(logs[key])

        # Check if user requested stop
        if self.stop_flag[0]:
            self.model.stop_training = True
            self.progress_label.value = (
                f'<b style="color: #e65100;">Training stopped by user at epoch {epoch + 1}.</b>'
            )

        # Update progress label
        val_auc = logs.get('val_auc', 0)
        val_acc = logs.get('val_accuracy', 0)
        if not self.stop_flag[0]:
            self.progress_label.value = (
                f"Epoch {epoch + 1}/{self.total_epochs}  —  "
                f"val_AUC: {val_auc:.4f}  |  val_Accuracy: {val_acc:.4f}"
            )

        # Redraw plots
        with self.output:
            clear_output(wait=True)

            fig, axes = plt.subplots(2, 2, figsize=(11, 7))

            plots = [
                (axes[0, 0], 'accuracy',          'val_accuracy',          'Accuracy'),
                (axes[0, 1], 'auc',               'val_auc',               'AUC'),
                (axes[1, 0], 'loss',              'val_loss',               'Loss'),
                (axes[1, 1], 'f1_score_weighted', 'val_f1_score_weighted',  'Weighted F1'),
            ]

            for ax, train_key, val_key, title in plots:
                if self.history[train_key]:
                    epochs = range(1, len(self.history[train_key]) + 1)
                    ax.plot(epochs, self.history[train_key], 'o-', markersize=3,
                            color='steelblue', label='Train')
                    ax.plot(epochs, self.history[val_key], 'o-', markersize=3,
                            color='indianred', label='Validation')
                    ax.set_title(title, fontsize=11)
                    ax.set_xlabel('Epoch', fontsize=9)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)

            plt.suptitle('Training Progress', fontsize=13)
            plt.tight_layout()
            plt.show()


# ── Results Display ──────────────────────────────────────────────────────────

def display_results(before, after, dataset_name, output_widget):
    """Display before/after comparison table and confusion matrices."""

    def delta_html(before_val, after_val):
        d = after_val - before_val
        if d > 0:
            return f'<span style="color: green;">▲ {d:.4f}</span>'
        elif d < 0:
            return f'<span style="color: red;">▼ {abs(d):.4f}</span>'
        return f'<span>─ 0.0000</span>'

    def cm_html(cm, title):
        return f"""
        <h4 style="margin: 10px 0 5px 0;">{title}</h4>
        <table style="border-collapse: collapse; font-size: 13px; margin-bottom: 10px;">
            <tr><th style="border:1px solid #ccc; padding:6px;"></th>
                <th style="border:1px solid #ccc; padding:6px;">Predicted Neg</th>
                <th style="border:1px solid #ccc; padding:6px;">Predicted Pos</th></tr>
            <tr><td style="border:1px solid #ccc; padding:6px;"><b>Actual Neg</b></td>
                <td style="border:1px solid #ccc; padding:6px; background:{'#e8f5e9' if cm[0,0] > 0 else '#fff'}; text-align:center;">{cm[0,0]}</td>
                <td style="border:1px solid #ccc; padding:6px; background:{'#ffebee' if cm[0,1] > 0 else '#fff'}; text-align:center;">{cm[0,1]}</td></tr>
            <tr><td style="border:1px solid #ccc; padding:6px;"><b>Actual Pos</b></td>
                <td style="border:1px solid #ccc; padding:6px; background:{'#ffebee' if cm[1,0] > 0 else '#fff'}; text-align:center;">{cm[1,0]}</td>
                <td style="border:1px solid #ccc; padding:6px; background:{'#e8f5e9' if cm[1,1] > 0 else '#fff'}; text-align:center;">{cm[1,1]}</td></tr>
        </table>
        """

    metrics = [
        ("AUC",              "auc"),
        ("Accuracy",         "acc"),
        ("Weighted F1",      "f1_w"),
        ("Precision (pos)",  "prec_pos"),
        ("Precision (neg)",  "prec_neg"),
        ("Recall (pos)",     "rec_pos"),
        ("Recall (neg)",     "rec_neg"),
    ]

    rows = ""
    for name, key in metrics:
        b, a = before[key], after[key]
        bold_b = f"<b>{b:.4f}</b>" if b >= a else f"{b:.4f}"
        bold_a = f"<b>{a:.4f}</b>" if a >= b else f"{a:.4f}"
        rows += f"""
        <tr>
            <td style="border:1px solid #ccc; padding:6px;">{name}</td>
            <td style="border:1px solid #ccc; padding:6px; text-align:center;">{bold_b}</td>
            <td style="border:1px solid #ccc; padding:6px; text-align:center;">{bold_a}</td>
            <td style="border:1px solid #ccc; padding:6px; text-align:center;">{delta_html(b, a)}</td>
        </tr>
        """

    html = f"""
    <h3 style="margin-top: 15px;">{dataset_name}</h3>
    <p style="font-size: 13px; color: #666;">
        {before['n_total']} tiles: {before['n_pos']} positive, {before['n_neg']} negative
    </p>
    <table style="border-collapse: collapse; font-size: 13px; margin-bottom: 10px;">
        <tr>
            <th style="border:1px solid #ccc; padding:6px;">Metric</th>
            <th style="border:1px solid #ccc; padding:6px;">Original</th>
            <th style="border:1px solid #ccc; padding:6px;">Fine-tuned</th>
            <th style="border:1px solid #ccc; padding:6px;">Change</th>
        </tr>
        {rows}
    </table>
    {cm_html(before['cm'], 'Original Model')}
    {cm_html(after['cm'], 'Fine-tuned Model')}
    """

    with output_widget:
        display(HTML(html))


# ── Main UI ──────────────────────────────────────────────────────────────────

def create_finetuning_ui():
    """Create the Step 7: Fine-Tune Model interactive panel."""

    project_dir = Path.cwd()
    model_dir = project_dir / "model"
    existing_tiles_dir = project_dir / "sample_data" / "tiles"

    # ── State ──
    state = {
        'model_path': None,
        'new_tiles_dir': None,
        'finetuned_model': None,
        'finetuned_model_path': None,
    }

    # ── UI Elements ──

    # Model selection
    model_chooser = FileChooser(str(model_dir), filter_pattern='*.h5',
                                title='Select model (.h5):')
    model_chooser.show_only_dirs = False

    # Tile folder selection
    tile_chooser = FileChooser(str(project_dir), title='Select new tile folder:')
    tile_chooser.show_only_dirs = True

    # Validation output
    validation_output = widgets.Output()

    # Data mode
    data_mode = widgets.RadioButtons(
        options=['New tiles only', 'Combine with existing tiles'],
        value='New tiles only',
        description='Training data:',
        style={'description_width': '110px'},
        layout=widgets.Layout(width='350px')
    )

    # Epochs
    epoch_slider = widgets.IntSlider(
        value=50, min=10, max=200, step=10,
        description='Max epochs:',
        style={'description_width': '110px'},
        layout=widgets.Layout(width='350px')
    )

    # Stopping mode
    stopping_mode = widgets.RadioButtons(
        options=['Auto (early stopping)', 'Manual (run all epochs)'],
        value='Auto (early stopping)',
        description='Stopping:',
        style={'description_width': '110px'},
        layout=widgets.Layout(width='350px')
    )

    # Early stopping criterion
    stopping_criterion = widgets.Dropdown(
        options=[
            ('AUC (recommended for screening)', 'val_auc'),
            ('Loss (general purpose)', 'val_loss'),
            ('Weighted F1 (balanced precision/recall)', 'val_f1_score_weighted'),
        ],
        value='val_auc',
        description='Stop on:',
        style={'description_width': '110px'},
        layout=widgets.Layout(width='350px')
    )

    # Patience slider (how many epochs without improvement before stopping)
    patience_slider = widgets.IntSlider(
        value=15, min=5, max=30, step=5,
        description='Patience:',
        style={'description_width': '110px'},
        layout=widgets.Layout(width='350px')
    )

    # Container for early stopping options (visible/hidden based on mode)
    early_stop_options = widgets.VBox([stopping_criterion, patience_slider])

    # Show/hide early stopping options based on stopping mode
    def on_stopping_mode_change(change):
        if change['new'] == 'Auto (early stopping)':
            early_stop_options.layout.display = ''
        else:
            early_stop_options.layout.display = 'none'

    stopping_mode.observe(on_stopping_mode_change, names='value')

    # Buttons
    train_btn = widgets.Button(
        description='Start Training',
        button_style='success',
        icon='play',
        layout=widgets.Layout(width='160px'),
        disabled=True
    )

    save_btn = widgets.Button(
        description='Save Model',
        button_style='warning',
        icon='save',
        layout=widgets.Layout(width='160px'),
        disabled=True
    )

    # Progress
    progress_label = widgets.HTML(value='<i>Not started</i>')

    # Output areas
    plot_output = widgets.Output()
    results_output = widgets.Output()
    save_output = widgets.Output()

    # Save location chooser
    save_chooser = FileChooser(str(model_dir), title='Save model to:')
    save_chooser.show_only_dirs = True

    # ── Auto-validation (runs whenever model or tile folder selection changes) ──
    def check_ready(*args):
        """Validate selections and enable/disable the train button."""
        train_btn.disabled = True
        state['model_path'] = None
        state['new_tiles_dir'] = None

        validation_output.clear_output()

        # Check model
        model_path = model_chooser.selected
        if not model_path or not Path(model_path).exists():
            with validation_output:
                display(HTML(
                    '<span style="color: #888;">Select a model file and a tile folder to begin.</span>'
                ))
            return

        # Check tile folder
        tile_path = tile_chooser.selected
        if not tile_path:
            with validation_output:
                display(HTML(
                    '<span style="color: #888;">Select a tile folder to continue.</span>'
                ))
            return

        folder = Path(tile_path)
        pos_dir = folder / "positive"
        neg_dir = folder / "negative"

        # Check for positive/negative subfolders
        if not pos_dir.exists() or not neg_dir.exists():
            missing = []
            if not pos_dir.exists():
                missing.append("positive/")
            if not neg_dir.exists():
                missing.append("negative/")
            with validation_output:
                display(HTML(
                    f'<div style="color: #c62828; background: #ffebee; padding: 10px; '
                    f'border-radius: 4px; margin: 5px 0;">'
                    f'<b>⚠ Invalid folder structure.</b><br>'
                    f'Missing subfolder(s): {", ".join(missing)}<br><br>'
                    f'The selected folder must have this structure:<br>'
                    f'<code style="background: #fff; padding: 2px 6px;">'
                    f'your_folder/<br>'
                    f'├── positive/ &nbsp;(malignant tile images)<br>'
                    f'└── negative/ &nbsp;(benign tile images)</code>'
                    f'</div>'
                ))
            return

        # Check for actual image files in subfolders
        valid, pos_count, neg_count, msg = validate_tile_folder(tile_path)
        if not valid:
            with validation_output:
                display(HTML(
                    f'<div style="color: #c62828; background: #ffebee; padding: 10px; '
                    f'border-radius: 4px; margin: 5px 0;">'
                    f'<b>⚠ No image files found.</b><br>'
                    f'The positive/ and negative/ subfolders must contain image files '
                    f'(.jpg, .jpeg, .png, .bmp, .tif, .tiff).<br>'
                    f'Found: {pos_count} positive images, {neg_count} negative images.'
                    f'</div>'
                ))
            return

        # Valid — show summary
        state['model_path'] = model_path
        state['new_tiles_dir'] = tile_path

        summary = (
            f'<div style="color: #2e7d32; background: #e8f5e9; padding: 10px; '
            f'border-radius: 4px; margin: 5px 0;">'
            f'<b>✓ Ready to train</b><br>'
            f'Model: {Path(model_path).name}<br>'
            f'Tiles: {pos_count} positive, {neg_count} negative ({pos_count + neg_count} total)'
        )

        if data_mode.value == 'Combine with existing tiles':
            ex_valid, ex_pos, ex_neg, _ = validate_tile_folder(existing_tiles_dir)
            if ex_valid:
                summary += (
                    f'<br>Existing tiles: {ex_pos} positive, {ex_neg} negative'
                    f'<br><b>Combined total: {pos_count + ex_pos} positive, '
                    f'{neg_count + ex_neg} negative</b>'
                )
            else:
                summary += (
                    f'<br><span style="color: #e65100;">⚠ Existing tiles not found at '
                    f'{existing_tiles_dir}. Will train on new tiles only.</span>'
                )

        summary += '</div>'

        with validation_output:
            display(HTML(summary))

        train_btn.disabled = False

    # Register auto-validation on selection changes
    tile_chooser.register_callback(check_ready)
    model_chooser.register_callback(check_ready)
    data_mode.observe(lambda change: check_ready(), names='value')

    # Shared stop flag — mutable list so the callback can read it
    # even though the button handler and callback run in different contexts.
    stop_flag = [False]

    # ── Train Handler ──
    def on_train(btn):
        if train_btn.description == 'Stop Training':
            # User clicked stop — set the flag, callback will handle the rest
            stop_flag[0] = True
            train_btn.description = 'Stopping...'
            train_btn.button_style = ''
            train_btn.disabled = True
            return

        # Switch button to Stop mode
        stop_flag[0] = False
        train_btn.description = 'Stop Training'
        train_btn.button_style = 'danger'
        train_btn.icon = 'stop'
        save_btn.disabled = True

        with results_output:
            clear_output()
        with plot_output:
            clear_output()
        with save_output:
            clear_output()

        progress_label.value = '<i>Preparing data...</i>'

        try:
            _run_training(state, data_mode.value, epoch_slider.value,
                          stopping_mode.value, stopping_criterion.value,
                          patience_slider.value,
                          progress_label, plot_output, results_output,
                          save_btn, existing_tiles_dir, stop_flag)
        except Exception as e:
            progress_label.value = f'<b style="color:red;">Error: {e}</b>'
            raise
        finally:
            # Reset button to Start mode
            train_btn.description = 'Start Training'
            train_btn.button_style = 'success'
            train_btn.icon = 'play'
            train_btn.disabled = False

    train_btn.on_click(on_train)

    # ── Save Handler ──
    def on_save(btn):
        with save_output:
            clear_output()

            if state['finetuned_model'] is None:
                print("No fine-tuned model to save. Run training first.")
                return

            # Determine save directory
            save_dir = save_chooser.selected
            if save_dir and Path(save_dir).is_dir():
                save_folder = Path(save_dir)
            else:
                # Default to model/ if no folder selected
                save_folder = model_dir

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"finetuned_model_{timestamp}.h5"
            save_path = save_folder / save_name

            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*HDF5.*")
                state['finetuned_model'].save(str(save_path))
            state['finetuned_model_path'] = save_path

            # Clean up any remaining temp directories
            temp_dir = project_dir / "_finetune_temp"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

            print(f"Model saved to: {save_path}")
            print(f"  Index 0 = Positive (malignant)")
            print(f"  Index 1 = Negative (benign)")
            print(f"\n  To use this model in the notebook, select it in Step 3 (Score Tiles).")
            print(f"\n  Temporary files cleaned up.")

    save_btn.on_click(on_save)

    # ── Layout ──
    header = widgets.HTML(
        '<h3 style="margin-bottom: 5px;">Step 7: Fine-Tune Model</h3>'
        '<p style="color: #666; font-size: 13px; margin-top: 0;">'
        'Extend the model with new labeled tiles. Only the classification head '
        'is trained — the convolutional backbone stays frozen.</p>'
    )

    config_box = widgets.VBox([
        widgets.HBox([
            widgets.VBox([
                widgets.HTML('<b>1. Select model:</b>'),
                model_chooser,
                widgets.HTML('<br><b>2. Select new tile folder:</b>'),
                tile_chooser,
                widgets.HTML('<br><b>3. Select where to save:</b>'),
                save_chooser,
            ], layout=widgets.Layout(width='50%')),
            widgets.VBox([
                widgets.HTML('<b>4. Options:</b>'),
                data_mode,
                epoch_slider,
                stopping_mode,
                early_stop_options,
                widgets.HBox([train_btn, save_btn],
                              layout=widgets.Layout(margin='10px 0 0 0')),
            ], layout=widgets.Layout(width='50%', padding='0 0 0 20px')),
        ]),
        validation_output,
    ])

    progress_box = widgets.VBox([
        widgets.HTML('<hr>'),
        progress_label,
        plot_output,
    ])

    results_box = widgets.VBox([
        widgets.HTML('<hr>'),
        results_output,
        save_output,
    ])

    ui = widgets.VBox([header, config_box, progress_box, results_box])
    display(ui)


# ── Training Logic ───────────────────────────────────────────────────────────

def _run_training(state, data_mode, max_epochs,
                  stop_mode, stop_criterion, stop_patience,
                  progress_label, plot_output, results_output,
                  save_btn, existing_tiles_dir, stop_flag):
    """Run the full fine-tuning pipeline."""

    project_dir = Path.cwd()
    new_tiles_dir = state['new_tiles_dir']
    model_path = state['model_path']

    # ── Prepare training directory ──
    train_dir = project_dir / "_finetune_temp"
    if train_dir.exists():
        shutil.rmtree(train_dir)
    (train_dir / "positive").mkdir(parents=True)
    (train_dir / "negative").mkdir(parents=True)

    progress_label.value = '<i>Copying tiles...</i>'

    # Copy new tiles
    new_counts = copy_tiles_to_dir(new_tiles_dir, train_dir, "new")

    # Optionally combine with existing tiles
    if data_mode == 'Combine with existing tiles' and existing_tiles_dir.exists():
        ex_counts = copy_tiles_to_dir(existing_tiles_dir, train_dir, "existing")

    total_pos = len(list((train_dir / "positive").iterdir()))
    total_neg = len(list((train_dir / "negative").iterdir()))
    total = total_pos + total_neg

    # ── Class weights ──
    weight_pos = total / (2.0 * total_pos)
    weight_neg = total / (2.0 * total_neg)
    class_weights = {0: weight_pos, 1: weight_neg}

    progress_label.value = f'<i>Loading {total} tiles...</i>'

    # ── Load datasets ──
    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(train_dir), validation_split=0.2, subset="training", seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode="categorical"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(train_dir), validation_split=0.2, subset="validation", seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode="categorical"
    )

    # ── Flip labels ──
    def flip_labels(images, labels):
        return images, labels[:, ::-1]

    train_ds = train_ds.map(flip_labels, num_parallel_calls=AUTOTUNE)
    val_ds   = val_ds.map(flip_labels, num_parallel_calls=AUTOTUNE)

    # ── Augment (light: flips + rotation only) ──
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
    ], name="data_augmentation")

    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    )

    # ── Normalize ──
    def preprocess(images, labels):
        images = tf.cast(images, tf.float32)
        images = (images / 127.5) - 1.0
        return images, labels

    train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    val_ds   = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds   = val_ds.prefetch(AUTOTUNE)

    # ── Load model ──
    progress_label.value = '<i>Loading model...</i>'
    model = load_model(str(model_path), compile=False)

    # Freeze backbone, train head only
    model.layers[0].trainable = False
    model.layers[1].trainable = True

    # ── Cosine LR schedule ──
    steps_per_epoch = total // BATCH_SIZE
    total_steps = steps_per_epoch * max_epochs

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=total_steps,
        alpha=1e-6
    )

    weighted_f1 = F1Score(average='weighted', threshold=0.5, name='f1_score_weighted')

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc", multi_label=False),
            tf.keras.metrics.Precision(name="precision_class0", class_id=0),
            tf.keras.metrics.Precision(name="precision_class1", class_id=1),
            tf.keras.metrics.Recall(name="recall_class0", class_id=0),
            tf.keras.metrics.Recall(name="recall_class1", class_id=1),
            weighted_f1
        ]
    )

    # ── Callbacks ──
    live_plot = LivePlotCallback(plot_output, progress_label, stop_flag)
    live_plot.set_total_epochs(max_epochs)

    callbacks = [live_plot]

    # Add early stopping if user selected auto mode
    use_early_stopping = (stop_mode == 'Auto (early stopping)')

    if use_early_stopping:
        # Determine mode: 'min' for loss (lower is better), 'max' for AUC/F1 (higher is better)
        es_mode = 'min' if stop_criterion == 'val_loss' else 'max'

        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=stop_criterion,
                mode=es_mode,
                patience=stop_patience,
                restore_best_weights=True,
                verbose=0
            )
        )

    # Pretty name for the criterion (used in progress label)
    criterion_names = {
        'val_auc': 'AUC',
        'val_loss': 'Loss',
        'val_f1_score_weighted': 'Weighted F1',
    }
    criterion_display = criterion_names.get(stop_criterion, stop_criterion)

    # ── Train ──
    progress_label.value = '<i>Training started...</i>'

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=0           # suppress default output, live_plot handles display
    )

    best_auc = max(history.history['val_auc'])
    final_epoch = len(history.history['loss'])

    # Build stop reason message
    if stop_flag[0]:
        stop_info = f'Stopped by user at epoch {final_epoch}'
    elif final_epoch < max_epochs and use_early_stopping:
        stop_info = f'Early stopping on {criterion_display} (patience={stop_patience})'
    elif use_early_stopping:
        stop_info = f'Reached max epochs (no early stopping triggered)'
    else:
        stop_info = f'Ran all {max_epochs} epochs (manual mode)'

    progress_label.value = (
        f'<b style="color: green;">Training complete.</b>  '
        f'{final_epoch} epochs  |  Best val_AUC: {best_auc:.4f}  |  {stop_info}'
    )

    # Store model in state
    state['finetuned_model'] = model

    # ── Evaluate: before vs after ──
    progress_label.value += '  |  <i>Evaluating...</i>'

    original_model = load_model(str(model_path), compile=False)

    # Evaluate on new tiles
    new_images, new_truths = load_tiles_for_eval(new_tiles_dir, IMG_SIZE)
    before_new = evaluate_model(original_model, new_images, new_truths)
    after_new  = evaluate_model(model, new_images, new_truths)

    display_results(before_new, after_new, "New Tiles — Before vs After", results_output)

    # Evaluate on existing tiles if they exist
    if existing_tiles_dir.exists():
        valid, _, _, _ = validate_tile_folder(existing_tiles_dir)
        if valid:
            ex_images, ex_truths = load_tiles_for_eval(existing_tiles_dir, IMG_SIZE)
            before_ex = evaluate_model(original_model, ex_images, ex_truths)
            after_ex  = evaluate_model(model, ex_images, ex_truths)
            display_results(before_ex, after_ex,
                           "Existing Tiles — Regression Check", results_output)

    progress_label.value = (
        f'<b style="color: green;">Done.</b>  '
        f'{final_epoch} epochs  |  Best val_AUC: {best_auc:.4f}  |  {stop_info}  |  '
        f'Click "Save Model" to keep the fine-tuned model.'
    )

    save_btn.disabled = False

    # ── Clean up temp directory ──
    if train_dir.exists():
        shutil.rmtree(train_dir)