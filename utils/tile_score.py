"""
tile_score.py

Python equivalent of Tile_Scorer_1.04.groovy + _keras_batch_score_qupath.py.
Loads a Teachable Machine Keras model and scores tile images for malignancy.

The pipeline:
    1. Load the trained Keras model (.h5) and class labels
    2. Score every tile image in a directory
    3. Infer ground truth from folder names (positive/negative) if available
    4. Compute ROC curve and summary statistics
    5. Generate ROC figure
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

def load_model(model_path: str) -> tf.keras.Model:
    """
    Load a Teachable Machine Keras model from an .h5 file.

    Args:
        model_path: path to the .h5 model file

    Returns:
        loaded Keras model
    """
    model = tf.keras.models.load_model(str(model_path), compile=False)
    return model


def load_labels() -> list[str]:
    return ["Positive", "Negative"]

def get_image_size_from_model(model: tf.keras.Model) -> int:
    """
    Read the expected input image size from the model's input shape.

    Teachable Machine models typically expect (None, 224, 224, 3).

    Returns:
        image size in pixels (e.g. 224)
    """
    ishape = model.input_shape
    if isinstance(ishape, list):
        ishape = ishape[0]

    h = ishape[1] if len(ishape) > 2 else None
    if h is None:
        return 224

    return int(h)


# ──────────────────────────────────────────────
# Image preprocessing
# ──────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def is_image_file(path: Path) -> bool:
    """Check if a file has a recognized image extension."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def preprocess_tile(path: Path, image_size: int) -> np.ndarray:
    """
    Load and preprocess a tile image for the Teachable Machine model.

    Steps:
        1. Open image and convert to RGB
        2. Resize to (image_size x image_size)
        3. Normalize pixel values from [0, 255] to [-1, 1]
        4. Add batch dimension

    This matches the exact normalization used by Teachable Machine:
        arr = (arr / 127.5) - 1.0

    Args:
        path:       path to the tile image
        image_size: target size (e.g. 224)

    Returns:
        numpy array of shape (1, image_size, image_size, 3), float32
    """
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    arr = np.asarray(img).astype(np.float32)
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)


# ──────────────────────────────────────────────
# Truth label inference
# ──────────────────────────────────────────────

POSITIVE_FOLDER_NAMES = {
    "positive", "pos", "malignant", "cancer",
    "adenoca", "adeno", "tumor", "tumour",
}
NEGATIVE_FOLDER_NAMES = {
    "negative", "neg", "benign", "normal", "nonneoplastic",
}


def infer_truth_from_parent(path: Path) -> str:
    """
    Infer ground truth label from the parent folder name.

    If the tile lives in a folder called "positive" (or "malignant", etc.),
    it is labeled as "positive". If in "negative" (or "benign", etc.),
    it is labeled as "negative". Otherwise returns empty string.

    Args:
        path: path to the tile image

    Returns:
        "positive", "negative", or ""
    """
    parent = path.parent.name.lower()
    if parent in POSITIVE_FOLDER_NAMES:
        return "positive"
    if parent in NEGATIVE_FOLDER_NAMES:
        return "negative"
    return ""


# ──────────────────────────────────────────────
# Identify the malignant probability column
# ──────────────────────────────────────────────

def identify_positive_label(labels: list[str]) -> str:
    """
    Determine which label corresponds to the "malignant" class.

    Logic (mirrors the Groovy Tile Scorer):
        - If only 1 label, use it
        - If 2 labels, pick the one that is NOT benign/negative-sounding
        - If 3+ labels, look for one containing "malig", "cancer", etc.

    Args:
        labels: list of class labels from labels.txt

    Returns:
        the label string to treat as positive/malignant
    """
    if len(labels) == 1:
        return labels[0]

    benign_keywords = {"benign", "normal", "negative", "nonneoplastic"}
    malignant_keywords = {"malig", "cancer", "adeno", "carcin", "tumor", "positive"}

    if len(labels) == 2:
        c0_lower = labels[0].lower()
        c1_lower = labels[1].lower()
        c0_benign = any(k in c0_lower for k in benign_keywords)
        c1_benign = any(k in c1_lower for k in benign_keywords)

        if c0_benign and not c1_benign:
            return labels[1]
        if c1_benign and not c0_benign:
            return labels[0]
        return labels[0]  # default to first

    # 3+ labels: look for malignant-sounding label
    for lab in labels:
        if any(k in lab.lower() for k in malignant_keywords):
            return lab

    return labels[0]


# ──────────────────────────────────────────────
# Score a directory of tiles
# ──────────────────────────────────────────────

def score_tiles(
    image_dir: str,
    model_path: str,
    output_csv: str = None,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Score every tile image in a directory using the Keras model.

    Walks the directory recursively to find all image files.
    If tiles are in positive/negative subfolders, ground truth
    is inferred automatically.

    Args:
        image_dir:          path to the tile directory
        model_path:         path to the Keras .h5 model
        output_csv:         if provided, save results to this CSV path
        progress_callback:  optional callable(scored_count, total_count)

    Returns:
        pandas DataFrame with columns:
            - file:         tile file path
            - truth:        inferred ground truth ("positive"/"negative"/"")
            - pred_label:   predicted class label
            - pred_prob:    probability of predicted class
            - p_{label}:    probability for each class
    """
    image_dir = Path(image_dir)
    labels = load_labels()
    model = load_model(model_path)
    image_size = get_image_size_from_model(model)
    pos_label = identify_positive_label(labels)

    print(f"  Labels: {labels}")
    print(f"  Positive label: {pos_label}")
    print(f"  Image size: {image_size}x{image_size}")

    # Find all image files
    paths = sorted([p for p in image_dir.rglob("*") if p.is_file() and is_image_file(p)])
    total = len(paths)
    print(f"  Found {total} tile images")

    if total == 0:
        raise RuntimeError(f"No image files found in {image_dir}")

    # Score each tile
    rows = []
    for i, p in enumerate(paths):
        x = preprocess_tile(p, image_size)
        probs = model.predict(x, verbose=0)[0]
        probs = np.asarray(probs).astype(float)

        pred_idx = int(np.argmax(probs))
        pred_label = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)
        pred_prob = float(probs[pred_idx])

        row = {
            "file": str(p),
            "truth": infer_truth_from_parent(p),
            "pred_label": pred_label,
            "pred_prob": pred_prob,
        }
        for j, lab in enumerate(labels):
            row[f"p_{lab}"] = float(probs[j]) if j < len(probs) else 0.0

        rows.append(row)

        if progress_callback:
            progress_callback(i + 1, total)

    # Build DataFrame
    prob_cols = [f"p_{lab}" for lab in labels]
    columns = ["file", "truth", "pred_label", "pred_prob"] + prob_cols
    df = pd.DataFrame(rows, columns=columns)

    # Save to CSV if requested
    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"  Saved scores to {output_csv}")

    print(f"  Scored {len(df)} tiles. Done.")
    return df


# ──────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────

def compute_statistics(df: pd.DataFrame, pos_label: str = None) -> dict:
    """
    Compute summary statistics from a scored DataFrame.

    Args:
        df:         DataFrame returned by score_tiles()
        pos_label:  which p_{label} column to use as the malignant score.
                    If None, auto-detected from the column names.

    Returns:
        dict with keys: n_tiles, score_min, score_mean, score_max,
                        has_truth, n_positive, n_negative
    """
    if pos_label is None:
        prob_cols = [c for c in df.columns if c.startswith("p_")]
        labels = [c.replace("p_", "") for c in prob_cols]
        pos_label = identify_positive_label(labels)

    score_col = f"p_{pos_label}"
    scores = df[score_col]

    has_truth = df["truth"].isin(["positive", "negative"]).any()

    return {
        "n_tiles": len(df),
        "score_min": float(scores.min()),
        "score_mean": float(scores.mean()),
        "score_max": float(scores.max()),
        "has_truth": has_truth,
        "n_positive": int((df["truth"] == "positive").sum()),
        "n_negative": int((df["truth"] == "negative").sum()),
        "positive_label": pos_label,
        "score_column": score_col,
    }


# ──────────────────────────────────────────────
# ROC computation
# ──────────────────────────────────────────────

def compute_roc(df: pd.DataFrame, pos_label: str = None) -> pd.DataFrame:
    """
    Compute ROC curve by sweeping thresholds across the malignant
    probability scores.

    This replicates the Groovy Tile Scorer ROC logic (lines 389-420):
        - Collect all unique scores as thresholds
        - At each threshold, compute TP/FP/TN/FN
        - Derive TPR, FPR, precision, recall, F1

    Only works if the DataFrame has truth labels ("positive"/"negative").
    Returns an empty DataFrame if no truth labels are found.

    Args:
        df:         DataFrame returned by score_tiles()
        pos_label:  which label to treat as positive

    Returns:
        DataFrame with columns: threshold, tpr, fpr, precision, recall, f1
    """
    if pos_label is None:
        prob_cols = [c for c in df.columns if c.startswith("p_")]
        labels = [c.replace("p_", "") for c in prob_cols]
        pos_label = identify_positive_label(labels)

    score_col = f"p_{pos_label}"

    # Filter to rows with truth labels
    labeled = df[df["truth"].isin(["positive", "negative"])].copy()
    if labeled.empty:
        print("  No truth labels found. ROC skipped.")
        return pd.DataFrame(columns=["threshold", "tpr", "fpr", "precision", "recall", "f1"])

    scores = labeled[score_col].values
    truths = (labeled["truth"] == "positive").values.astype(int)

    # Build threshold list (all unique scores + boundary values)
    thresholds = np.unique(scores)
    thresholds = np.concatenate([[1.0 + 1e-9], thresholds[::-1], [-1e-9]])

    roc_rows = []
    for thr in thresholds:
        pred = (scores >= thr).astype(int)

        tp = int(((pred == 1) & (truths == 1)).sum())
        fp = int(((pred == 1) & (truths == 0)).sum())
        tn = int(((pred == 0) & (truths == 0)).sum())
        fn = int(((pred == 0) & (truths == 1)).sum())

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        roc_rows.append({
            "threshold": float(thr),
            "tpr": tpr,
            "fpr": fpr,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    return pd.DataFrame(roc_rows)


def compute_auc(roc_df: pd.DataFrame) -> float:
    """
    Compute Area Under the ROC Curve using trapezoidal integration.

    Args:
        roc_df: DataFrame from compute_roc() with 'fpr' and 'tpr' columns

    Returns:
        AUC value (float between 0 and 1)
    """
    if roc_df.empty:
        return 0.0

    roc_sorted = roc_df.sort_values("fpr").drop_duplicates(subset=["fpr", "tpr"])
    return float(np.trapz(roc_sorted["tpr"].values, roc_sorted["fpr"].values))


# ──────────────────────────────────────────────
# ROC figure
# ──────────────────────────────────────────────

def plot_roc(
    roc_df: pd.DataFrame,
    auc: float = None,
    figsize: tuple = (5, 5),
    title: str = "ROC Curve",
):
    """
    Generate a publication-quality ROC curve figure.

    Args:
        roc_df:     DataFrame from compute_roc()
        auc:        pre-computed AUC value (computed automatically if None)
        figsize:    matplotlib figure size
        title:      plot title

    Returns:
        matplotlib figure
    """

    import matplotlib.pyplot as plt

    if auc is None:
        auc = compute_auc(roc_df)

    roc_sorted = roc_df.sort_values("fpr").drop_duplicates(subset=["fpr", "tpr"])

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(
        roc_sorted["fpr"], roc_sorted["tpr"],
        linewidth=2,
        label=f"ROC curve (AUC = {auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="orange", label="Random classifier")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig




#-----------------------------------------------------------------------------------------------
# ──────────────────────────────────────────────
# Heatmap overlay: malignant/benign color map on slide
# ──────────────────────────────────────────────

def parse_coordinates_from_filename(filename: str) -> tuple:
    """
    Extract x, y coordinates from a tile filename.

    Expected format: {name}_x{x}_y{y}__{serial}.jpg

    Args:
        filename: tile filename (not full path, just the name)

    Returns:
        (x, y) as integers, or None if parsing fails
    """
    import re
    match = re.search(r'_x(\d+)_y(\d+)__', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def create_heatmap_overlay(
    slide_path: str,
    scores_df,
    score_column: str = "p_Positive",
    max_size: int = 2000,
    alpha: float = 0.4,
    figsize: tuple = (16, 10),
    title: str = "Malignancy Heatmap Overlay",
    threshold: float = 0.5,
):
    """
    Overlay color-coded tiles on a whole slide thumbnail.

    Each tile is colored based on its malignancy score:
        - Red   = high malignancy probability
        - Green = low malignancy probability (benign)

    The color intensity reflects confidence — a tile with
    p_Positive=0.99 is deep red, while p_Positive=0.6 is
    a lighter red. Similarly, p_Positive=0.01 is deep green
    and p_Positive=0.4 is lighter green.

    Args:
        slide_path:     path to the whole slide image
        scores_df:      DataFrame from score_tiles(), or path to tile_scores.csv.
                        Must have a 'file' column (tile paths with x,y in filename)
                        and a score column (default 'p_Positive').
        score_column:   which column to use for coloring (default 'p_Positive')
        max_size:       max thumbnail dimension in pixels
        alpha:          transparency of the overlay (0.0 = invisible, 1.0 = opaque)
        figsize:        matplotlib figure size
        title:          plot title
        threshold:      score threshold for the legend (default 0.5)

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors
    from pathlib import Path
    from utils.tile_export import get_slide_thumbnail

    # Load CSV if a path was passed instead of a DataFrame
    if isinstance(scores_df, (str, Path)):
        scores_df = pd.read_csv(scores_df)

    if score_column not in scores_df.columns:
        raise ValueError(f"Column '{score_column}' not found in DataFrame. Available: {list(scores_df.columns)}")

    # Get slide thumbnail
    thumbnail, scale = get_slide_thumbnail(slide_path, max_size)

    # Parse coordinates from filenames
    tiles_with_coords = []
    for _, row in scores_df.iterrows():
        filepath = row["file"]
        filename = Path(filepath).name
        coords = parse_coordinates_from_filename(filename)

        if coords is not None:
            x, y = coords
            score = float(row[score_column])
            tiles_with_coords.append({"x": x, "y": y, "score": score})

    if not tiles_with_coords:
        print("  ⚠️ No tiles with coordinate information found in filenames.")
        print("     Expected format: {name}_x{x}_y{y}__{serial}.jpg")
        return None

    # Estimate tile extent from the first two tiles in the same row
    # (distance between consecutive x values)
    xs = sorted(set(t["x"] for t in tiles_with_coords))
    if len(xs) >= 2:
        # Find the smallest gap between consecutive x values
        gaps = [xs[i+1] - xs[i] for i in range(len(xs)-1) if xs[i+1] - xs[i] > 0]
        tile_extent = min(gaps) if gaps else int(512 * 4)  # fallback
    else:
        tile_extent = int(512 * 4)  # fallback: 512 tiles at ~4x downsample

    # Create the figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(thumbnail)

    # Draw colored rectangles for each tile
    for t in tiles_with_coords:
        score = t["score"]

        # Color: red for malignant, green for benign
        # Intensity scales with confidence
        if score >= threshold:
            # Malignant: red, more intense as score approaches 1.0
            intensity = (score - threshold) / (1.0 - threshold) if threshold < 1.0 else 1.0
            color = (1.0, 0.0, 0.0, alpha * (0.3 + 0.7 * intensity))
        else:
            # Benign: green, more intense as score approaches 0.0
            intensity = (threshold - score) / threshold if threshold > 0.0 else 1.0
            color = (0.0, 0.6, 0.0, alpha * (0.3 + 0.7 * intensity))

        # Convert level-0 coordinates to thumbnail coordinates
        rx = t["x"] * scale
        ry = t["y"] * scale
        rw = tile_extent * scale
        rh = tile_extent * scale

        rect = patches.Rectangle(
            (rx, ry), rw, rh,
            facecolor=color,
            edgecolor=None,
            linewidth=0,
        )
        ax.add_patch(rect)

    # Stats for the title
    n_malignant = sum(1 for t in tiles_with_coords if t["score"] >= threshold)
    n_benign = sum(1 for t in tiles_with_coords if t["score"] < threshold)
    n_total = len(tiles_with_coords)

    ax.set_title(
        f"{title}\n"
        f"{n_total} tiles: {n_malignant} malignant (red), {n_benign} benign (green) — threshold={threshold}",
        fontsize=13,
    )
    ax.axis("off")

    # Add a colorbar-style legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(1.0, 0.0, 0.0, 0.7), label=f'Malignant (p ≥ {threshold})'),
        Patch(facecolor=(0.0, 0.6, 0.0, 0.7), label=f'Benign (p < {threshold})'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    return fig