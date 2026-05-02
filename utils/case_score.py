"""
case_score.py

Case-level classification using top-K tile thresholding.

Given a set of scored slides (each with a tile_scores.csv),
this module determines whether each case (slide) is positive
or negative based on how many of its tiles exceed a malignancy
threshold.

The key parameter K controls the cutoff:
    - If a slide has >= K malignant tiles, it is called positive.
    - K can be expressed as an absolute count or a percentage
      of the slide's total tiles.
"""

from pathlib import Path
import pandas as pd


def load_case_data(
    wsi_dir: str,
    csv_filename: str = "batch_tile_scores.csv",
    score_column: str = "p_Positive",
) -> list[dict]:
    """
    Load tile scores from a folder structure where slides are
    organized into positive/ and negative/ subfolders.

    Expected structure:
        wsi_dir/
        ├── positive/
        │   ├── slide_1/
        │   │   └── batch_tile_scores.csv
        │   ├── slide_2/
        │   │   └── batch_tile_scores.csv
        ├── negative/
        │   ├── slide_3/
        │   │   └── batch_tile_scores.csv

    Args:
        wsi_dir:        path to the WSI directory
        csv_filename:   name of the scores CSV in each slide folder
        score_column:   which column contains the malignancy probability

    Returns:
        list of dicts, each with:
            - name:         slide folder name
            - truth:        "positive" or "negative"
            - scores:       list of float (malignancy probabilities)
            - n_tiles:      total number of tiles
    """
    wsi_dir = Path(wsi_dir)
    cases = []

    for truth_label in ["positive", "negative"]:
        truth_dir = wsi_dir / truth_label
        if not truth_dir.exists():
            continue

        for slide_folder in sorted(truth_dir.iterdir()):
            if not slide_folder.is_dir():
                continue

            csv_path = slide_folder / csv_filename
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            if score_column not in df.columns:
                continue

            scores = df[score_column].tolist()

            cases.append({
                "name": slide_folder.name,
                "truth": truth_label,
                "scores": scores,
                "n_tiles": len(scores),
            })

    return cases


def classify_cases(
    cases: list[dict],
    tile_threshold: float = 0.5,
    k_value: float = 1,
    k_mode: str = "count",
) -> list[dict]:
    """
    Classify each case as positive or negative based on the
    number of malignant tiles exceeding a threshold.

    Args:
        cases:          list of case dicts from load_case_data()
        tile_threshold: probability above which a tile is considered malignant
        k_value:        the top-K cutoff
        k_mode:         "count" (absolute number) or "percent" (percentage of total tiles)

    Returns:
        list of dicts, each with:
            - name, truth, n_tiles (from input)
            - n_malignant:  number of tiles above tile_threshold
            - k_threshold:  the actual count threshold used for this case
            - pred:         "positive" or "negative"
            - correct:      True if pred == truth
    """
    results = []

    for case in cases:
        n_malignant = sum(1 for s in case["scores"] if s >= tile_threshold)

        if k_mode == "percent":
            k_threshold = max(1, int(round(case["n_tiles"] * k_value / 100.0)))
        else:
            k_threshold = max(1, int(k_value))

        pred = "positive" if n_malignant >= k_threshold else "negative"

        results.append({
            "name": case["name"],
            "truth": case["truth"],
            "n_tiles": case["n_tiles"],
            "n_malignant": n_malignant,
            "k_threshold": k_threshold,
            "pred": pred,
            "correct": pred == case["truth"],
        })

    return results


def compute_confusion_matrix(results: list[dict]) -> dict:
    """
    Compute a confusion matrix from classification results.

    Returns:
        dict with: tp, fp, tn, fn, sensitivity, specificity, accuracy,
                   total, n_correct
    """
    tp = sum(1 for r in results if r["truth"] == "positive" and r["pred"] == "positive")
    fp = sum(1 for r in results if r["truth"] == "negative" and r["pred"] == "positive")
    tn = sum(1 for r in results if r["truth"] == "negative" and r["pred"] == "negative")
    fn = sum(1 for r in results if r["truth"] == "positive" and r["pred"] == "negative")

    total = tp + fp + tn + fn
    n_correct = tp + tn

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = n_correct / total if total > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "total": total,
        "n_correct": n_correct,
    }


def find_optimal_k(
    cases: list[dict],
    tile_threshold: float = 0.5,
    k_mode: str = "count",
) -> dict:
    """
    Find the maximum K value that still achieves 100% sensitivity.

    As K increases, fewer cases are called positive. At some point,
    a true positive case gets missed and sensitivity drops below 1.0.
    The optimal K is the largest value just before that happens —
    it gives the best specificity while maintaining perfect sensitivity.

    Args:
        cases:          list of case dicts from load_case_data()
        tile_threshold: probability above which a tile is malignant
        k_mode:         "count" or "percent"

    Returns:
        dict with:
            - k_optimal:      the best K value found
            - sensitivity:    sensitivity at that K (should be 1.0)
            - specificity:    specificity at that K
            - cm:             the confusion matrix dict at that K
            - found_perfect:  True if both sensitivity and specificity are 1.0
    """
    if k_mode == "count":
        max_k = max(c["n_tiles"] for c in cases)
        k_values = range(1, max_k + 1)
    else:
        k_values = [v / 10.0 for v in range(1, 1001)]  # 0.1 to 100.0

    best = {
        "k_optimal": 1,
        "sensitivity": 0.0,
        "specificity": 0.0,
        "cm": None,
        "found_perfect": False,
    }

    for k in k_values:
        results = classify_cases(
            cases,
            tile_threshold=tile_threshold,
            k_value=k,
            k_mode=k_mode,
        )
        cm = compute_confusion_matrix(results)

        if cm["sensitivity"] == 1.0:
            # This K still has perfect sensitivity — keep going
            best = {
                "k_optimal": k,
                "sensitivity": cm["sensitivity"],
                "specificity": cm["specificity"],
                "cm": cm,
                "found_perfect": cm["specificity"] == 1.0,
            }
        else:
            # Sensitivity dropped — the previous K was the max
            break

    # Fallback if we never hit sensitivity=1.0
    if best["cm"] is None:
        results = classify_cases(cases, tile_threshold=tile_threshold, k_value=1, k_mode=k_mode)
        cm = compute_confusion_matrix(results)
        best = {
            "k_optimal": 1,
            "sensitivity": cm["sensitivity"],
            "specificity": cm["specificity"],
            "cm": cm,
            "found_perfect": False,
        }

    return best