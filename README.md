# CRC Screening Tool

**Colorectal Carcinoma Screening Pipeline for Digital Pathology**

A self-contained Jupyter Notebook pipeline for screening colorectal adenocarcinoma from whole slide images (WSI). Built for deployment in low-resource settings, the tool uses a MobileNet-based CNN validated on a Kenyan cohort from Aga Khan University Hospital (Nairobi) and Tenwek Hospital (Bomet).

---

## Pipeline Overview

| Step | Description | Input | Output |
|------|-------------|-------|--------|
| 1. **View Slide** | Inspect the whole slide image before processing | `.svs` file | Visual inspection |
| 2. **Export Tiles** | Divide the WSI into 512×512 tiles at ~10× magnification | `.svs` file | JPEG tile images |
| 3. **Score Tiles** | Run each tile through the trained model | Tile folder + `.h5` model | `tile_scores.csv` |
| 4. **Evaluate** | Compute ROC curve, AUC, and summary statistics | `tile_scores.csv` | ROC figure + stats |
| 5. **Click-to-Predict** | Click on the slide to predict any region interactively | `.svs` file + `.h5` model | Live prediction |
| 6. **Heatmap Overlay** | Visualize tile predictions on the slide with optional spatial filtering | `.svs` file + `tile_scores.csv` | Heatmap figure |
| 7. **Case-Level Analysis** | Classify cases using top-K tile thresholding | Scored slide folders | Confusion matrix |
| 8. **Fine-Tune Model** | Extend the model with new labeled tiles | Tile folder + `.h5` model | Fine-tuned `.h5` model |

Steps 1–4 form the core pipeline. Step 5 provides interactive exploration. Step 6 overlays tile-level predictions on the slide as a color-coded heatmap, with an optional morphological opening filter that removes isolated false positive tiles. Step 7 performs case-level classification. Step 8 allows end users to improve the model with locally accumulated cases.

---

## Quick Start

### Requirements

- Python 3.10 or 3.11
- macOS or Windows

### Setup

**macOS:**
```bash
git clone https://github.com/ulbalis/CRC-Screening-Tool.git
cd CRC-Screening-Tool
bash setup.sh
```

**Windows:**
```
git clone https://github.com/ulbalis/CRC-Screening-Tool.git
cd CRC-Screening-Tool
setup.bat
```

The setup script will:
1. Create a virtual environment
2. Install all required packages
3. Register the Jupyter kernel
4. Launch the notebook in your browser

### Sample Data

The sample WSI slides and tile images are hosted separately due to file size. Download them from:

**[OSF Project Link]** *(link to be added upon publication)*

After downloading, place the data in the `sample_data/` directory:
```
sample_data/
├── tiles/
│   ├── positive/
│   └── negative/
└── wsi/
    ├── positive/
    │   └── {slide_name}/
    │       ├── {slide_name}.svs
    │       └── tiles/
    └── negative/
        └── {slide_name}/
            ├── {slide_name}.svs
            └── tiles/
```

---

## Project Structure

```
CRC-Screening-Tool/
├── CRC_Screening_Tool.ipynb    # Main notebook (Steps 1–8)
├── model/
│   ├── model.h5                # Pre-trained MobileNet model
│   └── labels.txt              # Class labels (Positive, Negative)
├── sample_data/                # Sample data (downloaded separately)
├── setup.sh                    # macOS setup script
├── setup.bat                   # Windows setup script
└── utils/
    ├── __init__.py
    ├── batch_process.py        # Headless multi-slide processing
    ├── case_score.py           # Case-level top-K classification
    ├── finetune_ui.py          # Step 8: Fine-tuning UI
    ├── heatmap_ui.py           # Step 6: Heatmap overlay UI
    ├── tile_export.py          # Slide reading and tile extraction
    ├── tile_opening.py         # Morphological opening (erosion/dilation)
    ├── tile_score.py           # Model inference and ROC computation
    ├── ui_dashboards.py        # Steps 1–5, 7 interactive widgets
    └── viewer.py               # OpenSlide viewport reader
```

---

## Model Details

- **Architecture:** MobileNet (via Google Teachable Machine)
- **Input:** 224×224 px tiles, normalized to [-1, 1]
- **Output:** 2-class softmax (index 0 = Positive/malignant, index 1 = Negative/benign)
- **Framework:** TensorFlow 2.15.1 / Keras 2.15.0
- **Training data:** Kenyan colorectal adenocarcinoma cohort (Aga Khan University Hospital, Nairobi; Tenwek Hospital, Bomet)

---

## Heatmap Overlay & Morphological Opening

Step 6 visualizes tile-level malignancy predictions directly on the whole slide image. Each tile is drawn as a semi-transparent colored rectangle — red for malignant, green for benign — with intensity proportional to the model's confidence.

An optional morphological opening filter can be applied to remove isolated false positive tiles:
- **Erosion:** tiles with fewer than a specified number of positive neighbors are removed
- **Dilation:** edges of surviving malignant clusters are restored
- Applied twice by default (2× erosion followed by 2× dilation)

This spatial filtering exploits the fact that true adenocarcinoma typically occupies multiple adjacent tiles, while false positives from inflammation, staining artifacts, or tissue folds tend to appear as isolated single-tile events.

---

## Fine-Tuning

Step 8 in the notebook allows end users to extend the model with new labeled tiles. The fine-tuning process:
- Freezes the convolutional backbone (only the classification head is retrained)
- Uses AdamW optimizer with cosine learning rate decay
- Applies class weights to handle imbalanced datasets
- Supports auto early stopping (AUC, Loss, or F1 criteria)
- Displays live training curves and before/after comparison

---

## Citation

*(To be added upon publication)*

---

## License

*(To be determined)*

---

## Authors

- **Dr. Ulysses Balis** — Department of Pathology, University of Michigan
- **Ye Chan Kim** — Center for Global Health Equity, University of Michigan

*Developed at the University of Michigan in collaboration with Aga Khan University Hospital (Nairobi) and Tenwek Hospital (Bomet), Kenya.*
