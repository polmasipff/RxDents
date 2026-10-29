# Dental OPG Impact Detection

Exploratory computer vision project for detecting and localising **impacted teeth** in panoramic dental X-rays (orthopantomograms, or OPGs). The project follows the complete learning path from inspecting COCO-style annotations and training image-level classifiers to evaluating explainability methods and building a Faster R-CNN object detector.

> **Status:** research and learning prototype. This repository is not a medical device and must not be used for clinical diagnosis or treatment decisions.

## Project goals

The project investigates three progressively more useful questions:

1. Can a pretrained CNN determine whether an OPG contains at least one impacted tooth?
2. Does the classifier focus on the actual abnormal tooth?
3. Can an object detector directly locate each impacted tooth with a bounding box?

The current main task is the third one: **single-class object detection of impacted teeth**.

## Dataset

The experiments use the public [DENTEX dataset](https://github.com/ibrahimethemhamamci/DENTEX), introduced for the MICCAI 2023 Dental Enumeration and Diagnosis on Panoramic X-rays Challenge. The data can be downloaded from [Zenodo](https://zenodo.org/records/7812323).

DENTEX contains hierarchical COCO-style annotations for:

- tooth quadrant;
- FDI tooth enumeration;
- diagnosis: caries, deep caries, periapical lesion and impacted tooth.

Only the `impacted` diagnosis is used by the current detector.

The working split contains:

| Split | OPGs | OPGs with impactations | Impacted-tooth boxes |
|---|---:|---:|---:|
| Training | 705 | 254 | 604 |
| Validation | 50 | 16 | 40 |

The dataset is **not included in this repository**. DENTEX data remain subject to their original licence and attribution requirements.

## Expected data layout

After downloading and extracting DENTEX, arrange the files as follows:

```text
data/
└── raw/
    └── dentex/
        ├── training_data/
        │   └── quadrant-enumeration-disease/
        │       ├── train_quadrant_enumeration_disease.json
        │       └── xrays/
        │           ├── train_0.png
        │           └── ...
        ├── validation_triple.json
        └── validation_data/
            └── quadrant_enumeration_disease/
                └── xrays/
                    ├── val_0.png
                    └── ...
```

The `data/` directory is ignored by Git.

## Repository structure

```text
.
├── 1_explore.ipynb
├── 2_exploring_pathology.ipynb
├── training_101.ipynb
├── training_2.ipynb
├── training_fine_tuning.ipynb
├── gradcam_comparison.ipynb
├── impact_detector_01.ipynb
├── impact_detector_02.ipynb
├── impact_detector_03_augm.ipynb
├── proba collage amb anotacions.png
├── requirements.txt
└── results/
    ├── average_precision_by_iou.csv
    ├── baseline_config.json
    ├── baseline_summary.csv
    ├── false_negatives.csv
    ├── false_positives.csv
    ├── matched_detections.csv
    └── per_image_detection_results.csv
```

Model checkpoints are written to `models/`, which is also ignored by Git.

## Notebook guide

The notebooks preserve the development and learning history of the project. They are best read in this order:

| Order | Notebook | Purpose |
|---:|---|---|
| 1 | `1_explore.ipynb` | Inspect images, JSON keys, hierarchical labels and bounding boxes. |
| 2 | `2_exploring_pathology.ipynb` | Convert tooth annotations into image-level pathology indicators and inspect label distributions. |
| 3 | `training_101.ipynb` | First transfer-learning experiment with ResNet18. |
| 4 | `training_2.ipynb` | Reproducible image-level impacted/not-impacted classifier and extended evaluation. |
| 5 | `training_fine_tuning.ipynb` | Compare feature extraction with partial fine-tuning. |
| 6 | `gradcam_comparison.ipynb` | Evaluate whether Grad-CAM highlights the annotated pathology boxes. |
| 7 | `impact_detector_01.ipynb` | First Faster R-CNN impacted-tooth detector. |
| 8 | `impact_detector_02.ipynb` | Detection baseline with threshold analysis, error galleries, IoU and AP evaluation. |
| 9 | `impact_detector_03_augm.ipynb` | Bounding-box-safe augmentation, model comparison and hard-negative/data-quality analysis. |

For the current object-detection pipeline, begin with `impact_detector_02.ipynb`, followed by `impact_detector_03_augm.ipynb`.

## Installation

Use a Python version supported by the selected PyTorch build. The current project environment has been run with Python 3.14 and a CUDA-enabled PyTorch build. GPU execution is strongly recommended for training the detector.

### Windows / PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name rxdental --display-name "Python (rxDental)"
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name rxdental --display-name "Python (rxDental)"
```

For an NVIDIA GPU, install the PyTorch build matching the local CUDA setup using the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/), then install the remaining dependencies.

Open the repository in VS Code, select the `Python (rxDental)` kernel and run the notebooks in order.

## Methods

### 1. Image-level classification

The initial experiments use an ImageNet-pretrained ResNet18. Its final layer is replaced with a two-class head:

```text
no impacted tooth / at least one impacted tooth
```

Feature extraction and partial fine-tuning are compared. Class weights are used to reduce the effect of label imbalance. Confusion matrices, precision, recall, F1, ROC-AUC and average precision are reported.

This stage provides a useful baseline but cannot indicate which tooth is abnormal.

### 2. Grad-CAM localisation analysis

Grad-CAM heatmaps are compared with the ground-truth bounding boxes using:

- pointing accuracy;
- energy inside the annotated box;
- concentration relative to box area.

The low localisation scores showed that a correct image-level prediction does not guarantee that the classifier is looking at the impacted tooth. This motivated the transition to supervised object detection.

### 3. Faster R-CNN object detection

The main detector is a COCO-pretrained Faster R-CNN with a ResNet50-FPN backbone. The classification head is replaced for two classes:

```text
background / impacted tooth
```

Main training configuration:

- eight epochs;
- SGD optimiser;
- initial learning rate `0.005`;
- momentum `0.9`;
- weight decay `0.0005`;
- StepLR schedule;
- fixed random seed `42`;
- GPU execution when CUDA is available.

The augmented experiment applies transformations consistently to both the OPG and its boxes:

- horizontal flip;
- mild brightness adjustment;
- mild contrast adjustment.

Validation images are never augmented.

## Current results

### Faster R-CNN comparison

The following results use the 50-image validation split and an IoU threshold of `0.50`.

| Model | Score threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Baseline | 0.30 | 0.679 | 0.900 | 0.774 |
| Baseline | 0.50 | 0.750 | 0.900 | 0.818 |
| Baseline | 0.70 | 0.778 | 0.875 | **0.824** |
| Augmented | 0.30 | 0.667 | 0.900 | 0.766 |
| Augmented | 0.50 | 0.700 | 0.875 | 0.778 |
| Augmented | 0.70 | 0.739 | 0.850 | 0.791 |

| Model | AP50 | mAP50–95 |
|---|---:|---:|
| Baseline | 0.730 | 0.468 |
| Augmented | **0.736** | **0.482** |

The augmented detector produces slightly better ranking and strict localisation metrics, while the baseline currently gives the best operating-point F1.

### Metric interpretation

- **IoU** measures the overlap between a predicted box and its reference box.
- **AP50** is the area under the precision-recall curve when a detection is correct at `IoU >= 0.50`.
- **mAP50–95** averages AP at IoU thresholds from `0.50` to `0.95`, and therefore rewards more precise localisation.
- **Score threshold** controls which predictions are retained at inference time; it changes the precision-recall trade-off but does not retrain the model.

Because the validation set contains only 40 impacted-tooth boxes, these values are exploratory and should not be interpreted as clinical performance estimates.

## Error analysis and annotation quality

The error analysis initially suggested that the detector often learns a shortcut:

```text
posterior third-molar region -> impacted tooth
```

rather than fully distinguishing an impacted tooth from an included, unerupted or normally positioned third molar.

High-confidence apparent false positives from the training set are therefore being reviewed by a dentist. The review has identified three distinct groups:

1. genuine hard negatives, including non-impacted included teeth;
2. uncertain cases requiring age or additional clinical context;
3. likely true impactations missing from the supplied annotations.

Images containing missing or uncertain labels are not treated as negative training examples. Mixed images must be reviewed box by box before their annotations can be corrected. This human-in-the-loop audit is ongoing and will precede hard-negative fine-tuning.

## Generated result files

| File | Contents |
|---|---|
| `baseline_config.json` | Reproducibility and evaluation configuration. |
| `baseline_summary.csv` | Main detector metrics. |
| `average_precision_by_iou.csv` | AP from IoU 0.50 through 0.95. |
| `matched_detections.csv` | Correctly matched predictions and their IoU values. |
| `false_positives.csv` | Unmatched predictions and error reason. |
| `false_negatives.csv` | Missed annotations and best available proposal. |
| `per_image_detection_results.csv` | TP, FP and FN counts for each validation image. |

## Next steps

- Complete expert review at prediction-box level.
- Correct confirmed missing annotations.
- Fine-tune with verified hard negatives using a lower learning rate.
- Reserve an independent test set or use cross-validation for less biased evaluation.
- Extend the detector to caries, deep caries and periapical lesions.
- Evaluate domain shift before any experiment with private clinical OPGs.
- Refactor the exploratory notebooks into reusable training and evaluation modules.

## Reproducibility and data privacy

- Raw data, model weights and local virtual environments are excluded through `.gitignore`.
- No private clinic radiographs or patient information should be committed.
- Results produced from private clinical data must be de-identified and reviewed before sharing.
- Model checkpoints can be regenerated by running the corresponding notebooks.

## Acknowledgements

This project uses the DENTEX benchmark:

> Hamamci et al., *DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays*, MICCAI 2023.

Please cite the original dataset and associated publications when reusing DENTEX data or annotations.

## Disclaimer

This repository documents an educational research prototype. The models have not undergone external clinical validation, regulatory assessment or prospective evaluation. Predictions must not be interpreted as diagnoses and must not replace assessment by a qualified dental professional.
