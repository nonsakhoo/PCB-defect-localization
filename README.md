
# PCB defect localization (MATLAB script)

## NOTE: The associated paper is currently under review. This script version is provided exclusively to reviewers during the review stage and is not the polished official public release.

This folder contains the runnable MATLAB script **v0_14_eval2_release1.m** for training-free PCB defect localization using a defect-free reference image (template), plus optional evaluation against VOC-style XML annotations.

## Prerequisites

- MATLAB (tested as a script with local functions).
- Recommended: Image Processing Toolbox (used for SSIM, morphology, and registration/alignment).
- Optional: Parallel Computing Toolbox (only if you enable `cfg.templateSelection.parallel.enable = true`).

## Dataset (required)

1. Download the dataset:
	 - https://robotics.pkusz.edu.cn/resources/dataset/
2. Unzip and place it at this **relative** path (repo root):
	 - `dev/PCB_DATASET/`

Expected dataset structure (GT XML is optional, but required for evaluation metrics):

- `dev/PCB_DATASET/images（Defective）/<DefectClass>/*`
- `dev/PCB_DATASET/PCB_USED（Defect-free）/*`
- `dev/PCB_DATASET/Annotations/<DefectClass>/<imageBase>.xml`

If your filesystem uses ASCII parentheses, these alternatives are also accepted:

- `images(Defective)`
- `PCB_USED(Defect-free)`

The script also tolerates common dataset-root suffix variants like `PCB_DATASET(1)` / `PCB_DATASET（1）`.

## How to run

1. Open MATLAB.
2. Set MATLAB **Current Folder** to `<repo>/dev`.
3. Run:

```matlab
v0_14_eval2_release1
```

If the dataset is not found, the script throws an error explaining the expected location.

## What the script does

Per defective image:

- Select a best-matching defect-free template (SSIM-based, with optional cheap preselect + caching)
- Optionally align/register the image pair
- Compute a difference score map (SSIM-map and/or absolute difference)
- Threshold + morphological cleanup to form a defect mask
- Extract connected components and output bounding boxes + confidence scores

Optionally:

- Evaluate predicted boxes against ground-truth VOC XML (`Annotations/…`) via IoU matching
- Export a compact evaluation bundle (MAT/JSON/CSV)
- Generate summary figures (can take time depending on CPU)

## Outputs

Outputs are written under `dev/outputs/` into a run folder named like:

- `dev/outputs/outputs_v0_14_eval2_release1/`

Depending on flags in `defaultConfig()` you may get:

- Figures (`.png`, and optionally `.fig`)
- `evaluation_report.mat` (most complete)
- `evaluation_report.json` (compact summary)
- CSV tables (per-image and per-detection summaries, PR/FROC curves, AP sweeps)

The export metadata is designed to avoid leaking absolute local paths (paths are sanitized).

## Configuration

All settings are in the `defaultConfig()` function inside **v0_14_eval2_release1.m**.

Common knobs:

- Run length: `cfg.run.maxImages = 10;` (set to `[]` to run all images)
- Visualization: `cfg.output.showFigures`, `cfg.output.saveFigures`
- Speed/accuracy tradeoffs:
	- `cfg.resize.*` (processing resolution)
	- `cfg.templateSelection.*` (template selection presets + caching)
	- `cfg.registration.enable` (alignment)
- Evaluation:
	- `cfg.eval.enable`
	- `cfg.eval.iouThreshold`
- Export bundle:
	- `cfg.export.enable`

## Troubleshooting

- **Evaluation is empty / missing**: ensure `dev/PCB_DATASET/Annotations/<DefectClass>/<imageBase>.xml` exists. The pipeline can still run without GT, but metrics won’t be computed.
- **Long runtime during figure generation**: the script prints an info message; this step can be slow on CPU. Disable via `cfg.paperFigs.enable = false` if you only need detections/evaluation.
- **Folder-name mismatch (parentheses)**: the script accepts both full-width `（ ）` and ASCII `( )` for the key dataset subfolders.

## Citation / status

If you use this code in academic work, cite the associated paper. If you don’t see a citation entry in the repository yet, check the header comment in **v0_14_eval2_release1.m** for the latest notes.

