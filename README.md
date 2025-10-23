
# Deep Learning Segmentation Project (Teaching Template)

This is a minimal, **teaching-friendly** PyTorch template for binary image segmentation.
Project structure:

```
Project1/
├── ADDA/
│   ├── train.csv
│   ├── val.csv
│   └── png_256/
│       ├── images/    # 600 training/validation images
│       └── labels/    # 600 corresponding labels
├── dataset.py
├── model.py
├── main.py
├── inference.py
├── utils.py
└── requirements.txt
```

Each CSV must contain at least these columns:
- `ct_path` : path to the input image
- `label_path` : path to the corresponding binary mask

> Tip: Paths can be relative (e.g., `png_256/images/xxx.png`). The dataset class will automatically look for files in `<data_dir>/png_256/images` and `<data_dir>/png_256/labels`.

## Quick Start

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Train
```bash
# Use default parameters (data_dir defaults to ./ADDA)
python main.py

# Or specify parameters
python main.py --data_dir "./ADDA" --epochs 30 --batch_size 8 --img_size 256 --lr 1e-3 --out_dir "./runs/exp_ADDA"
```

### 3) Inference (single image)
```bash
python inference.py --ckpt ./runs/exp_ADDA/best_model.pt --image "./ADDA/png_256/images/your_image.png" --out_dir "./runs/exp_ADDA/preds"
```

### 4) Inference (folder)
```bash
python inference.py --ckpt ./runs/exp_ADDA/best_model.pt --in_dir "./ADDA/png_256/images" --out_dir "./runs/exp_ADDA/preds"
```

## What Students Should Modify

- **`model.py` → `StudentModel`**: Replace the baseline FCN with your own architecture (e.g., U-Net, DeepLab, FPN).
- **`dataset.py`**: Add stronger data augmentation (random crop, intensity changes, elastic deformation, etc.).
- **`main.py`**:
  - Add early stopping and learning rate schedulers.
  - Try different loss mixes (e.g., BCE + Dice with different weights).
  - Extend metrics to multi-class if needed.
- **`inference.py`**: Save overlays or color masks; add post-processing (morphology, CRF).

## File Overview

- `dataset.py` — PyTorch Dataset for segmentation from CSV.
- `model.py` — Baseline FCN + `StudentModel` placeholder for student work.
- `utils.py` — Metrics (Dice, IoU), curves plotting, seeding.
- `main.py` — Training & validation loop, curve saving, best checkpoint saving.
- `inference.py` — Single/batch inference script.
- `requirements.txt` — Minimal dependencies.

## Dataset Information

- This project uses 600 images from the ADDA dataset (png_256 format)
- `train.csv` and `val.csv` contain image-label pairs for training and validation
- Images are single-channel grayscale, 256×256 pixels
- Labels are binary masks (0 and 255)

Good luck and have fun!
