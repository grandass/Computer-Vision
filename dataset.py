from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class SegDataset(Dataset):
    """
    Read image/mask pairs from a CSV file.

    Required CSV columns:
        - ct_path: image path
        - label_path: mask path

    The dataset tries to resolve relative paths in two steps:
        1) Look for files with the same filename under <root_dir>/sub_images and <root_dir>/sub_label.
        2) Fall back to resolving the path relative to <root_dir>.

    Images are treated as single-channel (grayscale). Change to RGB if needed.
    Masks are binarized to {0,1}.
    """
    def __init__(
        self,
        csv_path: str | Path,
        root_dir: str | Path,
        image_root_sub: str = "sub_images",
        label_root_sub: str = "sub_label",
        img_size: Optional[int] = 256,
        augment: bool = False,
        to_float32: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        self.root_dir = Path(root_dir)
        self.image_root = self.root_dir / image_root_sub
        self.label_root = self.root_dir / label_root_sub
        self.img_size = img_size
        self.augment = augment
        self.to_float32 = to_float32

        assert "ct_path" in self.df.columns and "label_path" in self.df.columns,             "CSV must contain 'ct_path' and 'label_path' columns."

        self.samples = []
        for _, row in self.df.iterrows():
            img_p = Path(str(row["ct_path"])).expanduser()
            lbl_p = Path(str(row["label_path"])).expanduser()

            # Prefer files placed under sub_images/sub_label (using base filename)
            img_abs = self.image_root / img_p.name if not img_p.is_absolute() else img_p
            lbl_abs = self.label_root / lbl_p.name if not lbl_p.is_absolute() else lbl_p

            if not img_abs.exists():
                img_abs = (self.root_dir / img_p).resolve()
            if not lbl_abs.exists():
                lbl_abs = (self.root_dir / lbl_p).resolve()

            self.samples.append((img_abs, lbl_abs))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> Image.Image:
        return Image.open(path).convert("L")  # use 'RGB' for color inputs

    def _load_mask(self, path: Path) -> Image.Image:
        return Image.open(path).convert("L")

    def _random_augment(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # Simple augmentation: horizontal flip + random 90-degree rotations
        # import torch
        # if torch.rand(1).item() < 0.5:
        #     img = TF.hflip(img); mask = TF.hflip(mask)
        # if torch.rand(1).item() < 0.5:
        #     k = int(torch.randint(0, 4, (1,)).item())
        #     if k:
        #         img = TF.rotate(img, 90 * k)
        #         mask = TF.rotate(mask, 90 * k)
        # return img, mask

        import torch
        import torchvision.transforms.functional as TF
        import random

        # Horizontal flip
        if torch.rand(1).item() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # Vertical flip
        if torch.rand(1).item() < 0.3:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        # Small random rotation (-90° to +90°)
        if torch.rand(1).item() < 0.5:
            angle = random.uniform(-90, 90)
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # Random affine transform (translation + scale)
        if torch.rand(1).item() < 0.4:
            translate = (random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05))
            scale = random.uniform(0.9, 1.1)
            img = TF.affine(img, angle=0, translate=(int(translate[0] * img.width), int(translate[1] * img.height)),
                            scale=scale, shear=[0.0, 0.0], interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.affine(mask, angle=0, translate=(int(translate[0] * mask.width), int(translate[1] * mask.height)),
                             scale=scale, shear=[0.0, 0.0], interpolation=TF.InterpolationMode.NEAREST)

        # Random brightness/contrast jitter
        if torch.rand(1).item() < 0.5:
            img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
            img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))

        return img, mask


    def __getitem__(self, idx: int):
        img_p, lbl_p = self.samples[idx]
        img = self._load_image(img_p)
        mask = self._load_mask(lbl_p)

        if self.img_size is not None:
            img = TF.resize(img, [self.img_size, self.img_size], antialias=True)
            mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            img, mask = self._random_augment(img, mask)

        img_t = TF.to_tensor(img)        # [1,H,W], float32 in [0,1]
        mask_t = TF.to_tensor(mask)      # [1,H,W], float32
        mask_t = (mask_t > 0.5).float()  # binarize

        img_t = img_t.float()
        mask_t = mask_t.float()

        return {
            "image": img_t,
            "mask": mask_t,
            "img_path": str(img_p),
            "mask_path": str(lbl_p),
        }
