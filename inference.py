import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from model import StudentModel

def load_image(path, img_size=256):
    im = Image.open(path).convert("L")
    if img_size is not None:
        im = TF.resize(im, [img_size, img_size], antialias=True)
    t = TF.to_tensor(im).unsqueeze(0)  # [1,1,H,W]
    return t, im

def save_mask(mask_prob, save_path, threshold=0.5):
    m = (mask_prob[0,0].cpu().numpy() >= threshold).astype(np.uint8) * 255
    Image.fromarray(m).save(save_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="path to best_model.pt")
    ap.add_argument("--image", type=str, required=False, help="single image path")
    ap.add_argument("--in_dir", type=str, required=False, help="directory of images (.png/.jpg)")
    ap.add_argument("--out_dir", type=str, default="./preds")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)

    model = StudentModel(in_ch=1, base_ch=32).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    if args.image:
        paths.append(Path(args.image))
    if args.in_dir:
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            paths += list(Path(args.in_dir).glob(ext))

    assert paths, "Please provide --image or --in_dir"

    with torch.no_grad():
        for p in paths:
            t, _ = load_image(p, img_size=args.img_size)
            t = t.to(device)
            logits = model(t)
            probs = torch.sigmoid(logits)  # [1,1,H,W]
            save_mask(probs, out_dir / f"{p.stem}_pred.png", threshold=args.thr)

    print(f"Saved predictions to: {out_dir}")

if __name__ == "__main__":
    main()
