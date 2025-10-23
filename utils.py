from pathlib import Path
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def dice_coeff(probs, targets, eps=1e-6):
    p = probs.contiguous().view(probs.size(0), -1)
    t = targets.contiguous().view(targets.size(0), -1)
    inter = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()

def iou_score(probs, targets, eps=1e-6):
    p = probs.contiguous().view(probs.size(0), -1)
    t = targets.contiguous().view(targets.size(0), -1)
    inter = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()

def save_history_and_curves(history, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Loss
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss")
    plt.tight_layout(); plt.savefig(out / "loss_curve.png"); plt.close()

    # Dice
    plt.figure()
    plt.plot(history["train_dice"], label="train")
    plt.plot(history["val_dice"], label="val")
    plt.xlabel("epoch"); plt.ylabel("dice"); plt.legend(); plt.title("Dice")
    plt.tight_layout(); plt.savefig(out / "dice_curve.png"); plt.close()

    # IoU
    plt.figure()
    plt.plot(history["train_iou"], label="train")
    plt.plot(history["val_iou"], label="val")
    plt.xlabel("epoch"); plt.ylabel("IoU"); plt.legend(); plt.title("IoU")
    plt.tight_layout(); plt.savefig(out / "iou_curve.png"); plt.close()
