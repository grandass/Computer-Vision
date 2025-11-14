import argparse
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SegDataset
from utils import set_seed, dice_coeff, iou_score, save_history_and_curves

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(1,2,3)) + self.eps
        den = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + self.eps
        loss = 1 - num / den
        return loss.mean()

def train_one_epoch(model, loader, device, criterion, bce, optimizer, scaler=None):
    model.train()
    running_loss, running_dice, running_iou = 0.0, 0.0, 0.0
    for batch in loader:
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            logits = model(imgs)
            loss_dice = criterion(logits, masks)
            loss_bce = bce(logits, masks)
            loss = 0.5 * loss_bce + 0.5 * loss_dice
            probs = torch.sigmoid(logits)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        running_dice += dice_coeff(probs.detach(), masks)
        running_iou  += iou_score(probs.detach(), masks)

    n = len(loader.dataset)
    return running_loss / n, running_dice / len(loader), running_iou / len(loader)

@torch.no_grad()
def evaluate(model, loader, device, criterion, bce):
    model.eval()
    running_loss, running_dice, running_iou = 0.0, 0.0, 0.0
    for batch in loader:
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(imgs)
        loss_dice = criterion(logits, masks)
        loss_bce = bce(logits, masks)
        loss = 0.5 * loss_bce + 0.5 * loss_dice
        probs = torch.sigmoid(logits)
        running_loss += loss.item() * imgs.size(0)
        running_dice += dice_coeff(probs, masks)
        running_iou  += iou_score(probs, masks)

    n = len(loader.dataset)
    return running_loss / n, running_dice / len(loader), running_iou / len(loader)

def main():
    from model import StudentModel
    print(StudentModel)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./ADDA")
    ap.add_argument("--train_csv", type=str, default="train.csv")
    ap.add_argument("--val_csv", type=str, default="val.csv")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="./runs/exp1")
    ap.add_argument("--amp", action="store_true", help="use mixed precision")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print("Using device:", device)
    print("CUDA available:", torch.cuda.is_available())
    print("Current device name:", torch.cuda.get_device_name(0))
    train_ds = SegDataset(csv_path=Path(args.data_dir) / args.train_csv,
                          root_dir=args.data_dir, 
                          image_root_sub="png_256/images",
                          label_root_sub="png_256/labels",
                          img_size=args.img_size, augment=True)
    val_ds   = SegDataset(csv_path=Path(args.data_dir) / args.val_csv,
                          root_dir=args.data_dir,
                          image_root_sub="png_256/images",
                          label_root_sub="png_256/labels",
                          img_size=args.img_size, augment=False)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_ld   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    model = StudentModel().to(device)
    dice_loss = DiceLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=10, T_mult=2, eta_min=1e-6
    # )
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    history = {"train_loss":[], "val_loss":[], "train_dice":[], "val_dice":[], "train_iou":[], "val_iou":[]}
    best_dice, best_path = -1.0, out_dir / "best_model.pt"
    best_val_dice = -1.0
    patience = 5
    counter = 0
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_dice, tr_iou = train_one_epoch(model, train_ld, device, dice_loss, bce_loss, optimizer, scaler)
        va_loss, va_dice, va_iou = evaluate(model, val_ld, device, dice_loss, bce_loss)

        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
        history["train_dice"].append(tr_dice); history["val_dice"].append(va_dice)
        history["train_iou"].append(tr_iou);   history["val_iou"].append(va_iou)

        print(f"Epoch {epoch:03d} | "
              f"Train L {tr_loss:.4f} D {tr_dice:.4f} IoU {tr_iou:.4f} || "
              f"Val L {va_loss:.4f} D {va_dice:.4f} IoU {va_iou:.4f}")

        if va_dice > best_dice:
            best_dice = va_dice
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "dice": best_dice, "cfg": vars(args)}, best_path)
            print(f"  -> Saved best: {best_path} (dice={best_dice:.4f})")
            counter = 0
        else:
            counter += 1
            print(f"  -> No improvement, counter={counter}/{patience}")

        if counter >= patience:
            print(f"Early stopping triggered. Best val dice: {best_val_dice:.4f}")
            break

        save_history_and_curves(history, out_dir)

        scheduler.step(va_loss)

    torch.save({"epoch": args.epochs, "state_dict": model.state_dict(), "dice": best_dice, "cfg": vars(args)}, out_dir / "last_model.pt")
    print("Training finished. Best dice:", best_dice)

if __name__ == "__main__":
    main()

# more data augmentation
# deep supervision
# drop / spatial dropout, decoder blocks, spatialdropout2d
# cosineannealingLR
# weight initialization
