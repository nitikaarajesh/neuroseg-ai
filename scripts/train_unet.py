"""Final MONAI 3D U-Net run on splits/split_v1.json. Usage: python scripts/train_unet.py --seed 0"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import nibabel as nib
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
                              NormalizeIntensityd, SpatialPadd, RandSpatialCropd,
                              RandFlipd, EnsureTyped, AsDiscrete)
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=200)
args = ap.parse_args()

ROOT = Path(r"C:\data\msd\Task04_Hippocampus")
OUT = Path(f"results/unet_seed{args.seed}"); (OUT / "preds").mkdir(parents=True, exist_ok=True)
PATCH = (32, 32, 32)
set_determinism(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

split = json.loads(Path("splits/split_v1.json").read_text())
def files(names):
    return [{"image": str(ROOT / "imagesTr" / n), "label": str(ROOT / "labelsTr" / n)} for n in names]

keys = ["image", "label"]
train_tf = Compose([LoadImaged(keys), EnsureChannelFirstd(keys), Orientationd(keys, axcodes="RAS"),
                    NormalizeIntensityd(keys="image"), SpatialPadd(keys, spatial_size=PATCH),
                    RandSpatialCropd(keys, roi_size=PATCH, random_size=False),
                    RandFlipd(keys, prob=0.5, spatial_axis=0), EnsureTyped(keys)])
eval_tf = Compose([LoadImaged(keys), EnsureChannelFirstd(keys), Orientationd(keys, axcodes="RAS"),
                   NormalizeIntensityd(keys="image"), EnsureTyped(keys)])

train_loader = DataLoader(CacheDataset(files(split["train"]), train_tf, num_workers=0),
                          batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(CacheDataset(files(split["val"]), eval_tf, num_workers=0), batch_size=1, num_workers=0)

model = UNet(spatial_dims=3, in_channels=1, out_channels=3, channels=(16, 32, 64, 128),
             strides=(2, 2, 2), num_res_units=2).to(device)
loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
post_pred, post_label = AsDiscrete(argmax=True, to_onehot=3), AsDiscrete(to_onehot=3)

history, best, best_epoch, t0 = [], -1.0, 0, time.time()
for epoch in range(1, args.epochs + 1):
    model.train(); losses = []
    for b in train_loader:
        opt.zero_grad()
        loss = loss_fn(model(b["image"].to(device)), b["label"].to(device))
        loss.backward(); opt.step(); losses.append(loss.item())
    row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            for b in val_loader:
                logits = sliding_window_inference(b["image"].to(device), PATCH, 4, model)
                dice_metric(y_pred=[post_pred(p) for p in decollate_batch(logits)],
                            y=[post_label(l) for l in decollate_batch(b["label"].to(device))])
        d = dice_metric.aggregate().cpu().numpy(); dice_metric.reset()
        row.update(dice_ant=float(d[0]), dice_post=float(d[1]))
        if d.mean() > best:
            best, best_epoch = float(d.mean()), epoch
            torch.save(model.state_dict(), OUT / "best_model.pt")
    history.append(row); print(row, flush=True)
train_minutes = (time.time() - t0) / 60

# Predict the locked test set ONCE with the best model, save as NIfTI with the original affine
model.load_state_dict(torch.load(OUT / "best_model.pt", map_location=device)); model.eval()
test_loader = DataLoader(CacheDataset(files(split["test"]), eval_tf, num_workers=0), batch_size=1, num_workers=0)
t1 = time.time()
with torch.no_grad():
    for name, b in zip(split["test"], test_loader):
        pred = sliding_window_inference(b["image"].to(device), PATCH, 4, model).argmax(1)[0]
        affine = nib.load(ROOT / "imagesTr" / name).affine
        nib.save(nib.Nifti1Image(pred.cpu().numpy().astype(np.uint8), affine), OUT / "preds" / name)
sec_per_case = (time.time() - t1) / len(split["test"])

(OUT / "history.json").write_text(json.dumps(history, indent=1))
(OUT / "run_info.json").write_text(json.dumps({
    "seed": args.seed, "epochs": args.epochs, "best_epoch": best_epoch, "best_val_dice": best,
    "train_minutes": round(train_minutes, 1), "infer_sec_per_case": round(sec_per_case, 3),
    "parameters": sum(p.numel() for p in model.parameters()), "device": str(device)}, indent=1))
print(f"DONE seed {args.seed}: best val Dice {best:.3f} at epoch {best_epoch}, {train_minutes:.1f} min")