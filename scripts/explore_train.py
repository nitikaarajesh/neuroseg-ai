"""Exploratory first training run, NOT the final experiment.
Random 80/20 split by case (may leak if one person has a left and a right crop).
Goal: watch a 3D U-Net learn hippocampus segmentation and produce outputs to study."""
import json, time, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
                              NormalizeIntensityd, SpatialPadd, RandSpatialCropd,
                              RandFlipd, EnsureTyped, AsDiscrete)
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

# ---------------- settings ----------------
ROOT = Path(r"C:\data\msd\Task04_Hippocampus")
OUT = Path("results/explore_run1"); OUT.mkdir(parents=True, exist_ok=True)
PATCH = (32, 32, 32)      # option (c): train on fixed cubes, predict with a sliding window
TIME_LIMIT_MIN = 75       # stop training after this many minutes
SEED = 42
set_determinism(SEED); random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------- data ----------------
meta = json.loads((ROOT / "dataset.json").read_text())
files = [{"image": str(ROOT / e["image"]), "label": str(ROOT / e["label"])} for e in meta["training"]]
random.shuffle(files)
val_files, train_files = files[:52], files[52:]
print(f"Train cases: {len(train_files)} | Val cases: {len(val_files)}")

keys = ["image", "label"]
train_tf = Compose([
    LoadImaged(keys), EnsureChannelFirstd(keys), Orientationd(keys, axcodes="RAS"),
    NormalizeIntensityd(keys="image"),                    # each scan to mean 0, std 1
    SpatialPadd(keys, spatial_size=PATCH),                # only pads scans smaller than the patch
    RandSpatialCropd(keys, roi_size=PATCH, random_size=False),
    RandFlipd(keys, prob=0.5, spatial_axis=0),            # flip LEFT-RIGHT only (see note)
    EnsureTyped(keys),
])
val_tf = Compose([
    LoadImaged(keys), EnsureChannelFirstd(keys), Orientationd(keys, axcodes="RAS"),
    NormalizeIntensityd(keys="image"), EnsureTyped(keys),
])
train_ds = CacheDataset(train_files, train_tf, cache_rate=1.0, num_workers=0)
val_ds = CacheDataset(val_files, val_tf, cache_rate=1.0, num_workers=0)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)  # sizes differ, so 1 at a time

# ---------------- model ----------------
model = UNet(spatial_dims=3, in_channels=1, out_channels=3,
             channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2).to(device)
print("Parameters:", sum(p.numel() for p in model.parameters()))
loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
post_pred = AsDiscrete(argmax=True, to_onehot=3)
post_label = AsDiscrete(to_onehot=3)

# ---------------- training ----------------
history, best, start, epoch = [], -1.0, time.time(), 0
while True:
    epoch += 1
    model.train(); losses = []
    for batch in train_loader:
        x, y = batch["image"].to(device), batch["label"].to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward(); opt.step()
        losses.append(loss.item())
    row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
           "minutes": round((time.time() - start) / 60, 1)}

    if epoch % 2 == 0:  # validate every 2 epochs on whole scans
        model.eval()
        with torch.no_grad():
            for b in val_loader:
                x, y = b["image"].to(device), b["label"].to(device)
                logits = sliding_window_inference(x, PATCH, 4, model)
                dice_metric(y_pred=[post_pred(p) for p in decollate_batch(logits)],
                            y=[post_label(l) for l in decollate_batch(y)])
        d = dice_metric.aggregate().cpu().numpy(); dice_metric.reset()
        row["dice_ant"], row["dice_post"] = float(d[0]), float(d[1])
        if d.mean() > best:
            best = float(d.mean()); row["saved_best"] = True
            torch.save(model.state_dict(), OUT / "best_model.pt")

    history.append(row); print(row, flush=True)
    (OUT / "history.json").write_text(json.dumps(history, indent=1))
    if row["minutes"] > TIME_LIMIT_MIN:
        break

# ---------------- outputs to study ----------------
h = pd.DataFrame(history)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(h["epoch"], h["train_loss"]); ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch")
v = h.dropna(subset=["dice_ant"])
ax[1].plot(v["epoch"], v["dice_ant"], label="anterior", color="red")
ax[1].plot(v["epoch"], v["dice_post"], label="posterior", color="dodgerblue")
ax[1].set_title("Validation Dice"); ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1); ax[1].legend()
plt.tight_layout(); plt.savefig(OUT / "curves.png", dpi=120); plt.close()

model.load_state_dict(torch.load(OUT / "best_model.pt", map_location=device)); model.eval()
rows, examples = [], []
with torch.no_grad():
    for i, b in enumerate(val_loader):
        x = b["image"].to(device)
        y = b["label"][0, 0].cpu().numpy()
        pred = sliding_window_inference(x, PATCH, 4, model).argmax(1)[0].cpu().numpy()
        rows.append({"case": Path(val_files[i]["image"]).name,
                     "true_ant_mm3": int((y == 1).sum()), "pred_ant_mm3": int((pred == 1).sum()),
                     "true_post_mm3": int((y == 2).sum()), "pred_post_mm3": int((pred == 2).sum())})
        examples.append((x[0, 0].cpu().numpy(), y, pred))
vol = pd.DataFrame(rows)
vol["ant_error_%"] = 100 * (vol.pred_ant_mm3 - vol.true_ant_mm3) / vol.true_ant_mm3
vol["post_error_%"] = 100 * (vol.pred_post_mm3 - vol.true_post_mm3) / vol.true_post_mm3
vol.round(1).to_csv(OUT / "val_volumes.csv", index=False)

# best 2 and worst 2 cases by volume error, shown as truth vs prediction
vol["abs_err"] = vol[["ant_error_%", "post_error_%"]].abs().mean(axis=1)
order = vol.sort_values("abs_err").index.tolist()
pick = order[:2] + order[-2:]
cmap = ListedColormap(["red", "dodgerblue"])
fig, axes = plt.subplots(len(pick), 2, figsize=(7, 3.2 * len(pick)))
for r, idx in enumerate(pick):
    img, y, p = examples[idx]
    s = np.argmax((y > 0).sum(axis=(1, 2)))       # slice with most hippocampus
    for c, (lab, title) in enumerate([(y, "truth"), (p, "prediction")]):
        axes[r, c].imshow(img[s].T, cmap="gray", origin="lower")
        axes[r, c].imshow(np.ma.masked_equal(lab[s], 0).T, cmap=cmap, vmin=1, vmax=2, alpha=0.5, origin="lower")
        axes[r, c].set_title(f"{vol.case[idx]} {title}"); axes[r, c].axis("off")
plt.tight_layout(); plt.savefig(OUT / "best_and_worst.png", dpi=120); plt.close()

print("\nDONE. Best mean val Dice:", round(best, 3))
print("Median absolute volume error (%):", vol[["ant_error_%", "post_error_%"]].abs().median().round(1).to_dict())