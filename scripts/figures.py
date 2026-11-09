"""README figures. Usage: python scripts/figures.py unet_seed0 nnunet_tta  (first model picks the worst cases)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(r"C:\data\msd\Task04_Hippocampus")
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
tags = sys.argv[1:]
dfs = {t: pd.read_csv(f"results/metrics/{t}.csv").set_index("case").sort_index() for t in tags}

def pred_dir(t):   # unet_seed0 -> results/unet_seed0/preds ; nnunet_tta -> results/preds_tta
    return Path("results") / t / "preds" if t.startswith("unet") else Path("results") / t.replace("nnunet_", "preds_")

# 1. Dice per test case
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, lab in zip(axes, ["ant", "post"]):
    for i, (t, df) in enumerate(dfs.items()):
        y = df[f"dice_{lab}"]
        ax.scatter(np.random.default_rng(i).normal(i, 0.06, len(y)), y, s=12, alpha=0.6)
        ax.hlines(y.median(), i - 0.25, i + 0.25, color="k")
    ax.set_xticks(range(len(dfs)), list(dfs), rotation=20)
    ax.set_title(f"Dice per test case, {lab}erior (line = median)")
axes[0].set_ylabel("Dice")
plt.tight_layout(); plt.savefig(FIG / "dice_per_case.png", dpi=150); plt.close()

# 2. Bland-Altman: does the predicted volume agree with the expert's?
fig, axes = plt.subplots(len(dfs), 2, figsize=(10, 3.6 * len(dfs)), squeeze=False)
for r, (t, df) in enumerate(dfs.items()):
    for c, lab in enumerate(["ant", "post"]):
        tv, pv = df[f"vol_true_{lab}"], df[f"vol_pred_{lab}"]
        diff = pv - tv
        bias, sd = diff.mean(), diff.std()
        ax = axes[r, c]
        ax.scatter((tv + pv) / 2, diff, s=12)
        for yv, ls in [(bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")]:
            ax.axhline(yv, color="k", ls=ls, lw=0.8)
        ax.set_title(f"{t}, {lab}: bias {bias:+.0f} mm³, 95% limits {bias-1.96*sd:+.0f} to {bias+1.96*sd:+.0f}", fontsize=9)
        ax.set_xlabel("mean of true and predicted volume (mm³)"); ax.set_ylabel("predicted − true (mm³)")
plt.tight_layout(); plt.savefig(FIG / "bland_altman.png", dpi=150); plt.close()

# 3. The 3 worst cases (by the first model), truth vs every model
d = dfs[tags[0]]
worst = (d.dice_ant + d.dice_post).nsmallest(3).index
cmap = ListedColormap(["red", "dodgerblue"])
fig, axes = plt.subplots(3, 1 + len(tags), figsize=(3.2 * (1 + len(tags)), 9.5), squeeze=False)
for r, case in enumerate(worst):
    img = nib.load(ROOT / "imagesTr" / case).get_fdata()
    truth = np.asarray(nib.load(ROOT / "labelsTr" / case).dataobj)
    s = np.argmax((truth > 0).sum(axis=(1, 2)))
    panels = [("truth", truth)] + [(t, np.asarray(nib.load(pred_dir(t) / case).dataobj)) for t in tags]
    for c, (title, lab) in enumerate(panels):
        ax = axes[r, c]
        ax.imshow(img[s].T, cmap="gray", origin="lower")
        ax.imshow(np.ma.masked_equal(lab[s], 0).T, cmap=cmap, vmin=1, vmax=2, alpha=0.5, origin="lower")
        sub = "" if title == "truth" else f"\nDice {dfs[title].loc[case, 'dice_ant']:.2f} / {dfs[title].loc[case, 'dice_post']:.2f}"
        ax.set_title(f"{case.replace('.nii.gz', '')} {title}{sub}", fontsize=8); ax.axis("off")
plt.tight_layout(); plt.savefig(FIG / "worst_cases.png", dpi=150); plt.close()
print("Saved figures/dice_per_case.png, figures/bland_altman.png, figures/worst_cases.png")