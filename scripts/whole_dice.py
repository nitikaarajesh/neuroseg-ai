import json, sys, numpy as np, nibabel as nib
from pathlib import Path
split = json.load(open("splits/split_v1.json"))
lab = Path(r"C:\data\msd\Task04_Hippocampus\labelsTr")
dirs = {"unet_seed0": Path("results/unet_seed0/preds"), "nnunet_tta": Path("results/preds_tta")}
for tag, d in dirs.items():
    ds = []
    for name in split["test"]:
        t = nib.load(lab / name).get_fdata() > 0      # any hippocampus label
        p = nib.load(d / name).get_fdata() > 0
        ds.append(2 * (t & p).sum() / (t.sum() + p.sum()))
    ds = np.array(ds)
    print(f"{tag:12s} whole-hippocampus Dice mean {ds.mean():.3f}  median {np.median(ds):.3f}  min {ds.min():.3f}")