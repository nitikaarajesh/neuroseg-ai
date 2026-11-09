"""Score a folder of test predictions against ground truth.
Usage:  python scripts/evaluate.py --selftest
        python scripts/evaluate.py results/unet_seed0/preds unet_seed0"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib, torch
from monai.metrics import compute_dice, compute_hausdorff_distance
from monai.networks.utils import one_hot

ROOT = Path(r"C:\data\msd\Task04_Hippocampus")
LABELS = {1: "ant", 2: "post"}

def score_case(pred, truth):
    """Dice, HD95 (mm, voxels are 1 mm) and volume error for each label of one case."""
    p = one_hot(torch.as_tensor(pred)[None, None].long(), 3)
    t = one_hot(torch.as_tensor(truth)[None, None].long(), 3)
    dice = compute_dice(p, t, include_background=False)[0].numpy()
    hd95 = compute_hausdorff_distance(p, t, include_background=False, percentile=95)[0].numpy()
    row = {}
    for i, (v, name) in enumerate(LABELS.items()):
        tv, pv = int((truth == v).sum()), int((pred == v).sum())
        row[f"dice_{name}"] = float(dice[i])
        row[f"hd95_{name}"] = float(hd95[i])
        row[f"vol_true_{name}"], row[f"vol_pred_{name}"] = tv, pv      # mm^3 because voxels are 1 mm
        row[f"vol_err_pct_{name}"] = 100 * (pv - tv) / tv
    return row

def selftest():
    """Check the metric code on shapes where we KNOW the answer."""
    z, y, x = np.ogrid[:40, :40, :40]
    ball = (z - 20) ** 2 + (y - 20) ** 2 + (x - 20) ** 2 <= 8 ** 2
    a = np.zeros((40, 40, 40), int); a[ball & (x < 20)] = 1; a[ball & (x >= 20)] = 2
    r = score_case(a, a)
    assert abs(r["dice_ant"] - 1) < 1e-6 and r["hd95_ant"] < 1e-6 and r["vol_err_pct_ant"] == 0
    r = score_case(np.roll(a, 2, axis=0), a)                            # shift by 2 voxels = 2 mm
    print(f"Shifted by 2 mm -> HD95 ant {r['hd95_ant']:.2f} (expect about 2), Dice ant {r['dice_ant']:.3f} (expect < 1)")
    print("Selftest OK")

def main(pred_dir, tag):
    split = json.loads(Path("splits/split_v1.json").read_text())
    rows = []
    for name in split["test"]:
        lab_img = nib.load(ROOT / "labelsTr" / name)
        truth = np.asarray(lab_img.dataobj).astype(np.int64)
        pred_img = nib.load(Path(pred_dir) / name)
        pred = np.asarray(pred_img.dataobj).astype(np.int64)
        assert pred.shape == truth.shape, f"{name}: shape {pred.shape} vs {truth.shape}"
        assert np.allclose(pred_img.affine, lab_img.affine), f"{name}: prediction is in a different space"
        rows.append({"case": name, "model": tag, **score_case(pred, truth)})
    df = pd.DataFrame(rows)
    out = Path("results/metrics"); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{tag}.csv", index=False)
    print(f"{tag}: {len(df)} test cases scored -> results/metrics/{tag}.csv")
    print(df.drop(columns=["case", "model"]).median().round(3).to_string())

if __name__ == "__main__":
    selftest() if sys.argv[1] == "--selftest" else main(sys.argv[1], sys.argv[2])