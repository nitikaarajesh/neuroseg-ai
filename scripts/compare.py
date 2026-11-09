"""Comparison table with case-level bootstrap 95% CIs, plus paired differences vs the FIRST model given.
Usage: python scripts/compare.py unet_seed0 unet_seed1 unet_seed2
       python scripts/compare.py unet_mean unet_seed0 unet_seed1 unet_seed2 nnunet_tta nnunet_notta"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

METRICS = ["dice_ant", "dice_post", "hd95_ant", "hd95_post", "abs_vol_err_ant", "abs_vol_err_post"]
rng = np.random.default_rng(0)
B = 5000

def load(tag):
    df = pd.read_csv(Path("results/metrics") / f"{tag}.csv").set_index("case").sort_index()
    for n in ["ant", "post"]:
        df[f"abs_vol_err_{n}"] = df[f"vol_err_pct_{n}"].abs()
    if not np.isfinite(df[METRICS].values).all():
        print(f"WARNING: {tag} has infinite values (a label predicted as empty?)")
    return df[METRICS]

def boot(values):
    v = np.asarray(values, dtype=float)
    means = v[rng.integers(0, len(v), (B, len(v)))].mean(axis=1)
    return v.mean(), *np.percentile(means, [2.5, 97.5])

tags = sys.argv[1:]
data = {t: load(t) for t in tags if t != "unet_mean"}
if "unet_mean" in tags:   # per-case average over all U-Net seeds = "the U-Net" as one model
    seeds = [t for t in data if t.startswith("unet_seed")]
    data = {"unet_mean": sum(data[s] for s in seeds) / len(seeds), **data}

rows = []
for name, df in data.items():
    rows.append({"model": name, **{m: "{:.3f} [{:.3f}, {:.3f}]".format(*boot(df[m])) for m in METRICS}})
table = pd.DataFrame(rows)
print("Mean [95% bootstrap CI] over", len(next(iter(data.values()))), "test cases")
print(table.to_string(index=False))
Path("results").mkdir(exist_ok=True)
table.to_csv("results/comparison.csv", index=False)

ref = next(iter(data))
print(f"\nPaired differences vs {ref} (same cases). If the CI does not include 0, the difference is real:")
for name, df in data.items():
    if name == ref:
        continue
    for m in ["dice_ant", "dice_post", "hd95_ant", "hd95_post"]:
        mean, lo, hi = boot((df[m] - data[ref][m]).values)
        flag = "" if lo <= 0 <= hi else "   <-- CI excludes 0"
        print(f"  {name:13} {m:10} {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]{flag}")