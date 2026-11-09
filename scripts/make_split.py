import json, random
from pathlib import Path

ROOT = Path(r"C:\data\msd\Task04_Hippocampus")
meta = json.loads((ROOT / "dataset.json").read_text())
cases = sorted(Path(e["image"]).name for e in meta["training"])

rng = random.Random(42)
rng.shuffle(cases)
test, val, train = cases[:52], cases[52:94], cases[94:]   # 20% test, then 42 val, 166 train

Path("splits").mkdir(exist_ok=True)
Path("splits/split_v1.json").write_text(json.dumps({
    "seed": 42,
    "note": "Random case-level split; grouping by person not possible (see DECISIONS.md D2). "
            "Test cases are used once, for the final comparison only.",
    "train": sorted(train), "val": sorted(val), "test": sorted(test)}, indent=1))
print(f"train {len(train)} | val {len(val)} | test {len(test)}")