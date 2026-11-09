# Decisions

A log of the choices behind this project, with the reasoning. Each entry records
whether it was fixed *before* the test results were seen (pre-registered) or
*after* (post-hoc). Post-hoc choices are labelled so they are not mistaken for
pre-registered analysis.

## D1 — Data split (pre-registered)

A single fixed split of the 260 labelled MSD Task04 Hippocampus cases, drawn once
with seed 42 (`scripts/make_split.py`, stored in `splits/split_v1.json`):
166 train / 42 validation / 52 test (20% test, then 42 val from the remainder).

- The split is case-level and random.
- The same `split_v1.json` is used by both the MONAI U-Net and nnU-Net v2 (for
  nnU-Net, the train+val cases become its custom `splits_final.json` fold and the
  52 test cases are held out of its dataset entirely). This keeps the comparison fair.
- The test set is used **once**, for the final comparison only.

## D2 — Subject grouping cannot be verified (pre-registered limitation)

MSD Task04 Hippocampus is distributed as pre-cut crops. The full release is about
390 crops (260 labelled + 130 unlabelled), roughly 195 people × 2 sides, but:

- the original scan origins are wiped,
- case IDs are interleaved,
- all crops face one direction,
- there are no intensity "twins" that would let crops be paired.

So crops **cannot be grouped back to a person**. A case-level split may therefore
put a left and a right crop from the same person on different sides of the split.
Both models use the identical split, so the *comparison* between them is still fair,
but the *absolute* Dice may be slightly optimistic relative to a subject-grouped split.
This limitation is acknowledged and not fixable with the released data.

## D3 — Comparison rule (pre-registered, written BEFORE the nnU-Net results were seen)

Fixed before any nnU-Net test score existed, to avoid choosing the rule to fit the result:

- **Reference model:** `unet_mean` — the per-case average of the three U-Net seeds,
  treated as "the U-Net" as a single model.
- **When a difference counts as real:** a difference is only counted if (a) its 95%
  paired bootstrap CI excludes 0 **and** (b) it is larger than the seed-to-seed
  spread (about 0.006 Dice). Both conditions must hold.
- **Primary endpoints:** Dice and HD95, per label (anterior, posterior).
- **Secondary endpoints:** absolute volume error and Bland–Altman agreement.
- **nnU-Net is reported both with and without test-time augmentation (TTA).**

## D4 — Whole-hippocampus Dice check (POST-HOC, written AFTER the results were seen)

Added after seeing the per-label results, and labelled post-hoc for that reason.

Hypothesis to test: that most of the segmentation error sits on the anterior/posterior
*boundary* (the internal split between labels 1 and 2). If so, merging the two labels
into one whole-hippocampus mask and re-scoring Dice should remove most of the error.

Method: merge labels 1 and 2 into a single foreground mask and recompute Dice per case
(`scripts/whole_dice.py`), for `unet_seed0` and `nnunet_tta`.

This is exploratory and was not part of the pre-registered D3 rule.
