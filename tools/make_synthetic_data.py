#!/usr/bin/env python3
"""make_synthetic_data.py - build a small, fully synthetic ABIDE-A-shaped dataset.

No real participant data is used or derived here: every value is drawn from a
fixed-seed generator, so the smoke test can run without the restricted inputs.

Layout produced (matching baalexp.data.load_abide_a):

    <out>/ABIDE_A/processed_standard_data.csv   features..., label (1/2)
    <out>/ABIDE_A/modal_feat_dict.npy           {group_name: [feature names]}

Usage: python make_synthetic_data.py --out ./synthetic_data --n 120 --seed 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "synthetic_data"))
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--d", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n, d = args.n, args.d

    sex = rng.integers(0, 2, size=n)
    group_signal = rng.normal(0.0, 1.0, size=n) + 0.6 * (sex - 0.5)
    X = rng.normal(0.0, 1.0, size=(n, d))
    X[:, 0] = group_signal
    X[:, 1] = 1.0 - sex          # SEX_2, the exact complement of SEX_1
    X[:, 0] = rng.normal(0.0, 1.0, size=n)   # keep SEX_1 free of the signal
    X[:, 2] = X[:, 0] + 0.5 * sex

    logit = 1.2 * X[:, 2] + 0.4 * X[:, 3] - 0.3 * X[:, 4] + 0.8 * sex
    p = 1.0 / (1.0 + np.exp(-logit))
    y01 = (rng.random(n) < p).astype(int)
    y = y01 + 1                  # dataset labels are 1/2, loader normalises to 0/1

    names = ["SEX_1", "SEX_2"] + [f"feat_{i}" for i in range(2, d)]
    out = Path(args.out) / "ABIDE_A"
    out.mkdir(parents=True, exist_ok=True)

    import csv
    with open(out / "processed_standard_data.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(names + ["label"])
        for row, label in zip(X, y):
            w.writerow([f"{v:.6f}" for v in row] + [int(label)])

    np.save(out / "modal_feat_dict.npy",
            {"synthetic_a": names[2:6], "synthetic_b": names[6:]}, allow_pickle=True)

    print(f"wrote {out/'processed_standard_data.csv'} n={n} d={d} "
          f"labels={np.bincount(y)} sex0={int((sex==0).sum())} sex1={int((sex==1).sum())}")


if __name__ == "__main__":
    main()
