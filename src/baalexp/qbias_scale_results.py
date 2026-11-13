from __future__ import annotations

"""Utilities for converting legacy v7 L1 Q-Bias result CSVs to TV Q-Bias.

Use this only when you want to inspect old results without rerunning.  For formal
manuscript results, rerun the experiments with the updated v8 code so that
strategy decisions themselves are made using TV-based Q-Bias.
"""

import argparse
from pathlib import Path

import pandas as pd


def convert_file(path: Path, out: Path | None = None, overwrite: bool = False) -> Path:
    df = pd.read_csv(path)
    if "q_bias" not in df.columns:
        raise ValueError(f"{path} has no q_bias column")
    # Avoid double conversion for v8 files.
    if "q_bias_definition" in df.columns and df["q_bias_definition"].astype(str).str.contains(r"TV=0.5\*L1", regex=True).any():
        target = out or path
        if target != path:
            df.to_csv(target, index=False)
        return target
    df["q_bias_l1_legacy"] = df["q_bias"]
    df["q_bias"] = df["q_bias"] / 2.0
    df["q_bias_definition"] = "TV=0.5*L1; converted_from_legacy_l1"
    target = out or path
    if target.exists() and target == path and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return target


def main() -> None:
    p = argparse.ArgumentParser(description="Convert legacy v7 L1 Q-Bias CSV files to TV Q-Bias")
    p.add_argument("paths", nargs="+", help="CSV files to convert")
    p.add_argument("--out-dir", default=None, help="Write converted files to this directory")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else None
    for raw in args.paths:
        path = Path(raw)
        out = None if out_dir is None else out_dir / path.name
        target = convert_file(path, out=out, overwrite=args.overwrite)
        print(f"{path} -> {target}")


if __name__ == "__main__":
    main()
