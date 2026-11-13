
from __future__ import annotations

import argparse
import sys

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate FAIR-AL result consistency")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--expected-budget", type=int, default=None)
    p.add_argument("--expected-labeled-size", type=int, default=None)
    p.add_argument("--allow-incomplete", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    idx = df.groupby(["dataset", "backend", "classifier", "strategy", "seed"])["round"].idxmax()
    final = df.loc[idx].copy()
    problems: list[str] = []
    if args.expected_budget is not None:
        bad = final[final["queried_size"] != args.expected_budget]
        if not bad.empty:
            problems.append("queried_size mismatch:\n" + bad[["dataset", "backend", "classifier", "strategy", "seed", "queried_size"]].to_string(index=False))
    if args.expected_labeled_size is not None:
        bad = final[final["labeled_size"] != args.expected_labeled_size]
        if not bad.empty:
            problems.append("labeled_size mismatch:\n" + bad[["dataset", "backend", "classifier", "strategy", "seed", "labeled_size"]].to_string(index=False))
    print("Final rows by strategy:")
    print(final.groupby(["dataset", "backend", "classifier", "strategy"])[["queried_size", "labeled_size"]].agg(["min", "max", "mean"]))
    if problems:
        print("\nVALIDATION FAILED")
        print("\n\n".join(problems))
        if not args.allow_incomplete:
            sys.exit(1)
    print("\nVALIDATION PASSED")


if __name__ == "__main__":
    main()

