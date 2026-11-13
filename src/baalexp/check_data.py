
from __future__ import annotations

import argparse

import numpy as np

from .data import binary_column_report, load_dataset
from .metrics import distribution


def parse_args():
    p = argparse.ArgumentParser(description="Check ABIDE data files and sensitive-attribute handling")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--sensitive-index", type=int, default=None, help="For ABIDE_B sex-proxy checks use 46.")
    p.add_argument("--scan-binary-cols", type=int, default=80, help="Show binary/low-cardinality columns in first N columns.")
    p.add_argument("--datasets", default="ABIDE_A,ABIDE_B")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for name in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        try:
            data = load_dataset(args.data_dir, name, sensitive_index=args.sensitive_index if name.upper()=="ABIDE_B" else None)
        except Exception as exc:
            print("=" * 70)
            print(name)
            print(f"FAILED to load sensitive attribute: {exc}")
            if name.upper() == "ABIDE_B" and args.sensitive_index is None:
                print("Hint: rerun with --sensitive-index 46 for the current sex-proxy experiments.")
            continue
        print("=" * 70)
        print(name)
        print("X:", data.X.shape, "y:", data.y.shape, "modal_dims:", data.modal_dims)
        print("label distribution:", {int(v): int((data.y == v).sum()) for v in np.unique(data.y)})
        print("sensitive:", data.sensitive_name, data.notes)
        print("sensitive distribution:", distribution(data.sensitive, 2))
        print("sensitive counts:", {int(v): int((data.sensitive == v).sum()) for v in np.unique(data.sensitive)})
        print(f"candidate low-cardinality columns in first {args.scan_binary_cols} features:")
        for row in binary_column_report(data.X, max_cols=args.scan_binary_cols):
            print("  ", row)


if __name__ == "__main__":
    main()

