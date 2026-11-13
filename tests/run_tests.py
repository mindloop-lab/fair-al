#!/usr/bin/env python3
"""Run the package tests and report passed / skipped / failed counts.

Data-dependent tests skip (with a reason) when FAIRAL_DATA_DIR or FAIRAL_FROZEN_DIR is
absent.  A run with skips is NOT reported as a full regression pass.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG / "src"))

DATA_DEPENDENT = {
    "test_drop_sensitive_keeps_split_and_attribute",
    "test_reproduce_frozen_random_trajectory",
    "test_delayed_control_wiring",
    "test_intra_batch_runs_and_differs",
}


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def main() -> int:
    data_ok = os.path.isdir(os.environ.get("FAIRAL_DATA_DIR", ""))
    frozen_ok = os.path.isdir(os.environ.get("FAIRAL_FROZEN_DIR", ""))
    passed = skipped = failed = 0
    for module_name in ("test_rev1", "test_fal_cur_v2", "test_pairing_guards"):
        path = HERE / f"{module_name}.py"
        if not path.exists():
            print(f"SKIP module {module_name}: not present")
            skipped += 1
            continue
        mod = load(module_name)
        tests = [v for k, v in sorted(vars(mod).items()) if k.startswith("test_") and callable(v)]
        for fn in tests:
            needs = fn.__name__ in DATA_DEPENDENT
            if needs and not (data_ok and frozen_ok):
                print(f"SKIP {module_name}.{fn.__name__}: needs FAIRAL_DATA_DIR and FAIRAL_FROZEN_DIR")
                skipped += 1
                continue
            try:
                fn()
                print(f"PASS {module_name}.{fn.__name__}")
                passed += 1
            except FileNotFoundError as exc:
                print(f"SKIP {module_name}.{fn.__name__}: input not available ({exc})")
                skipped += 1
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {module_name}.{fn.__name__}: {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=2)
                failed += 1
    print(f"\npassed={passed} skipped={skipped} failed={failed}")
    if skipped:
        print("NOTE: skipped tests are not evidence of a passing regression.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
