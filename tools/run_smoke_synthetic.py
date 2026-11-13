#!/usr/bin/env python3
"""Data-free smoke test v2 for the FAIR-AL code package.

Runs the real experiment entry point on synthetic data, then asserts - per acquisition
rule - the structural and containment properties the manuscript's protocol relies on:

  * one evaluation row per round plus the initial row
  * one history record per query event
  * Q-Bias within [0, 1]
  * constrained arm reports its fallback counters and reaches the budget
  * selections are unique across all rounds of an arm (not only within one batch)
  * every selected index is a valid row id (0 <= i < N)
  * every selected index lies in the synthetic training pool and in neither the initial
    labelled set nor the validation or test sets
  * the dropped-attribute ablation removes exactly the two group-encoding columns

Usage: python run_smoke_synthetic.py [--workdir <dir>]   (exit 0 on success)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
BUDGET, BATCH, INIT = 20, 10, 20
STRATEGIES = "uncertainty,baal_weighted,baal_hard,fal_cur,quota"


def run(cmd, env):
    print("$", " ".join(map(str, cmd)), flush=True)
    r = subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print((r.stdout or "")[-1200:]), print((r.stderr or "")[-2500:])
        raise SystemExit(f"command failed ({r.returncode})")
    return r


def check(cond, msg, tally):
    tally.append(bool(cond))
    print(("  ok   " if cond else "  FAIL ") + msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=str(HERE / "_smoke_work"))
    args = ap.parse_args()
    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PKG / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    run([sys.executable, HERE / "make_synthetic_data.py", "--out", wd / "data", "--n", "160", "--seed", "0"], env)
    common = ["--data-dir", wd / "data", "--dataset", "ABIDE_A", "--backend", "sklearn",
              "--classifier", "svm", "--seeds", "0", "--budget", BUDGET,
              "--batch-size", BATCH, "--initial-size", INIT,
              "--strategies", STRATEGIES, "--save-history"]
    run([sys.executable, "-m", "baalexp.run_experiment", *common, "--output", wd / "smoke.csv"], env)
    run([sys.executable, "-m", "baalexp.run_experiment", *common,
         "--drop-sensitive-inputs", "--output", wd / "smoke_drop.csv"], env)

    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(PKG / "src"))
    from baalexp.data import initial_labeled_split, load_dataset, train_val_test_split_indices

    df = pd.read_csv(wd / "smoke.csv")
    hist = json.loads((wd / "smoke.history.json").read_text())
    drop = pd.read_csv(wd / "smoke_drop.csv")
    n_rounds = BUDGET // BATCH
    tally: list[bool] = []

    per = df.groupby("strategy").size().to_dict()
    check(set(per.values()) == {n_rounds + 1}, f"each arm has {n_rounds + 1} evaluation points", tally)
    check(set(df.strategy) == set(STRATEGIES.split(",")), "all five acquisition rules ran", tally)
    check(bool(((df.q_bias >= -1e-12) & (df.q_bias <= 1 + 1e-12)).all()), "Q-Bias within [0, 1]", tally)
    check(int(df[df.strategy == "baal_hard"].queried_size.max()) == BUDGET,
          "constrained arm reaches the query budget", tally)
    nf = int(df.n_features.iloc[0]) - int(drop.n_features.iloc[0])
    check(nf == 2, f"dropped-attribute ablation removes exactly the 2 group encodings (got {nf})", tally)

    data = load_dataset(wd / "data", "ABIDE_A")
    tr, va, te = train_val_test_split_indices(data.y, data.sensitive, seed=0,
                                              test_size=0.2, val_size=0.1)
    init, _ = initial_labeled_split(tr, data.y, data.sensitive, initial_size=INIT, seed=0)
    pool, init_set, val_set, test_set = set(map(int, tr)), set(map(int, init)), \
        set(map(int, va)), set(map(int, te))
    n_rows = int(len(data.y))

    for strat in STRATEGIES.split(","):
        recs = [h for h in hist if h["strategy"] == strat]
        check(len(recs) == n_rounds, f"{strat}: one history record per query event ({n_rounds})", tally)
        sel = [int(i) for h in recs for i in h["selected_indices"]]
        check(len(sel) == len(set(sel)), f"{strat}: selections unique across all rounds", tally)
        check(all(0 <= i < n_rows for i in sel), f"{strat}: every index is a valid row id", tally)
        check(set(sel) <= pool, f"{strat}: every selection lies in the training pool", tally)
        check(not (set(sel) & (init_set | val_set | test_set)),
              f"{strat}: no selection repeats the initial set or touches val/test", tally)

    ev = [h for h in hist if h["strategy"] == "baal_hard"]
    check(all("constraint_dropped" in h and "feasible_fraction" in h for h in ev),
          "fallback counters present on every constrained event", tally)
    check([h["n_selected"] for h in ev] == [BATCH] * n_rounds,
          "exactly batch_size selected per event", tally)

    print(f"\nchecks passed: {sum(tally)}/{len(tally)}")
    if not all(tally):
        raise SystemExit("SMOKE FAILED")
    print("SMOKE OK: synthetic end-to-end run passed every structural and containment check.")


if __name__ == "__main__":
    main()
