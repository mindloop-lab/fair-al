"""Strict pairing and comparison core for the FAIR-AL revision tables.

This module exists because the first revision builder silently accepted incomplete
inputs: it took the maximum round across files, joined on basenames, and intersected
whatever seeds happened to be present.  Every function here fails loudly instead.

Contract
--------
* A source is selected by its canonical repository-relative path and identified by its
  SHA-256.  A bare basename is accepted only when it resolves to exactly one file.
* An endpoint is the declared query budget.  An arm that never reaches the budget is an
  error, not a reason to fall back to its last evaluation point.
* The seed set is declared per analysis.  Missing, extra or duplicated seeds are errors.
* Paired statistics are computed on a one-to-one seed join.
* A multiple-comparison method must be one of the declared ones; anything else is an
  error.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-12
METHODS = {"holm", "bh", "none"}


# --------------------------------------------------------------------------- errors
class PairingError(RuntimeError):
    """Base class: any condition that must not be worked around silently."""


class SourceNotFound(PairingError):
    pass


class SourceAmbiguous(PairingError):
    pass


class MissingEndpoint(PairingError):
    pass


class DuplicateEndpoint(PairingError):
    pass


class SeedCoverageError(PairingError):
    pass


class UnknownAdjustment(PairingError):
    pass


# --------------------------------------------------------------------------- sources
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Source:
    path: Path
    rel_path: str
    sha256: str


def resolve_source(spec: str, code_root: Path, project_root: Path) -> Source:
    """Resolve one declared source to exactly one file.

    Order: path relative to the code tree, then relative to the project root, then a
    basename lookup in the two frozen result directories - and that last one only if it
    is unique.  A basename that matches two files is an error, never a silent choice.
    """
    spec = str(spec).strip()
    if not spec:
        raise SourceNotFound("empty source specification")
    candidates = [code_root / spec, project_root / spec]
    for cand in candidates:
        if cand.is_file():
            rel = os.path.relpath(cand.resolve(), project_root.resolve())
            return Source(cand.resolve(), rel, sha256_file(cand))
    base = os.path.basename(spec)
    hits = [p for p in (project_root / "results_as_submitted" / base,
                        project_root / "results_as_submitted/enhanced" / base)
            if p.is_file()]
    if not hits:
        # last resort: a unique match anywhere in the revision result tree
        for tree in (code_root / "results_rev1", code_root / "results"):
            if tree.is_dir():
                hits.extend(p for p in tree.rglob(base) if p.is_file())
    if len(hits) == 1:
        rel = os.path.relpath(hits[0].resolve(), project_root.resolve())
        return Source(hits[0].resolve(), rel, sha256_file(hits[0]))
    if len(hits) > 1:
        raise SourceAmbiguous(
            f"basename {base!r} matches {len(hits)} files: "
            + ", ".join(str(h) for h in sorted(hits)[:6]) + " - declare the full path")
    raise SourceNotFound(f"source not found: {spec!r} (tried {candidates[0]}, {candidates[1]})")


# --------------------------------------------------------------------------- endpoints
def endpoint_series(df: pd.DataFrame, strategy: str, metric: str, budget: float) -> pd.Series:
    """Metric values at exactly queried_size == budget, indexed by seed, one row per seed."""
    if metric not in df.columns:
        raise MissingEndpoint(f"metric column {metric!r} absent")
    s = df[df["strategy"] == strategy]
    if s.empty:
        raise MissingEndpoint(f"strategy {strategy!r} absent from the source")
    at = s[np.isclose(s["queried_size"].to_numpy(float), float(budget))]
    if at.empty:
        actual = float(s["queried_size"].max())
        raise MissingEndpoint(
            f"strategy {strategy!r} never reaches the declared budget {budget:g}"
            f" (max queried_size = {actual:g}); refusing to fall back to an earlier endpoint")
    if at["seed"].duplicated().any():
        dup = sorted(at.loc[at["seed"].duplicated(), "seed"].unique().tolist())
        raise DuplicateEndpoint(
            f"strategy {strategy!r} has {int(at['seed'].duplicated().sum())} duplicated seed(s) "
            f"at the budget endpoint: {dup} - refusing to average duplicates")
    values = at.set_index("seed")[metric].astype(float)
    if not np.isfinite(values).all():
        bad_seeds = values.index[~np.isfinite(values)].tolist()
        raise MissingEndpoint(
            f"strategy {strategy!r} has nonfinite {metric} at budget {budget:g}"
            f" for seeds {bad_seeds}")
    return values


def check_seed_coverage(index: pd.Index, expected_seeds: list[int], label: str) -> None:
    got = set(int(i) for i in index)
    want = set(int(s) for s in expected_seeds)
    missing = sorted(want - got)
    extra = sorted(got - want)
    if missing or extra:
        raise SeedCoverageError(
            f"{label}: seed coverage mismatch; missing={missing} unexpected={extra} "
            f"(declared {min(want)}..{max(want)}, n={len(want)})")


# --------------------------------------------------------------------------- statistics
def paired_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired t on aligned arrays; zero variance is reported, never turned into t=inf."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = int(len(a))
    if n == 0:
        return dict(n=0, mean_a=np.nan, mean_b=np.nan, delta=np.nan, sd=np.nan,
                    t=None, p=None, dz=np.nan, ci95_low=np.nan, ci95_high=np.nan,
                    note="no_pairs")
    delta = float((a - b).mean())
    if n < 2:
        return dict(n=n, mean_a=float(a.mean()), mean_b=float(b.mean()), delta=delta,
                    sd=np.nan, t=None, p=None, dz=np.nan, ci95_low=np.nan,
                    ci95_high=np.nan, note="too_few_pairs")
    sd = float((a - b).std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    if sd <= EPS:
        note = "identical_arms" if abs(delta) <= EPS else "degenerate_zero_variance"
        return dict(n=n, mean_a=float(a.mean()), mean_b=float(b.mean()), delta=delta, sd=sd,
                    t=None, p=None, dz=np.nan, ci95_low=delta, ci95_high=delta, note=note)
    t, p = stats.ttest_rel(a, b)
    return dict(n=n, mean_a=float(a.mean()), mean_b=float(b.mean()), delta=delta, sd=sd,
                t=float(t), p=float(p), dz=delta / sd, ci95_low=delta - half,
                ci95_high=delta + half, note="ok")


def holm(p) -> np.ndarray:
    p = np.asarray(list(p) if not isinstance(p, np.ndarray) else p, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    raw = (n - np.arange(1, n + 1) + 1.0) * p[order]
    adj = np.maximum.accumulate(raw)
    out = np.empty(n)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out


def bh_fdr(p) -> np.ndarray:
    p = np.asarray(list(p) if not isinstance(p, np.ndarray) else p, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranks = np.arange(1, n + 1, dtype=float)
    raw = p[order] * n / ranks
    adj = np.minimum.accumulate(raw[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out


def adjust(p_values, method: str) -> np.ndarray:
    m = str(method).strip().lower()
    if m not in METHODS:
        raise UnknownAdjustment(f"unknown multiple-comparison method {method!r}; "
                                f"declared methods are {sorted(METHODS)}")
    if m == "none":
        return np.asarray(list(p_values), float)
    return holm(p_values) if m == "holm" else bh_fdr(p_values)


# --------------------------------------------------------------------------- one comparison
def run_comparison(spec: dict, code_root: Path, project_root: Path,
                   cache: dict | None = None) -> tuple[dict, pd.DataFrame]:
    """Run one declared comparison. Returns the summary row and the per-seed pairs."""
    cache = {} if cache is None else cache
    budget = float(spec.get("budget", 120))
    method = str(spec.get("adjustment", "none")).strip().lower()
    if method not in METHODS:
        raise UnknownAdjustment(
            f"{spec.get('analysis_id')}: undeclared multiple-comparison method "
            f"{spec.get('adjustment')!r}; declared methods are {sorted(METHODS)}")
    expected = [int(s) for s in str(spec.get("expected_seeds", "0,1,2,3,4,5,6,7,8,9")).split(",")]

    def table(src_spec: str) -> pd.DataFrame:
        src = resolve_source(src_spec, code_root, project_root)
        if src.sha256 not in cache:
            cache[src.sha256] = pd.read_csv(src.path)
        return src, cache[src.sha256]

    src_a, df_a = table(spec["source_a"])
    src_b, df_b = table(spec["source_b"])
    a = endpoint_series(df_a, spec["strategy_a"], spec["metric"], budget)
    b = endpoint_series(df_b, spec["strategy_b"], spec["metric"], budget)
    check_seed_coverage(a.index, expected, f"{spec['analysis_id']} side A ({spec['strategy_a']})")
    check_seed_coverage(b.index, expected, f"{spec['analysis_id']} side B ({spec['strategy_b']})")

    common = sorted(set(map(int, a.index)) & set(map(int, b.index)))
    if sorted(map(int, a.index)) != common or sorted(map(int, b.index)) != common:
        raise PairingError(f"{spec['analysis_id']}: seed sets differ after coverage check")
    st = paired_stats(a.loc[common].to_numpy(), b.loc[common].to_numpy())

    row = {
        **{k: spec.get(k) for k in ("analysis_id", "dataset", "classifier", "metric",
                                    "strategy_a", "strategy_b", "family_id", "adjustment",
                                    "analysis_status")},
        "budget": budget,
        "expected_seeds": ",".join(map(str, expected)),
        "source_a": src_a.rel_path, "source_a_sha256": src_a.sha256,
        "source_b": src_b.rel_path, "source_b_sha256": src_b.sha256,
        "n_pairs": st["n"], "mean_a": st["mean_a"], "mean_b": st["mean_b"],
        "delta": st["delta"], "sd_diff": st["sd"], "paired_t": st["t"], "p_raw": st["p"],
        "cohen_dz": st["dz"], "ci95_low": st["ci95_low"], "ci95_high": st["ci95_high"],
        "note": st["note"], "seeds_paired": ",".join(map(str, common)),
        "status": "ok",
    }
    pairs = pd.DataFrame({
        "analysis_id": spec["analysis_id"], "dataset": spec["dataset"],
        "classifier": spec["classifier"], "metric": spec["metric"],
        "arm_a": spec["strategy_a"], "arm_b": spec["strategy_b"],
        "source_a": src_a.rel_path, "source_a_sha256": src_a.sha256,
        "source_b": src_b.rel_path, "source_b_sha256": src_b.sha256,
        "budget": budget, "seed": common,
        "value_a": a.loc[common].to_numpy(float), "value_b": b.loc[common].to_numpy(float),
    })
    pairs["delta"] = pairs.value_a - pairs.value_b
    pairs["split_proof_grade"] = spec.get("split_proof_grade", "reconstructed_protocol")
    return row, pairs


def family_ledger(rows: pd.DataFrame) -> pd.DataFrame:
    """Planned vs evaluated counts, method and zero-variance handling per family."""
    out = []
    for fam, g in rows.groupby("family_id"):
        declared = str(g["adjustment"].iloc[0])
        testable = int(g["p_raw"].notna().sum())
        out.append({
            "family_id": fam, "declared_adjustment": declared,
            "planned_comparisons": int(len(g)), "evaluated_comparisons": int(len(g)),
            "testable_comparisons": testable,
            "zero_variance_pairs": int((g["note"] != "ok").sum()),
            "p_adjusted_filled": int(g["p_adjusted"].notna().sum()),
            "analysis_status": ";".join(sorted(set(g["analysis_status"].dropna().astype(str)))),
        })
    return pd.DataFrame(out)
