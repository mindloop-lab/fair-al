"""Generate every formal revision table from the frozen/fresh CSVs (no hand-typed numbers).

Inputs
  results_as_submitted/enhanced/*            (v9: R1-A and R1-B, historical)
  results_as_submitted/*_core_*              (frozen core, used as the "kept" arm for S2-3)
  results_rev1/formal/**                     (new runs)
  results_rev1/stat/comparisons_manifest.csv (declared comparisons)

Outputs (results_rev1/tables/)
  manifest_stats.csv            every declared comparison with delta, CI, raw and corrected p
  T_<family>.md                 one compact markdown table per analysis family
  resolution_floor.csv          achieved k, nearest reachable k, min achievable Q-Bias
  process_qbias.csv             budget-axis integral per arm and seed
  fallback_rates.csv            per-query constraint fallback and feasibility rates (history)
  lambda_overlap.csv            per-seed query-set overlap between lambda arms and the default
  REVISION_TABLES.md            index of everything above

Only CSV inputs are read; every number here is reproducible by rerunning this script.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CODE_ROOT = Path(os.environ.get("FAIRAL_CODE_ROOT", Path(__file__).resolve().parents[1]))
ROOT = Path(os.environ.get("FAIRAL_PROJECT_ROOT",
                           CODE_ROOT.parent if CODE_ROOT.name == "code" else CODE_ROOT.parents[1]))
REV = Path(os.environ.get("FAIRAL_REV_DIR", CODE_ROOT / "results_rev1"))
FORMAL = REV / "formal"
TBL = REV / "tables"
FROZEN = ROOT / "results_as_submitted"
sys.path.insert(0, str(CODE_ROOT / "src"))

from baalexp.stats_rev1 import process_qbias_stats  # noqa: E402

BUDGET = 120.0


def ci95(diff: np.ndarray) -> tuple[float, float]:
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n < 2:
        return (float("nan"), float("nan"))
    half = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)
    return (float(diff.mean() - half), float(diff.mean() + half))


def load_all() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(f"{FROZEN}/*.csv") + glob.glob(f"{FROZEN}/enhanced/*.csv")
                       + glob.glob(f"{FORMAL}/**/*.csv", recursive=True)):
        if "regression" in path or os.path.basename(path).startswith("comparisons"):
            continue
        try:
            d = pd.read_csv(path)
        except Exception:
            continue
        if not {"dataset", "classifier", "strategy", "seed", "round"}.issubset(d.columns):
            continue
        d["_source"] = os.path.basename(path)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["classifier"] = df["classifier"].astype(str).str.lower()
    return df


def stats_tables(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate every comparison in the manifest (same logic as stats_rev1 manifest mode)."""
    from baalexp.stats_rev1 import paired, holm, bh_fdr  # local import

    man = pd.read_csv(f"{REV}/stat/comparisons_manifest.csv")
    rows = []
    for _, m in man.iterrows():
        sub = df[(df.dataset == m.dataset) & (df.classifier == m.classifier)]
        if sub.empty:
            rows.append({**m.to_dict(), "status": "missing_setting"})
            continue
        metric = m.metric
        first = sub[sub["round"] == sub.groupby(["strategy", "seed"])["round"].transform("max")]

        def side(strategy, src):
            # sources are recorded as repo-relative paths in the manifest but only the
            # basename is kept on the loaded frame
            s = first if pd.isna(src) else first[first["_source"] == os.path.basename(str(src))]
            s = s[s["strategy"] == strategy]
            if s.empty:
                return None
            return s.set_index("seed")[metric]

        sa, sb = side(m.strategy_a, m.source_a), side(m.strategy_b, m.source_b)
        if sa is None or sb is None:
            rows.append({**m.to_dict(), "status": "missing_arm"})
            continue
        common = sa.index.intersection(sb.index)
        if len(common) == 0:
            rows.append({**m.to_dict(), "status": "no_common_seeds"})
            continue
        a, b = sa.loc[common].to_numpy(float), sb.loc[common].to_numpy(float)
        r = paired(a, b)
        lo, hi = ci95(a - b)
        rows.append({**m.to_dict(), "status": "ok", "n_pairs": r.n,
                     "mean_a": r.mean_a, "mean_b": r.mean_b, "delta": r.delta,
                     "ci95_low": lo, "ci95_high": hi, "sd_diff": r.sd,
                     "cohen_dz": r.dz, "paired_t": r.t, "p_raw": r.p, "note": r.note})
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["p_adjusted"] = np.nan
    out["family_size_evaluated"] = 0
    out["rank_within_family"] = 0
    for fam, g in out.groupby("family_id"):
        method = str(g["adjustment"].iloc[0]).lower()
        mask = g["p_raw"].notna().to_numpy()
        idx = g.index[mask]                      # positional-safe index labels
        p = g.loc[idx, "p_raw"].to_numpy(float)
        if not len(idx) or method == "none":
            continue
        adj = holm(p) if method == "holm" else bh_fdr(p)
        order = np.argsort(p)
        ranks = np.empty(len(p), dtype=int)
        ranks[order] = np.arange(1, len(p) + 1)
        out.loc[idx, "p_adjusted"] = adj
        out.loc[idx, "rank_within_family"] = ranks
        out.loc[idx, "family_size_evaluated"] = len(p)
    out.to_csv(f"{TBL}/manifest_stats.csv", index=False)
    return out


def family_markdown(st: pd.DataFrame) -> None:
    groups = {
        "R1A": "R1-A strong composition baselines (historical v9, retrospective analysis)",
        "R1B": "R1-B target-distribution sensitivity (descriptive: different references, not one scale)",
        "S23": "S2-3 sensitive-attribute removal from the model input (kept = frozen submission)",
        "S24": "S2-4 batch-prior vs intra-batch FAIR-AL (historical vs new variant)",
        "R21LAM": "R2-1 lambda sensitivity vs the pre-set default 0.5",
        "R21TAU": "R2-1 tau sensitivity vs the pre-set default 0.03 (fallback rate reported separately)",
        "R22": "R2-2 external fair-AL comparison: FAL-CUR (our implementation of the published algorithm)",
        "R1Ai": "R1-A follow-up (post hoc): intra-batch FAIR-AL vs composition controls at equal budget",
        "R23": "R2-3 delayed vs continuous drift control",
    }
    for key, title in groups.items():
        sub = st[st["family_id"].str.startswith(f"FAM_{key}|")]
        if sub.empty:
            continue
        lines = [f"# {title}", "",
                 "| family | metric | contrast | mean_a | mean_b | delta | 95% CI | p_raw | p_adj | m | note |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        sub = sub.assign(contrast=sub["strategy_a"] + " - " + sub["strategy_b"])
        for _, r in sub.sort_values(["family_id", "contrast"]).iterrows():
            ci = (f"[{r.ci95_low:+.4f}, {r.ci95_high:+.4f}]"
                  if pd.notna(r.get("ci95_low")) else "")
            lines.append(
                f"| {r.family_id} | {r.metric} | {r.strategy_a} - {r.strategy_b} | "
                f"{r.mean_a:.4f} | {r.mean_b:.4f} | {r.delta:+.4f} | {ci} | "
                f"{'' if pd.isna(r.p_raw) else f'{r.p_raw:.4g}'} | "
                f"{'' if pd.isna(r.p_adjusted) else f'{r.p_adjusted:.4g}'} | "
                f"{int(r.family_size_evaluated)} | {r.note} |")
        open(f"{TBL}/T_{key}.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")


def resolution(df: pd.DataFrame) -> None:
    rows = []
    for (ds, clf), g in df[(df.dataset == "ABIDE_A") | (df.dataset == "ABIDE_B")].groupby(["dataset", "classifier"]):
        last = g[g["round"] == g.groupby(["strategy", "seed"])["round"].transform("max")]
        for seed, gs in last.groupby("seed"):
            pi0 = float(gs["target_p_group0"].iloc[0])
            n = int(gs["queried_size"].iloc[0])
            if n <= 0:
                continue
            ks = np.arange(0, n + 1)
            q = np.abs(ks / n - pi0)
            k_best, q_best = int(ks[int(np.argmin(q))]), float(q.min())
            for _, r in gs.iterrows():
                k = int(round(float(r["query_p_group0"]) * n)) if pd.notna(r["query_p_group0"]) else -1
                rows.append(dict(dataset=ds, classifier=clf, seed=int(seed), arm=r["strategy"],
                                 source=r["_source"], pi0=round(pi0, 6), n=n, k_minority=k,
                                 k_nearest=k_best, sample_gap=abs(k - k_best),
                                 q_bias=round(float(r["q_bias"]), 6),
                                 q_best_possible=round(q_best, 6)))
    out = pd.DataFrame(rows)
    out.to_csv(f"{TBL}/resolution_floor.csv", index=False)


def process_qbias(df: pd.DataFrame) -> None:
    rows = []
    for src, g in df.groupby("_source"):
        for (ds, clf, strat), gg in g.groupby(["dataset", "classifier", "strategy"]):
            for seed, s in gg.groupby("seed"):
                st = process_qbias_stats(s, strat, BUDGET)
                rows.append(dict(source=src, dataset=ds, classifier=clf, arm=strat, seed=int(seed),
                                 i_main=st["i_main"], i_from_zero=st["i_from_zero"],
                                 u_start=st["u_start"], u_end=st["u_end"], n_points=st["n_points"]))
    out = pd.DataFrame(rows)
    out.to_csv(f"{TBL}/process_qbias.csv", index=False)


def fallback_rates() -> None:
    rows = []
    for path in sorted(glob.glob(f"{FORMAL}/**/*.history.json", recursive=True) +
                       glob.glob(f"{FROZEN}/*.history.json")):
        try:
            h = json.load(open(path))
        except Exception:
            continue
        dfh = pd.DataFrame(h)
        if dfh.empty or "strategy" not in dfh:
            continue
        for (ds, strat), g in dfh.groupby(["dataset", "strategy"]):
            n_query = len(g)
            dropped = int(g["constraint_dropped"].sum()) if "constraint_dropped" in g else 0
            rows.append(dict(source=os.path.basename(path), dataset=ds, arm=strat,
                             n_query_events=n_query, fallback_query_events=dropped,
                             fallback_rate=dropped / n_query if n_query else np.nan,
                             feasible_frac_mean=(float(g["feasible_fraction"].mean())
                                                 if "feasible_fraction" in g else np.nan)))
    pd.DataFrame(rows).to_csv(f"{TBL}/fallback_rates.csv", index=False)


def lambda_overlap() -> None:
    rows = []
    for f in sorted(glob.glob(f"{FORMAL}/lambda_tau/*_lambda_*.csv")):
        base = os.path.basename(f).replace(".csv", "")
        ds, clf = base.split("_lambda_")[0].rsplit("_", 1)
        lam = base.split("_lambda_")[1]
        default = f.replace(f"_lambda_{lam}.csv", "_lambda_0.5.csv")
        if not os.path.exists(default) or lam == "0.5":
            continue
        hp = f.replace(".csv", ".history.json")
        dp = default.replace(".csv", ".history.json")
        if not (os.path.exists(hp) and os.path.exists(dp)):
            continue
        h = pd.DataFrame(json.load(open(hp)))
        d = pd.DataFrame(json.load(open(dp)))
        for seed in sorted(set(h["seed"]) & set(d["seed"])):
            hs = h[(h.seed == seed) & (h.strategy == "baal_weighted")].sort_values("query_round")
            dsr = d[(d.seed == seed) & (d.strategy == "baal_weighted")].sort_values("query_round")
            inter = union = 0
            for (_, a), (_, b) in zip(hs.iterrows(), dsr.iterrows()):
                sa, sb = set(a["selected_indices"]), set(b["selected_indices"])
                inter += len(sa & sb)
                union += len(sa | sb)
            rows.append(dict(dataset=ds, classifier=clf, seed=int(seed), lambda_value=lam,
                             jaccard=inter / union if union else np.nan,
                             n_query_events=len(hs)))
    pd.DataFrame(rows).to_csv(f"{TBL}/lambda_overlap.csv", index=False)


def index() -> None:
    files = sorted(os.path.basename(p) for p in glob.glob(f"{TBL}/*"))
    lines = ["# Revision tables (generated by scripts/26_make_revision_tables.py)", "",
             "All numbers come from the CSVs; nothing is hand-copied.", "",
             "| file | rows |", "|---|---|"]
    for f in files:
        p = os.path.join(TBL, f)
        if f.endswith(".csv"):
            try:
                n = len(pd.read_csv(p))
            except Exception:
                n = 0
            lines.append(f"| {f} | {n} |")
    open(f"{TBL}/REVISION_TABLES.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main() -> None:
    os.makedirs(TBL, exist_ok=True)
    df = load_all()
    print("loaded", len(df), "rows from", df["_source"].nunique(), "files")
    st = stats_tables(df)
    family_markdown(st)
    resolution(df)
    process_qbias(df)
    fallback_rates()
    lambda_overlap()
    index()
    print(st["status"].value_counts().to_dict())
    print("wrote tables to", TBL)


if __name__ == "__main__":
    main()
