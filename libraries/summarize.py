"""Aggregate experiment results across seeds and algorithms into summary CSV files.

Execution:
    python -m libraries.summarize

Inputs:
    outputs/results/<algo>_<tag>_seed<k>.json
    private/logs/<algo>/<tag>_seed<k>.csv
    private/logs/<algo>/<tag>_seed<k>_eval.csv
    outputs/results/baseline_random.csv (if available)

Outputs:
    outputs/results/summary.csv
    outputs/results/per_seed.csv
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from .utils import DEFAULT_LOG_DIR, DEFAULT_RESULTS_DIR, ensure_dir, trailing_ma

SOLVE_THRESHOLD = 475.0
MA_WINDOW = 100
RECENT_EVAL_FRACTION = 0.2
DEFAULT_SUMMARY_CSV = f"{DEFAULT_RESULTS_DIR}/summary.csv"
DEFAULT_PER_SEED_CSV = f"{DEFAULT_RESULTS_DIR}/per_seed.csv"


def first_reach(steps, values, thr=SOLVE_THRESHOLD):
    """Find the first step count where metric values reach or exceed a given threshold.

    Input:  steps  -- iterable of environment step numbers;
            values -- iterable of corresponding metric values;
            thr    -- numerical threshold to cross (default SOLVE_THRESHOLD = 475.0).
    Output: float step count where values >= thr, or np.nan if never reached.
    Flow:   iterate through (steps, values) pairs and return the first step where values >= thr.
    """
    for s, v in zip(steps, values):
        if v >= thr:
            return float(s)
    return np.nan


def per_seed_metrics(res, thr=SOLVE_THRESHOLD, ma_window=MA_WINDOW,
                     recent_fraction=RECENT_EVAL_FRACTION):
    """Compute individual seed performance metrics from result JSON and log files.

    Input:  res             -- result dictionary loaded from outputs/results/<algo>_<tag>_seed<k>.json;
            thr             -- target solve score threshold (default SOLVE_THRESHOLD = 475.0);
            ma_window       -- moving average window size (default MA_WINDOW = 100);
            recent_fraction -- fraction of late evaluation rounds to assess stability (default 0.2).
    Output: dictionary containing before_mean, best_mean, best_std, best_min,
            best_frac500, final_mean, steps_to_ma100_475, steps_to_eval_475,
            stability_std, drop_after_peak.
    Flow:   extract fields from result dictionary, read training and evaluation CSV logs,
            compute trailing_ma(ma_window) -> steps_to_ma100_475,
            calculate late-stage evaluation stability standard deviation, return dictionary.
    """
    algo, tag, seed = res["algo"], res["tag"], res["seed"]
    m = {
        "algo": algo,
        "tag": tag,
        "seed": seed,
        "before_mean": res["before"]["mean"],
        "best_mean": res["after_best"]["mean"],
        "best_std": res["after_best"]["std"],
        "best_min": res["after_best"]["min"],
        "best_frac500": res["after_best"]["frac_500"],
        "final_mean": res["after_final"]["mean"],
    }
    tr = f"{DEFAULT_LOG_DIR}/{algo}/{tag}_seed{seed}.csv"
    ev = f"{DEFAULT_LOG_DIR}/{algo}/{tag}_seed{seed}_eval.csv"
    m["steps_to_ma100_475"] = m["steps_to_eval_475"] = m["stability_std"] = m["drop_after_peak"] = np.nan
    if os.path.exists(tr):
        d = pd.read_csv(tr)
        if len(d):
            m["steps_to_ma100_475"] = first_reach(d["env_step"], trailing_ma(d["return"], ma_window), thr)
    if os.path.exists(ev):
        e = pd.read_csv(ev)
        if len(e):
            m["steps_to_eval_475"] = first_reach(e["env_step"], e["eval_mean"], thr)
            k = max(2, int(np.ceil(recent_fraction * len(e))))
            m["stability_std"] = float(e["eval_mean"].iloc[-k:].std(ddof=0))
            m["drop_after_peak"] = float(e["eval_mean"].max() - e["eval_mean"].iloc[-1])
    return m


def summarize_results(summary_csv=DEFAULT_SUMMARY_CSV,
                      per_seed_csv=DEFAULT_PER_SEED_CSV,
                      thr=SOLVE_THRESHOLD):
    """Aggregate all experiment results into summary DataFrames and persist to CSV.

    Input:  summary_csv  -- destination path for aggregated group summary CSV;
            per_seed_csv -- destination path for detailed per-seed CSV;
            thr          -- score threshold to classify an episode as solved (default SOLVE_THRESHOLD).
    Output: DataFrame containing aggregated summary metrics across runs.
    Flow:   find all result JSON files, call per_seed_metrics on each,
            group by (algo, tag) to calculate summary statistics,
            append baseline runs (random, heuristic), write CSV files, return DataFrame.
    """
    ensure_dir(DEFAULT_RESULTS_DIR)
    files = sorted(glob.glob(f"{DEFAULT_RESULTS_DIR}/*_seed*.json")) or sorted(glob.glob("results/*_seed*.json"))
    per_seed = [per_seed_metrics(json.load(open(f, encoding="utf-8")), thr=thr) for f in files]
    rows = []
    if per_seed:
        df = pd.DataFrame(per_seed)
        df.to_csv(per_seed_csv, index=False)
        for (algo, tag), g in df.groupby(["algo", "tag"]):
            rows.append({
                "algo": algo,
                "tag": tag,
                "n_seeds": len(g),
                "before_mean": g["before_mean"].mean(),
                "after_best_mean": g["best_mean"].mean(),
                "after_best_std_across_seeds": g["best_mean"].std(ddof=0),
                "after_best_within_std": g["best_std"].mean(),
                "after_best_min_seed": g["best_mean"].min(),
                "frac500": g["best_frac500"].mean(),
                "after_final_mean": g["final_mean"].mean(),
                "n_seeds_ge475": int((g["best_mean"] >= thr).sum()),
                "steps_to_eval_475_median": g["steps_to_eval_475"].median(),
                "n_seeds_reached_eval_475": int(g["steps_to_eval_475"].notna().sum()),
                "steps_to_ma100_475_median": g["steps_to_ma100_475"].median(),
                "stability_std_mean": g["stability_std"].mean(),
                "drop_after_peak_mean": g["drop_after_peak"].mean(),
            })
    for name in ("heuristic", "random"):
        bl = f"{DEFAULT_RESULTS_DIR}/baseline_{name}.csv" if os.path.exists(f"{DEFAULT_RESULTS_DIR}/baseline_{name}.csv") else f"results/baseline_{name}.csv"
        if os.path.exists(bl):
            b = pd.read_csv(bl)["return"]
            rows.insert(0, {
                "algo": name,
                "tag": "-",
                "n_seeds": b.size // 100,
                "after_best_mean": b.mean(),
                "after_best_within_std": b.std(ddof=0),
                "after_best_min_seed": b.min(),
                "frac500": (b >= 500).mean()
            })
    out = pd.DataFrame(rows)
    out.to_csv(summary_csv, index=False)
    return out


def main():
    """CLI entrypoint: generate result summaries and print aggregated table.

    Input:  None.
    Output: None (prints tabular summary to stdout).
    Flow:   call summarize_results(), display formatted results.
    """
    out = summarize_results()
    print(out.round(2).to_string(index=False))
    print(f"\nSaved {DEFAULT_SUMMARY_CSV}, {DEFAULT_PER_SEED_CSV}")


if __name__ == "__main__":
    main()
