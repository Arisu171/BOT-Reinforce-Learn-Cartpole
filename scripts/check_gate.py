#!/usr/bin/env python3
"""Phase acceptance gate checks.

Execution from project root:
    python scripts/check_gate.py <phase>        phase = 0..6 (or: all)

Exits with code 0 if all items PASS, code 1 if any FAIL.
"""
import csv
import glob
import importlib.util
import json
import os
import re
import sys

SEEDS = [0, 1, 2, 3, 4]
results = []


def check(name, ok, detail=""):
    """Record verification check outcome and display status line.

    Input:  name   -- name or description of the gate requirement;
            ok     -- boolean verification condition;
            detail -- optional contextual detail string.
    Output: None (appends boolean to results list and prints result).
    Flow:   append ok to results, format PASS or FAIL with detail, print to stdout.
    """
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))


def exists(path):
    """Verify that a path exists and points to a non-empty file.

    Input:  path -- filesystem path string.
    Output: boolean indicating file exists and size > 0.
    Flow:   evaluate os.path.exists and os.path.getsize.
    """
    return os.path.exists(path) and os.path.getsize(path) > 0


def read_csv(path):
    """Read a CSV file and return a list of row dictionaries.

    Input:  path -- path to CSV file.
    Output: list of dictionaries representing CSV rows.
    Flow:   open file with UTF-8 encoding, pass to csv.DictReader, return list.
    """
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_results(algo, tag="base"):
    """Load JSON result artifacts for all seeds of a given algorithm and tag.

    Input:  algo -- algorithm identifier ('dqn', 'reinforce');
            tag  -- experiment tag (default 'base').
    Output: dictionary mapping seed integer to loaded JSON object.
    Flow:   iterate through SEEDS, check outputs/results/ then results/, load JSON into dict.
    """
    out = {}
    for s in SEEDS:
        p = f"outputs/results/{algo}_{tag}_seed{s}.json" if exists(f"outputs/results/{algo}_{tag}_seed{s}.json") else f"results/{algo}_{tag}_seed{s}.json"
        if exists(p):
            out[s] = json.load(open(p, encoding="utf-8"))
    return out


def phase0():
    """Verify Phase 0 environment setup and shared library dependencies.

    Input:  None.
    Output: None (executes check assertions for Phase 0).
    Flow:   verify core libraries exist, requirements.txt, PROGRESS.md, and essential Python packages.
    """
    for f in ["utils", "envs", "evaluate", "policies", "baseline"]:
        check(f"libraries/{f}.py exists", exists(f"libraries/{f}.py"))
    check("requirements.txt", exists("requirements.txt"))
    check("PROGRESS.md", exists("PROGRESS.md") or exists("private/arcs/PROGRESS.md"))
    for mod in ["gymnasium", "numpy", "matplotlib"]:
        check(f"installed {mod}", importlib.util.find_spec(mod) is not None)
    check("installed torch", importlib.util.find_spec("torch") is not None)


def phase1():
    """Verify Phase 1 random baseline results, histogram figure, and simulation GIF.

    Input:  None.
    Output: None (executes check assertions for Phase 1).
    Flow:   check baseline_random.csv exists with >= 100 episodes, check random_hist.png and random.gif.
    """
    p = "outputs/results/baseline_random.csv" if exists("outputs/results/baseline_random.csv") else "results/baseline_random.csv"
    ok = exists(p)
    check(p, ok)
    if ok:
        rows = read_csv(p)
        rets = [float(r["return"]) for r in rows]
        mean = sum(rets) / len(rets)
        check("has >= 100 episodes", len(rets) >= 100, f"n={len(rets)}")
        check("mean random return in [10, 40]", 10 <= mean <= 40, f"mean={mean:.1f}")
    hist_p = "outputs/figures/random_hist.png" if exists("outputs/figures/random_hist.png") else "figures/random_hist.png"
    check(hist_p, exists(hist_p))
    vid_p = "outputs/videos/random.gif" if exists("outputs/videos/random.gif") else "videos/random.gif"
    check(vid_p, exists(vid_p))


def algo_phase(algo, thr, min_pass):
    """Verify results, training logs, and saved checkpoints for a given algorithm.

    Input:  algo     -- algorithm name ('dqn', 'reinforce');
            thr      -- score threshold for considering an agent solved;
            min_pass -- minimum number of passing seeds required.
    Output: None (executes check assertions).
    Flow:   verify result JSONs, CSV logs (>= 50 rows), checkpoint files, and mean return improvements.
    """
    res = load_results(algo)
    res_label = "outputs/results/" if any(exists(f"outputs/results/{algo}_base_seed{s}.json") for s in SEEDS) else "results/"
    check(f"all 5 files {res_label}{algo}_base_seed*.json exist", len(res) == 5, f"found {len(res)}/5")
    for s in SEEDS:
        log_p = (
            f"private/logs/{algo}/base_seed{s}.csv"
            if exists(f"private/logs/{algo}/base_seed{s}.csv")
            else f"logs/{algo}/base_seed{s}.csv"
        )
        check(f"{log_p} has >= 50 rows",
              exists(log_p) and len(read_csv(log_p)) >= 50)
        check(f"models/{algo}_base_seed{s}_best.pt exists", exists(f"models/{algo}_base_seed{s}_best.pt"))
    if res:
        n_ok = sum(1 for r in res.values() if r["after_best"]["mean"] >= thr)
        check(f">= {min_pass}/5 seeds achieve after_best.mean >= {thr}", n_ok >= min_pass,
              "; ".join(f"seed{s}: {r['before']['mean']:.0f}->{r['after_best']['mean']:.0f}" for s, r in sorted(res.items())))
        check("post-training mean exceeds pre-training across all seeds",
              all(r["after_best"]["mean"] > r["before"]["mean"] for r in res.values()))


def phase2():
    """Verify Phase 2 Double DQN training results and threshold criteria.

    Input:  None.
    Output: None (calls algo_phase for dqn with threshold 475 and min_pass 4).
    Flow:   invoke algo_phase('dqn', 475, 4).
    """
    algo_phase("dqn", 475, 4)


def phase3():
    """Verify Phase 3 REINFORCE policy gradient results and threshold criteria.

    Input:  None.
    Output: None (calls algo_phase for reinforce with threshold 200 and min_pass 3).
    Flow:   invoke algo_phase('reinforce', 200, 3).
    """
    algo_phase("reinforce", 200, 3)


def phase4():
    """Verify Phase 4 comparative experiments, ablations, learning curves, and robustness.

    Input:  None.
    Output: None (executes check assertions for Phase 4).
    Flow:   verify summary.csv containing base and ablation runs, learning_curves.png, and noise robustness files.
    """
    p = "outputs/results/summary.csv" if exists("outputs/results/summary.csv") else "results/summary.csv"
    ok = exists(p)
    check(p, ok)
    if ok:
        rows = read_csv(p)
        keys = {(r["algo"], r["tag"]) for r in rows}
        check("contains dqn:base entry", ("dqn", "base") in keys)
        check("contains reinforce:base entry", ("reinforce", "base") in keys)
        ablations = {k for k in keys if k[0] == "dqn" and k[1] not in ("base", "smoke")}
        check("has >= 2 DQN ablation variants", len(ablations) >= 2, f"{sorted(ablations)}")
    learn_p = "outputs/figures/learning_curves.png" if exists("outputs/figures/learning_curves.png") else "figures/learning_curves.png"
    check(learn_p, exists(learn_p))
    noise_p = "outputs/figures/noise_robustness.png" if exists("outputs/figures/noise_robustness.png") else "figures/noise_robustness.png"
    check(noise_p, exists(noise_p))
    noise_csv = "outputs/results/noise_robustness.csv" if exists("outputs/results/noise_robustness.csv") else "results/noise_robustness.csv"
    check(noise_csv, exists(noise_csv))


def phase5():
    """Verify Phase 5 error analysis, demo notebooks, video GIFs, and trajectory plots.

    Input:  None.
    Output: None (executes check assertions for Phase 5).
    Flow:   verify error_analysis.csv, animation GIFs, trajectories.png, policy_heatmap.png, and valid 04_demo.ipynb.
    """
    err_csv = "outputs/results/error_analysis.csv" if exists("outputs/results/error_analysis.csv") else "results/error_analysis.csv"
    targets = [
        "outputs/videos/random.gif" if exists("outputs/videos/random.gif") else "videos/random.gif",
        "outputs/videos/dqn_best.gif" if exists("outputs/videos/dqn_best.gif") else "videos/dqn_best.gif",
        "outputs/videos/reinforce_best.gif" if exists("outputs/videos/reinforce_best.gif") else "videos/reinforce_best.gif",
        "outputs/videos/dqn_new_data.gif" if exists("outputs/videos/dqn_new_data.gif") else "videos/dqn_new_data.gif",
        err_csv,
        "outputs/figures/trajectories.png" if exists("outputs/figures/trajectories.png") else "figures/trajectories.png",
        "outputs/figures/policy_heatmap.png" if exists("outputs/figures/policy_heatmap.png") else "figures/policy_heatmap.png",
        "notebooks/04_demo.ipynb"
    ]
    for p in targets:
        check(p, exists(p))
    if exists("notebooks/04_demo.ipynb"):
        try:
            json.load(open("notebooks/04_demo.ipynb", encoding="utf-8"))
            check("04_demo.ipynb is valid JSON", True)
        except Exception as e:
            check("04_demo.ipynb is valid JSON", False, str(e))


def phase6():
    """Verify Phase 6 reporting artifacts, markdown tables, presentation slides, and README.

    Input:  None.
    Output: None (executes check assertions for Phase 6).
    Flow:   verify tables.md, report.md, slide files, README.md, and ensure report mentions key figures and terms.
    """
    tbl_p = "public/docs/tables.md" if exists("public/docs/tables.md") else "report/tables.md"
    check(tbl_p, exists(tbl_p))
    rep_p = "public/docs/report.md" if exists("public/docs/report.md") else "report/report.md"
    check(rep_p, exists(rep_p))
    slides_ok = (exists("public/docs/slides.md") or exists("public/docs/slides_outline.md") or
                 exists("public/docs/slides.pptx") or
                 exists("report/slides_outline.md") or exists("report/slides.pptx"))
    slides_label = "public/docs/slides.md or public/docs/slides.pptx"
    check(slides_label, slides_ok)
    check("README.md", exists("README.md"))
    if exists(rep_p):
        txt = open(rep_p, encoding="utf-8").read()
        check("no placeholders remain (TODO/XXX/{{/TBD)", not re.search(r"TODO|XXX|\{\{|TBD", txt))
        for fig in ["learning_curves.png", "noise_robustness.png", "trajectories.png"]:
            check(f"report references {fig}", fig in txt)
        for sec in ["state", "action", "reward", "episode"]:
            check(f"report describes '{sec}'", sec in txt.lower())
        check("report contains limitations section", "hạn chế" in txt.lower())


PHASES = {"0": phase0, "1": phase1, "2": phase2, "3": phase3, "4": phase4, "5": phase5, "6": phase6}


def main():
    """CLI entrypoint: execute phase verification checks and display summary.

    Input:  sys.argv[1] -- phase number ('0' through '6') or 'all'.
    Output: exit code 0 if all checks pass, exit code 1 if any check fails.
    Flow:   determine target phases, execute respective check functions, summarize outcomes.
    """
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    todo = list(PHASES) if arg == "all" else [arg]
    if any(t not in PHASES for t in todo):
        print(__doc__); sys.exit(2)
    for t in todo:
        print(f"== Phase {t} Acceptance Gate ==")
        PHASES[t]()
    ok = all(results)
    print("\nRESULT:", "PASS" if ok else "FAIL", f"({sum(results)}/{len(results)} items)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
