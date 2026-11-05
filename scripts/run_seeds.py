"""Execution script to run training across 5 seeds sequentially for a specified algorithm and tag.

Usage:
    python scripts/run_seeds.py dqn base
    python scripts/run_seeds.py dqn nodouble --no_double
    python scripts/run_seeds.py reinforce base
"""
import subprocess
import sys

DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def main():
    """Run sequential seeds (0 to 4) for a target algorithm and tag configuration.

    Input:  argv[1] = algo ('dqn' or 'reinforce'), argv[2] = tag, argv[3:] = additional arguments.
    Output: None (executes child processes and streams logs to stdout).
    Flow:   iterate through seeds 0 to 4, construct CLI command, execute via subprocess.run,
            exit with error code on failure.
    """
    if len(sys.argv) < 3:
        print("Usage: python scripts/run_seeds.py <algo> <tag> [extra args...]")
        sys.exit(1)
    algo = sys.argv[1]
    tag = sys.argv[2]
    extra = sys.argv[3:]
    python = sys.executable
    for seed in DEFAULT_SEEDS:
        cmd = [python, "-m", f"libraries.{algo}", "--seed", str(seed), "--tag", tag] + extra
        print(f"\n=== {algo} seed={seed} tag={tag} ===")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"FAILED: seed={seed} exited with code {result.returncode}")
            sys.exit(result.returncode)
    print(f"\nCompleted {algo}:{tag} ({len(DEFAULT_SEEDS)} seeds)")


if __name__ == "__main__":
    main()

