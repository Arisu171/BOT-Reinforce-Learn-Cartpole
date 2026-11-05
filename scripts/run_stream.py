"""Launch CartPole-v1 web streaming server for infinite execution.

Usage:
    python scripts/run_stream.py
    python scripts/run_stream.py --port 8000 --algo dqn --mode infinite
    python scripts/run_stream.py --algo reinforce --fps 60
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from libraries.stream import (
    DEFAULT_ALGO,
    DEFAULT_FPS,
    DEFAULT_HOST,
    DEFAULT_MODE,
    DEFAULT_PORT,
    DEFAULT_SIGMA,
    start_server,
)


def main():
    """Parse CLI options and run the CartPole streaming server.

    Input:  sys.argv command line arguments.
    Output: None (runs server indefinitely).
    Flow:   configure ArgumentParser, parse options, invoke start_server.
    """
    ap = argparse.ArgumentParser(description="CartPole-v1 Infinite Web Streaming Server")
    ap.add_argument("--host", default=DEFAULT_HOST, help="Bind IP address (default: 0.0.0.0)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on (default: 8000)")
    ap.add_argument("--algo", choices=["dqn", "reinforce", "heuristic", "random", "manual"],
                    default=DEFAULT_ALGO, help="Initial policy algorithm (default: dqn)")
    ap.add_argument("--model", help="Path to checkpoint model .pt file")
    ap.add_argument("--mode", choices=["infinite", "standard"], default=DEFAULT_MODE,
                    help="Infinite horizon (no 500 step cutoff) or standard 500-step episodes")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Target simulation FPS (default: 30)")
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA, help="Observation noise sigma (default: 0.0)")
    args = ap.parse_args()

    start_server(
        host=args.host,
        port=args.port,
        algo=args.algo,
        model=args.model,
        mode=args.mode,
        fps=args.fps,
        sigma=args.sigma,
    )


if __name__ == "__main__":
    main()

