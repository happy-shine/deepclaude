"""CLI entry point: python -m deepclaude"""

import argparse
import sys

from deepclaude.orchestrator import Config, Orchestrator


def main():
    parser = argparse.ArgumentParser(description="DeepClaude: Autonomous quant factor research")
    parser.add_argument("--parallel", "-n", type=int, default=3, help="Concurrent Claude instances per round")
    parser.add_argument("--rounds", "-r", type=int, default=10, help="Max evolution rounds")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Top K factors for selection")
    parser.add_argument("--max-iter", type=int, default=20, help="Max iterations per Claude instance")
    parser.add_argument("--data-dir", type=str, default="/Users/shine/trader-data", help="Path to trader-data")
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    args = parser.parse_args()

    config = Config(
        n_parallel=args.parallel,
        max_rounds=args.rounds,
        top_k=args.top_k,
        max_iterations=args.max_iter,
        data_dir=args.data_dir,
        project_root=args.project_root,
    )
    orch = Orchestrator(config)
    orch.run()


if __name__ == "__main__":
    main()
