"""CLI entry point: python -m deepclaude"""

import argparse
import sys

from deepclaude.orchestrator import Config, Orchestrator


def main():
    parser = argparse.ArgumentParser(description="DeepClaude: Autonomous quant factor research")
    parser.add_argument("--rounds", "-r", type=int, default=10, help="Max evolution rounds")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Top K factors for selection")
    parser.add_argument("--max-iter", type=int, default=999, help="Max iterations per Claude instance")
    parser.add_argument("--data-dir", type=str, default="/Users/shine/trader-data", help="Path to trader-data")
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    parser.add_argument("--resume", action="store_true", help="Resume previous interrupted run")
    args = parser.parse_args()

    config = Config(
        max_rounds=args.rounds,
        top_k=args.top_k,
        max_iterations=args.max_iter,
        data_dir=args.data_dir,
        project_root=args.project_root,
    )
    orch = Orchestrator(config)
    orch.run(resume=args.resume)


if __name__ == "__main__":
    main()
