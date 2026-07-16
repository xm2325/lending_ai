from __future__ import annotations

import argparse
import json

from lending_ai_lab.deep_real_experiment import run_deep_uci_benchmark
from lending_ai_lab.experiment import run_demo
from lending_ai_lab.real_experiment import run_uci_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lending-ai")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="Run the full synthetic underwriting experiment")
    demo.add_argument("--n-samples", type=int, default=12_000)
    demo.add_argument("--epochs", type=int, default=8)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--output-dir", default="artifacts")
    uci = subparsers.add_parser("uci", help="Run the public UCI real-data benchmark")
    uci.add_argument("--data-path", required=True)
    uci.add_argument("--seed", type=int, default=42)
    uci.add_argument("--output-dir", default="artifacts_real")
    deep_uci = subparsers.add_parser(
        "deep-uci", help="Compare deep sequence models with the shared-fold tabular champion"
    )
    deep_uci.add_argument("--data-path", required=True)
    deep_uci.add_argument("--seed", type=int, default=42)
    deep_uci.add_argument("--folds", type=int, default=3)
    deep_uci.add_argument("--epochs", type=int, default=8)
    deep_uci.add_argument("--output-dir", default="artifacts_deep")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "demo":
        summary = run_demo(
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            epochs=args.epochs,
            seed=args.seed,
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "uci":
        summary = run_uci_benchmark(args.data_path, args.output_dir, args.seed)
        print(json.dumps(summary, indent=2))
    elif args.command == "deep-uci":
        summary = run_deep_uci_benchmark(
            args.data_path,
            args.output_dir,
            folds=args.folds,
            epochs=args.epochs,
            seed=args.seed,
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
