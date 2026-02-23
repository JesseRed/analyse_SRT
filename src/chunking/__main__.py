"""CLI for chunking analysis: --method, --benchmark, single file or batch."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import list_methods, run_batch, run_benchmark, run_single_file


def _build_parser() -> argparse.ArgumentParser:
    methods = list_methods()
    parser = argparse.ArgumentParser(
        description="SRTT chunking analysis. Multiple methods available."
    )
    parser.add_argument(
        "--method",
        choices=methods,
        default=methods[0] if methods else None,
        help="Chunking method to use.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run all methods on the same files; write per-method dirs and combined benchmark_summary.csv.",
    )
    parser.add_argument(
        "--input-dir",
        default="SRT",
        help="Directory containing participant CSV files.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Single file to analyze (overrides --input-dir for one file).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern for input files (batch mode).",
    )
    parser.add_argument(
        "--sequence-type",
        default="blue",
        choices=["blue", "green", "yellow"],
        help="Sequence type to analyze.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of files (batch/benchmark).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )
    # Method-specific (community_network)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        help="Intralayer resolution (community_network).",
    )
    parser.add_argument(
        "--coupling",
        type=float,
        default=0.03,
        help="Interlayer coupling (community_network).",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Community-detection repeats per file (community_network).",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=20,
        help="Null-model permutations per file (community_network).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    method_params = {
        "gamma": args.gamma,
        "coupling": args.coupling,
        "n_iter": args.n_iter,
        "n_permutations": args.n_permutations,
        "random_state": args.seed,
    }

    if args.input_file is not None:
        out_dir = Path(args.output_dir) / args.method
        run_single_file(
            args.method,
            args.input_file,
            out_dir,
            sequence_type=args.sequence_type,
            **method_params,
        )
        print(f"Single-file analysis complete: {out_dir}")
        return 0

    if args.benchmark:
        result = run_benchmark(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            pattern=args.pattern,
            sequence_type=args.sequence_type,
            limit=args.limit,
            combine_summary=True,
            **method_params,
        )
        print("Benchmark complete:")
        print(f"  output_dir: {result['output_dir']}")
        print(f"  methods:   {result['methods']}")
        if result.get("benchmark_summary_path"):
            print(f"  combined:   {result['benchmark_summary_path']}")
        return 0

    result = run_batch(
        method_name=args.method,
        input_dir=args.input_dir,
        output_dir=Path(args.output_dir) / args.method,
        pattern=args.pattern,
        sequence_type=args.sequence_type,
        limit=args.limit,
        progress_log=True,
        **method_params,
    )
    print("Batch chunking analysis complete:")
    print(f"  method:  {args.method}")
    print(f"  total:   {result['n_files_total']}")
    print(f"  success: {result['n_files_success']}")
    print(f"  failed:  {result['n_files_failed']}")
    print(f"  summary: {result['summary_path']}")
    print(f"  trials:  {result['trials_path']}")
    print(f"  errors:  {result['errors_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
