"""CLI for chunking analysis: --method, --benchmark, single file or batch."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import (
    list_methods,
    merge_sequence_summaries,
    run_batch,
    run_benchmark,
    run_single_file,
)


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
        choices=["blue", "green", "yellow", "all"],
        help="Sequence type to analyze (or 'all' for blue/green/yellow).",
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
    # Method-specific (change_point_pelt)
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Sliding-window size in blocks (change_point_pelt).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Sliding-window step in blocks (change_point_pelt).",
    )
    parser.add_argument(
        "--min-blocks",
        type=int,
        default=6,
        help="Minimum blocks per window and condition (change_point_pelt).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=500,
        help="Bootstrap resamples per window (change_point_pelt).",
    )
    parser.add_argument(
        "--fpr-target",
        type=float,
        default=0.05,
        help="Target false-positive rate for penalty calibration (change_point_pelt).",
    )
    parser.add_argument(
        "--mean-cp-target",
        type=float,
        default=0.1,
        help="Target mean change-points per null window (change_point_pelt).",
    )
    parser.add_argument(
        "--n-null-splits",
        type=int,
        default=32,
        help="Null random-minus-random splits per window (change_point_pelt).",
    )
    parser.add_argument(
        "--penalty-grid-size",
        type=int,
        default=80,
        help="Number of candidate penalties in calibration grid (change_point_pelt).",
    )
    parser.add_argument(
        "--min-windows-reliable",
        type=int,
        default=3,
        help="Minimum valid windows for reliable=True (change_point_pelt).",
    )
    parser.add_argument(
        "--allow-fallback-penalty",
        action="store_true",
        help="Allow fallback penalty when yellow/null calibration data is insufficient (change_point_pelt).",
    )
    parser.add_argument(
        "--fallback-penalty",
        type=float,
        default=None,
        help="Optional fixed fallback penalty value (change_point_pelt).",
    )
    parser.add_argument(
        "--penalty-sensitive",
        action="store_true",
        help="Use less conservative penalty (fpr_target=0.10, mean_cp_target=0.2) for more change points (change_point_pelt).",
    )
    parser.add_argument(
        "--short-session",
        action="store_true",
        help="Use smaller windows for short sessions (window_size=20, step=5, min_blocks=4) to get more valid windows (change_point_pelt).",
    )
    parser.add_argument(
        "--cost-model",
        choices=["l2", "rbf"],
        default="l2",
        help="Cost function for PELT: l2 (piecewise constant mean) or rbf (kernel-based, more sensitive) (change_point_pelt).",
    )
    parser.add_argument(
        "--rbf-gamma",
        type=float,
        default=None,
        help="RBF kernel bandwidth (change_point_pelt). If unset, ruptures uses median heuristic.",
    )
    parser.add_argument(
        "--merge-summaries",
        action="store_true",
        help="After --sequence-type all, write combined_summary.csv (blue/green/yellow merge).",
    )
    # Method-specific (hsmm_chunking)
    parser.add_argument(
        "--n-states",
        type=int,
        default=None,
        help="Number of HSMM states (hsmm_chunking). Default: 3 for blue/green, 2 for yellow.",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=7,
        help="Max state duration in steps (hsmm_chunking).",
    )
    parser.add_argument(
        "--baseline-correction",
        action="store_true",
        default=True,
        help="Apply yellow median baseline correction for blue/green (hsmm_chunking).",
    )
    parser.add_argument(
        "--no-baseline-correction",
        action="store_false",
        dest="baseline_correction",
        help="Disable baseline correction (hsmm_chunking).",
    )
    parser.add_argument(
        "--n-gibbs-iter",
        type=int,
        default=150,
        help="Number of Gibbs sampling iterations for pyhsmm HDP-HSMM (hsmm_chunking).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.method == "community_network":
        method_params = {
            "gamma": args.gamma,
            "coupling": args.coupling,
            "n_iter": args.n_iter,
            "n_permutations": args.n_permutations,
            "random_state": args.seed,
        }
    elif args.method == "change_point_pelt":
        window_size = args.window_size
        step = args.step
        min_blocks = args.min_blocks
        fpr_target = args.fpr_target
        mean_cp_target = args.mean_cp_target
        if args.short_session:
            window_size = 20
            step = 5
            min_blocks = 4
        if args.penalty_sensitive:
            fpr_target = 0.10
            mean_cp_target = 0.2
        method_params = {
            "window_size": window_size,
            "step": step,
            "min_blocks": min_blocks,
            "n_bootstrap": args.n_bootstrap,
            "fpr_target": fpr_target,
            "mean_cp_target": mean_cp_target,
            "n_null_splits": args.n_null_splits,
            "penalty_grid_size": args.penalty_grid_size,
            "min_windows_reliable": args.min_windows_reliable,
            "allow_fallback_penalty": args.allow_fallback_penalty,
            "fallback_penalty": args.fallback_penalty,
            "cost_model": args.cost_model,
            "rbf_gamma": args.rbf_gamma,
            "random_state": args.seed,
        }
    elif args.method == "hsmm_chunking":
        method_params = {
            "n_states": args.n_states,
            "max_duration": args.max_duration,
            "window_size": args.window_size,
            "step": args.step,
            "min_blocks": args.min_blocks,
            "min_windows_reliable": args.min_windows_reliable,
            "baseline_correction": args.baseline_correction,
            "random_state": args.seed,
            "n_gibbs_iter": args.n_gibbs_iter,
        }
    else:
        method_params = {"random_state": args.seed}

    sequences = (
        ["blue", "green", "yellow"]
        if args.sequence_type == "all"
        else [args.sequence_type]
    )

    if args.input_file is not None:
        for seq in sequences:
            out_dir = Path(args.output_dir) / args.method / seq
            run_single_file(
                args.method,
                args.input_file,
                out_dir,
                sequence_type=seq,
                **method_params,
            )
            print(f"Single-file analysis complete ({seq}): {out_dir}")
        if args.merge_summaries and args.sequence_type == "all":
            combined = merge_sequence_summaries(Path(args.output_dir) / args.method)
            print(f"Combined summary written: {combined}")
        return 0

    if args.benchmark:
        for seq in sequences:
            result = run_benchmark(
                input_dir=args.input_dir,
                output_dir=Path(args.output_dir) / seq,
                pattern=args.pattern,
                sequence_type=seq,
                limit=args.limit,
                combine_summary=True,
                **method_params,
            )
            print(f"Benchmark complete ({seq}):")
            print(f"  output_dir: {result['output_dir']}")
            print(f"  methods:   {result['methods']}")
            if result.get("benchmark_summary_path"):
                print(f"  combined:   {result['benchmark_summary_path']}")
        if args.merge_summaries and args.sequence_type == "all":
            print("Skipping --merge-summaries for benchmark mode.")
            print("Use per-method output directories and merge them separately.")
        return 0

    for seq in sequences:
        result = run_batch(
            method_name=args.method,
            input_dir=args.input_dir,
            output_dir=Path(args.output_dir) / args.method / seq,
            pattern=args.pattern,
            sequence_type=seq,
            limit=args.limit,
            progress_log=True,
            **method_params,
        )
        print(f"Batch chunking analysis complete ({seq}):")
        print(f"  method:  {args.method}")
        print(f"  total:   {result['n_files_total']}")
        print(f"  success: {result['n_files_success']}")
        print(f"  failed:  {result['n_files_failed']}")
        print(f"  summary: {result['summary_path']}")
        print(f"  trials:  {result['trials_path']}")
        print(f"  errors:  {result['errors_path']}")
    if args.merge_summaries and args.sequence_type == "all":
        combined = merge_sequence_summaries(Path(args.output_dir) / args.method)
        print(f"Combined summary written: {combined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
