"""CLI: python -m src.simulation <command> ...

Commands
  calibrate     estimate empirical parameters from the real SRT files
  fit-noise     grid-search the residual-noise parameters against the real data
  generate      generate a simulation tier from a YAML config
  realism       compare simulated with real data (markdown report + plot)
  run-methods   run chunking methods on simulated data
  evaluate      score method outputs against the ground truth
"""

from __future__ import annotations

import os

# The commands parallelise over processes; multi-threaded BLAS inside every worker
# oversubscribes the CPU massively (observed load > 100 on 16 cores). Must be set
# before numpy is imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402


def _csv(value):
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.simulation", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("calibrate", help="estimate empirical parameters from real data")
    p.add_argument("--srt-dir", default="SRT")
    p.add_argument("--pattern", default="VR_ML*.csv")
    p.add_argument("--n-files", type=int, default=400)
    p.add_argument("--output", default="configs/simulation/empirical_calibration.json")

    p = sub.add_parser("fit-noise", help="fit residual-noise parameters (grid search)")
    p.add_argument("--calibration", default="configs/simulation/empirical_calibration.json")
    p.add_argument("--n-participants", type=int, default=24)
    p.add_argument("--jobs", type=int, default=8)

    p = sub.add_parser("generate", help="generate simulated data")
    p.add_argument("--config", required=True)
    p.add_argument("--families", type=_csv, default=None, help="comma-separated subset, e.g. A_null,B_static")
    p.add_argument("--jobs", type=int, default=8)

    p = sub.add_parser("realism", help="compare simulated vs real data")
    p.add_argument("sim_dirs", nargs="+", help="family directories, optionally restricted to one cell: simulations/core/A_null@A5-seqspeedup")
    p.add_argument("--calibration", default="configs/simulation/empirical_calibration.json")
    p.add_argument("--output", default="simulations/core/realism_report.md")
    p.add_argument("--max-files", type=int, default=300)

    p = sub.add_parser("run-methods", help="run chunking methods on simulated data")
    p.add_argument("--sim-dir", required=True, help="tier directory, e.g. simulations/core")
    p.add_argument("--results", required=True, help="output root, e.g. outputs/sim_core")
    p.add_argument("--families", type=_csv, default=None)
    p.add_argument("--methods", type=_csv, default=None)
    p.add_argument("--sequences", type=_csv, default=None)
    p.add_argument("--config", default="methods_config.json")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="max files per family (testing)")

    p = sub.add_parser("evaluate", help="score method outputs against the ground truth")
    p.add_argument("--sim-dir", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--families", type=_csv, default=None)
    p.add_argument("--methods", type=_csv, default=None)

    args = parser.parse_args(argv)

    if args.cmd == "calibrate":
        from .calibrate import run_calibration

        c = run_calibration(args.srt_dir, args.pattern, args.n_files, output_path=args.output)
        print(f"calibration written: {args.output} ({c['n_files']} files, {c['n_iki']} IKIs)")
    elif args.cmd == "fit-noise":
        from .noisefit import fit_noise

        fit = fit_noise(args.calibration, n_participants=args.n_participants, jobs=args.jobs)
        print(f"best noise parameters: {fit['params']} (loss {fit['loss']:.3f})")
    elif args.cmd == "generate":
        from .generate import generate_from_file

        generate_from_file(args.config, jobs=args.jobs, families=args.families)
    elif args.cmd == "realism":
        from .realism import realism_report

        out = realism_report(args.sim_dirs, args.calibration, args.output, max_files=args.max_files)
        print(f"realism report: {out}")
    elif args.cmd == "run-methods":
        from .run_methods import run_methods

        run_methods(args.sim_dir, args.results, families=args.families, methods=args.methods,
                    sequences=args.sequences, config_path=args.config, jobs=args.jobs, limit=args.limit)
    elif args.cmd == "evaluate":
        from .evaluate import run_evaluation

        res = run_evaluation(args.sim_dir, args.results, args.output, families=args.families, methods=args.methods)
        print(f"evaluation written to {res['output_dir']} ({res['n_blocks']} blocks, methods={res['methods']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
