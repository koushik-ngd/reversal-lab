"""Command-line interface.

Only commands backed by working functionality are exposed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from reversallab import __version__
from reversallab.exceptions import ConfigurationError, ReversalLabError
from reversallab.experiments.config import load_config
from reversallab.experiments.estimator_validation import run_estimator_validation
from reversallab.experiments.metadata import load_run, to_jsonable, write_run

logger = logging.getLogger("reversallab")


def _fmt(value: Any, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def format_report(run: dict[str, Any]) -> str:
    """Human-readable summary of an estimator-validation ``run.json``."""
    lines = [
        f"Run:      {run['run_id']}",
        f"Config:   sha256 {run['config_sha256'][:12]}  seed={run['config']['seed']}  "
        f"replications={run['config']['n_replications']}",
        f"Results:  sha256 {run['results_sha256'][:12]}",
        f"Git:      {run['provenance']['git']}",
        "",
    ]
    header = (
        f"{'scenario':<24}{'n':>7}  {'est':<6}{'true V':>9}{'mean':>9}{'bias':>10}{'±mcse':>9}"
        f"{'rmse':>9}{'cover':>7}{'ESS/N':>7}{'fail':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for sc in run["results"]["scenarios"]:
        true_v = sc["evaluation_policy_value"]["value"]
        for cell in sc["cells"]:
            for name, s in cell["estimators"].items():
                lines.append(
                    f"{sc['name']:<24}{cell['n_samples']:>7}  {name:<6}{true_v:>9.4f}"
                    f"{_fmt(s.get('mean_estimate'), '9.4f')}{_fmt(s.get('bias'), '+10.4f')}"
                    f"{_fmt(s.get('bias_monte_carlo_se'), '9.4f')}{_fmt(s.get('rmse'), '9.4f')}"
                    f"{_fmt(s.get('ci_coverage'), '7.3f')}"
                    f"{cell['diagnostics']['mean_ess_fraction']:>7.3f}{s['n_failed']:>6}"
                )
    if run["warnings"]:
        lines += ["", "Warnings:"] + [f"  - {w}" for w in run["warnings"]]
    lines += ["", f"Reproduce: {run['reproduction_command']}"]
    return "\n".join(lines)


def _cmd_validate_estimators(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.n_replications is not None:
        try:
            config = type(config).model_validate(
                {**config.model_dump(), "n_replications": args.n_replications}
            )
        except ValidationError as exc:
            raise ConfigurationError(f"Invalid --n-replications override:\n{exc}") from exc
    config_dict = to_jsonable(config.model_dump(mode="json"))
    logger.info(
        "experiment=%s seed=%d replications=%d", config.name, config.seed, config.n_replications
    )
    output = run_estimator_validation(config)
    command = f"reversallab validate-estimators --config {Path(args.config).as_posix()}"
    if args.n_replications is not None:
        command += f" --n-replications {args.n_replications}"
    run_dir = write_run(
        Path(args.output_dir),
        name=config.name,
        config=config_dict,
        results=output.results,
        warnings=output.warnings,
        runtime=output.runtime,
        reproduction_command=command,
    )
    print(format_report(load_run(run_dir)))
    print(f"\nWrote {run_dir / 'run.json'}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    try:
        run = load_run(Path(args.run_dir))
    except FileNotFoundError as exc:
        raise ReversalLabError(str(exc)) from exc
    print(format_report(run))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reversallab",
        description="Statistically grounded off-policy evaluation and decision-robustness "
        "analysis.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (logs go to stderr).",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p_val = sub.add_parser(
        "validate-estimators",
        help="Validate IPS/SNIPS against synthetic ground truth (experiment 001).",
        description="Simulate logged bandit data for each configured scenario, apply the "
        "configured estimators, and report bias, RMSE, CI coverage and diagnostics against the "
        "known policy value.",
    )
    p_val.add_argument("--config", required=True, help="Path to an experiment YAML file.")
    p_val.add_argument(
        "--output-dir", default="runs", help="Directory in which a new run directory is created."
    )
    p_val.add_argument(
        "--n-replications",
        type=int,
        default=None,
        help="Override the configured number of replications (recorded in the run config).",
    )
    p_val.set_defaults(func=_cmd_validate_estimators)

    p_rep = sub.add_parser(
        "report",
        help="Print the summary of an existing run.",
        description="Read run.json from a run directory and print a summary table.",
    )
    p_rep.add_argument("--run-dir", required=True, help="Run directory containing run.json.")
    p_rep.set_defaults(func=_cmd_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s level=%(levelname)s logger=%(name)s %(message)s",
        stream=sys.stderr,
    )
    try:
        result: int = args.func(args)
    except ReversalLabError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return result


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
