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
from reversallab.experiments.advanced_estimators import run_advanced_estimator_validation
from reversallab.experiments.config import (
    load_advanced_config,
    load_config,
    load_independent_reliability_config,
    load_parity_config,
    load_reliability_config,
    load_reversal_config,
    load_uncertainty_config,
)
from reversallab.experiments.estimator_validation import run_estimator_validation
from reversallab.experiments.independent_reliability import run_independent_reliability
from reversallab.experiments.metadata import load_run, to_jsonable, write_run
from reversallab.experiments.open_bandit_parity import run_open_bandit_parity
from reversallab.experiments.reversal_prototype import run_reversal_prototype
from reversallab.experiments.reversal_reliability import run_reversal_reliability
from reversallab.experiments.uncertainty_calibration import run_uncertainty_calibration

logger = logging.getLogger("reversallab")


def _fmt(value: Any, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def format_report(run: dict[str, Any]) -> str:
    """Human-readable summary of a ``run.json`` produced by this package."""
    results = run.get("results", {})
    if isinstance(results, dict) and "comparisons" in results:
        return _format_parity_report(run)
    if isinstance(results, dict) and "scenarios" in results:
        scs = results["scenarios"]
        if scs and "method_d_f_duality_validation" in scs[0]:
            return _format_independent_reliability_report(run)
        if scs and "reversal_search_validation" in scs[0]:
            return _format_reliability_report(run)
        if scs and "reversal_search" in scs[0]:
            return _format_reversal_report(run)
        if scs and scs[0].get("cells") and "methods" in scs[0]["cells"][0]:
            return _format_uncertainty_report(run)
    lines = [
        f"Run:      {run['run_id']}",
        f"Config:   sha256 {run['config_sha256'][:12]}  seed={run['config']['seed']}  "
        f"replications={run['config']['n_replications']}",
        f"Results:  sha256 {run['results_sha256'][:12]}",
        f"Git:      {run['provenance']['git']}",
        "",
    ]
    header = (
        f"{'scenario':<32}{'n':>7}  {'est':<6}{'true V':>9}{'mean':>9}{'bias':>10}{'±mcse':>9}"
        f"{'rmse':>9}{'cover':>7}{'ESS/N':>7}{'fail':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for sc in run["results"]["scenarios"]:
        true_v = sc["evaluation_policy_value"]["value"]
        for cell in sc["cells"]:
            for name, s in cell["estimators"].items():
                lines.append(
                    f"{sc['name']:<32}{cell['n_samples']:>7}  {name:<6}{true_v:>9.4f}"
                    f"{_fmt(s.get('mean_estimate'), '9.4f')}{_fmt(s.get('bias'), '+10.4f')}"
                    f"{_fmt(s.get('bias_monte_carlo_se'), '9.4f')}{_fmt(s.get('rmse'), '9.4f')}"
                    f"{_fmt(s.get('ci_coverage'), '7.3f')}"
                    f"{cell['diagnostics']['mean_ess_fraction']:>7.3f}{s['n_failed']:>6}"
                )
    if run["warnings"]:
        lines += ["", "Warnings:"] + [f"  - {w}" for w in run["warnings"]]
    lines += ["", f"Reproduce: {run['reproduction_command']}"]
    return "\n".join(lines)


def _format_uncertainty_report(run: dict[str, Any]) -> str:
    lines = [
        f"Run:      {run['run_id']}",
        (
            f"Config:   sha256 {run['config_sha256'][:12]}  seed={run['config']['seed']}  "
            f"replications={run['config']['n_replications']}  "
            f"bootstraps={run['config'].get('n_bootstraps')}"
        ),
        f"Results:  sha256 {run['results_sha256'][:12]}",
        f"Git:      {run['provenance']['git']}",
        "",
    ]
    header = (
        f"{'scenario':<30}{'n':>6}  {'method':<28}{'true V':>8}{'mean':>8}{'bias':>9}"
        f"{'cover':>7}{'±mcse':>7}{'width':>8}{'fail':>5}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for sc in run["results"]["scenarios"]:
        true_v = sc["evaluation_policy_value"]["value"]
        for cell in sc["cells"]:
            for name, s in cell["methods"].items():
                lines.append(
                    f"{sc['name']:<30}{cell['n_samples']:>6}  {name:<28}{true_v:>8.4f}"
                    f"{_fmt(s.get('point_estimate_mean'), '8.4f')}{_fmt(s.get('bias'), '+9.4f')}"
                    f"{_fmt(s.get('empirical_coverage'), '7.3f')}"
                    f"{_fmt(s.get('coverage_monte_carlo_se'), '7.3f')}"
                    f"{_fmt(s.get('mean_interval_width'), '8.4f')}{s['n_failed']:>5}"
                )
    if run["warnings"]:
        lines += ["", "Warnings:"] + [f"  - {w}" for w in run["warnings"]]
    lines += ["", f"Reproduce: {run['reproduction_command']}"]
    return "\n".join(lines)


def _format_reversal_report(run: dict[str, Any]) -> str:
    lines = [
        f"Run:      {run['run_id']}",
        (
            f"Config:   sha256 {run['config_sha256'][:12]}  seed={run['config']['seed']}  "
            f"confidence_level={run['config'].get('confidence_level', 0.95)}"
        ),
        f"Results:  sha256 {run['results_sha256'][:12]}",
        f"Git:      {run['provenance']['git']}",
        "",
    ]
    header = (
        f"{'scenario':<22}{'N':>5}  {'margin':>6}  {'true Delta':>10}  {'est Delta':>10}  "
        f"{'CI(95%)':^17}  {'naive':<9}{'baseline':<13}{'d_rev*':>8}  {'status':<20}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for sc in run["results"]["scenarios"]:
        pe = sc["paired_estimation"]
        ci_str = f"[{pe['ci_lower']:+.3f}, {pe['ci_upper']:+.3f}]"
        dec = sc["decisions"]
        rev = sc["reversal_search"]
        d_rev_val = rev["reversal_distance"]
        d_rev_str = f"{d_rev_val:.4f}" if d_rev_val is not None else "None"
        row = (
            f"{sc['scenario']:<22}{sc['n_samples']:>5}  {sc['margin']:>6.3f}  "
            f"{sc['ground_truth']['true_delta']:>+10.4f}  {pe['point_estimate']:>+10.4f}  "
            f"{ci_str:^17}  {dec['naive_point_estimate']:<9}"
            f"{dec['baseline_confidence_bound']:<13}"
            f"{d_rev_str:>8}  {rev['status']:<20}"
        )
        lines.append(row)
    if run["warnings"]:
        lines += ["", "Warnings:"] + [f"  - {w}" for w in run["warnings"]]
    lines += ["", f"Reproduce: {run['reproduction_command']}"]
    return "\n".join(lines)


def _format_independent_reliability_report(run: dict[str, Any]) -> str:
    lines = [
        f"Run:      {run['run_id']}",
        (
            f"Config:   sha256 {run['config_sha256'][:12]}  seed={run['config']['seed']}  "
            f"replications={run['config'].get('n_replications', 500)}  "
            f"deploy_eps={run['config'].get('deploy_threshold_epsilon', 0.10)}"
        ),
        f"Results:  sha256 {run['results_sha256'][:12]}",
        f"Git:      {run['provenance']['git']}",
        "",
    ]
    header = (
        f"{'scenario':<28}{'N':>5}  {'margin':>6}  {'true Delta':>10} {'worst Delta':>11}  "
        f"{'Rec A':>6} {'Rec B':>6} {'Rec C':>6} {'Rec D':>6} {'Rec E':>6} {'Rec F':>6}  "
        f"{'SimCov':>7} {'PWCov':>6} {'D-F Equiv':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for sc in run["results"]["scenarios"]:
        gt = sc["ground_truth"]
        m = sc["methods"]
        cov = sc["simultaneous_inference_validation"]
        duality = sc["method_d_f_duality_validation"]
        rec_a = f"{m['method_a_point_estimate']['recommend_rate'] * 100:.1f}%"
        rec_b = f"{m['method_b_baseline_ci']['recommend_rate'] * 100:.1f}%"
        rec_c = f"{m['method_c_worst_stratum']['recommend_rate'] * 100:.1f}%"
        rec_d = f"{m['method_d_pointwise_reversal']['recommend_rate'] * 100:.1f}%"
        rec_e = f"{m['method_e_simultaneous_band']['recommend_rate'] * 100:.1f}%"
        rec_f = f"{m['method_f_robust_mixture']['recommend_rate'] * 100:.1f}%"
        sim_cov = f"{cov['simultaneous_interval_coverage'] * 100:.1f}%"
        pw_cov = f"{cov['pointwise_coverage_at_q0'] * 100:.1f}%"
        equiv = (
            "YES"
            if duality["exact_decision_equivalence"]
            else f"{duality['agreement_rate'] * 100:.1f}%"
        )
        row = (
            f"{sc['scenario']:<28}{sc['n_samples']:>5}  {sc['margin']:>6.3f}  "
            f"{gt['true_delta']:>+10.4f} {gt['worst_true_delta_deploy']:>+11.4f}  "
            f"{rec_a:>6} {rec_b:>6} {rec_c:>6} {rec_d:>6} {rec_e:>6} {rec_f:>6}  "
            f"{sim_cov:>7} {pw_cov:>6} {equiv:>9}"
        )
        lines.append(row)
    if run["warnings"]:
        lines += ["", "Warnings:"] + [f"  - {w}" for w in run["warnings"]]
    lines += ["", f"Reproduce: {run['reproduction_command']}"]
    return "\n".join(lines)


def _format_reliability_report(run: dict[str, Any]) -> str:
    lines = [
        f"Run:      {run['run_id']}",
        (
            f"Config:   sha256 {run['config_sha256'][:12]}  seed={run['config']['seed']}  "
            f"replications={run['config'].get('n_replications', 20)}  "
            f"deploy_eps={run['config'].get('deploy_threshold_epsilon', 0.10)}"
        ),
        f"Results:  sha256 {run['results_sha256'][:12]}",
        f"Git:      {run['provenance']['git']}",
        "",
    ]
    header = (
        f"{'scenario':<28}{'N':>5}  {'margin':>6}  {'true Delta':>10}  "
        f"{'Meth A':>7} {'Meth B':>7} {'Meth C':>7} {'Meth D':>7} {'Meth E':>7}  "
        f"{'max |ana-num|':>13} {'DiagFail':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for sc in run["results"]["scenarios"]:
        gt = sc["ground_truth"]
        m = sc["methods"]
        rec_a = f"{m['method_a_point_estimate']['recommend_rate'] * 100:.1f}%"
        rec_b = f"{m['method_b_baseline_ci']['recommend_rate'] * 100:.1f}%"
        rec_c = f"{m['method_c_worst_stratum']['recommend_rate'] * 100:.1f}%"
        rec_d = f"{m['method_d_pointwise_reversal']['recommend_rate'] * 100:.1f}%"
        rec_e = f"{m['method_e_simultaneous_band']['recommend_rate'] * 100:.1f}%"
        disc = sc["reversal_search_validation"]["max_analytical_discrepancy"]
        disc_str = f"{disc:.2e}" if disc is not None else "0.0"
        diag_fail = f"{sc['diagnostics']['diagnostic_gate_failed_rate'] * 100:.1f}%"
        row = (
            f"{sc['scenario']:<28}{sc['n_samples']:>5}  {sc['margin']:>6.3f}  "
            f"{gt['true_delta']:>+10.4f}  "
            f"{rec_a:>7} {rec_b:>7} {rec_c:>7} {rec_d:>7} {rec_e:>7}  "
            f"{disc_str:>13} {diag_fail:>8}"
        )
        lines.append(row)
    if run["warnings"]:
        lines += ["", "Warnings:"] + [f"  - {w}" for w in run["warnings"]]
    lines += ["", f"Reproduce: {run['reproduction_command']}"]
    return "\n".join(lines)


def _format_parity_report(run: dict[str, Any]) -> str:
    lines = [
        f"Run:      {run['run_id']}",
        f"Config:   sha256 {run['config_sha256'][:12]}",
        f"Results:  sha256 {run['results_sha256'][:12]}",
        f"Git:      {run['provenance']['git']}",
        f"Lock:     {run['provenance'].get('uv_lock_sha256')}",
        "",
    ]
    header = (
        f"{'policy':<8}{'campaign':<8}{'eval':<18}{'slice':<6}{'est':<6}"
        f"{'ours':>12}{'obp':>12}{'abs diff':>12}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in run["results"]["comparisons"]:
        for name, ours_key, obp_key in (
            ("ips", "ips", "ipw"),
            ("snips", "snips", "snipw"),
        ):
            ours = row["reversallab"][ours_key]
            reference = row["obp"][obp_key]
            prefix = (
                f"{row['behavior_policy']:<8}{row['campaign']:<8}{row['evaluation_policy']:<18}"
                f"{row['position']!s:<6}{name:<6}"
            )
            if ours is None or reference is None:
                lines.append(f"{prefix}{'undefined':>12}{'undefined':>12}{'n/a':>12}")
            else:
                lines.append(f"{prefix}{ours:12.6g}{reference:12.6g}{abs(ours - reference):12.3g}")
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


def _cmd_validate_advanced_estimators(args: argparse.Namespace) -> int:
    config = load_advanced_config(args.config)
    if args.n_replications is not None:
        try:
            config = type(config).model_validate(
                {**config.model_dump(), "n_replications": args.n_replications}
            )
        except ValidationError as exc:
            raise ConfigurationError(f"Invalid --n-replications override:\n{exc}") from exc
    config_dict = to_jsonable(config.model_dump(mode="json"))
    logger.info(
        "experiment=%s seed=%d replications=%d folds=%d",
        config.name,
        config.seed,
        config.n_replications,
        config.n_folds,
    )
    output = run_advanced_estimator_validation(config)
    command = f"reversallab validate-advanced-estimators --config {Path(args.config).as_posix()}"
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


def _cmd_calibrate_uncertainty(args: argparse.Namespace) -> int:
    config = load_uncertainty_config(args.config)
    overrides: dict[str, Any] = {}
    if args.n_replications is not None:
        overrides["n_replications"] = args.n_replications
    if args.n_bootstraps is not None:
        overrides["n_bootstraps"] = args.n_bootstraps
    if overrides:
        try:
            config = type(config).model_validate({**config.model_dump(), **overrides})
        except ValidationError as exc:
            raise ConfigurationError(f"Invalid CLI argument overrides:\n{exc}") from exc
    config_dict = to_jsonable(config.model_dump(mode="json"))
    logger.info(
        "experiment=%s seed=%d replications=%d bootstraps=%s",
        config.name,
        config.seed,
        config.n_replications,
        config.n_bootstraps,
    )
    output = run_uncertainty_calibration(config)
    command = f"reversallab calibrate-uncertainty --config {Path(args.config).as_posix()}"
    if args.n_replications is not None:
        command += f" --n-replications {args.n_replications}"
    if args.n_bootstraps is not None:
        command += f" --n-bootstraps {args.n_bootstraps}"
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


def _cmd_compare_reference(args: argparse.Namespace) -> int:
    config = load_parity_config(args.config)
    config_dict = to_jsonable(config.model_dump(mode="json"))
    logger.info("experiment=%s", config.name)
    output = run_open_bandit_parity(config)
    command = f"reversallab compare-reference --config {Path(args.config).as_posix()}"
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
    return 0 if output.matched else 1


def _cmd_run_reversal(args: argparse.Namespace) -> int:
    config = load_reversal_config(args.config)
    overrides: dict[str, Any] = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if overrides:
        try:
            config = type(config).model_validate({**config.model_dump(), **overrides})
        except ValidationError as exc:
            raise ConfigurationError(f"Invalid CLI argument overrides:\n{exc}") from exc
    config_dict = to_jsonable(config.model_dump(mode="json"))
    logger.info(
        "experiment=%s seed=%d scenarios=%d",
        config.name,
        config.seed,
        len(config.scenarios),
    )
    output = run_reversal_prototype(config)
    command = f"reversallab run-reversal --config {Path(args.config).as_posix()}"
    if args.seed is not None:
        command += f" --seed {args.seed}"
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


def _cmd_run_reliability(args: argparse.Namespace) -> int:
    config = load_reliability_config(args.config)
    overrides: dict[str, Any] = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.n_replications is not None:
        overrides["n_replications"] = args.n_replications
    if overrides:
        try:
            config = type(config).model_validate({**config.model_dump(), **overrides})
        except ValidationError as exc:
            raise ConfigurationError(f"Invalid CLI argument overrides:\n{exc}") from exc
    config_dict = to_jsonable(config.model_dump(mode="json"))
    logger.info(
        "experiment=%s seed=%d replications=%d scenarios=%d",
        config.name,
        config.seed,
        config.n_replications,
        len(config.scenarios),
    )
    output = run_reversal_reliability(config)
    command = f"reversallab run-reliability --config {Path(args.config).as_posix()}"
    if args.seed is not None:
        command += f" --seed {args.seed}"
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


def _cmd_run_independent_reliability(args: argparse.Namespace) -> int:
    config = load_independent_reliability_config(args.config)
    overrides: dict[str, Any] = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.n_replications is not None:
        overrides["n_replications"] = args.n_replications
    if overrides:
        try:
            config = type(config).model_validate({**config.model_dump(), **overrides})
        except ValidationError as exc:
            raise ConfigurationError(f"Invalid CLI argument overrides:\n{exc}") from exc
    config_dict = to_jsonable(config.model_dump(mode="json"))
    logger.info(
        "experiment=%s seed=%d replications=%d scenarios=%d",
        config.name,
        config.seed,
        config.n_replications,
        len(config.scenarios),
    )
    output = run_independent_reliability(config)
    command = f"reversallab run-independent-reliability --config {Path(args.config).as_posix()}"
    if args.seed is not None:
        command += f" --seed {args.seed}"
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

    p_adv = sub.add_parser(
        "validate-advanced-estimators",
        help="Validate DM, DR, and cross-fitting against ground truth (experiment 003).",
        description=(
            "Simulate logged bandit data, run K-fold cross-fitting for reward modeling, "
            "and compare IPS, SNIPS, DM, and DR under good/poor overlap and "
            "correct/misspecified models."
        ),
    )
    p_adv.add_argument("--config", required=True, help="Path to an experiment YAML file.")
    p_adv.add_argument(
        "--output-dir", default="runs", help="Directory in which a new run directory is created."
    )
    p_adv.add_argument(
        "--n-replications",
        type=int,
        default=None,
        help="Override the configured number of replications (recorded in the run config).",
    )
    p_adv.set_defaults(func=_cmd_validate_advanced_estimators)

    p_unc = sub.add_parser(
        "calibrate-uncertainty",
        help="Benchmark uncertainty and CI calibration against ground truth (Exp 004).",
        description=(
            "Simulate logged bandit data, evaluate Wald and bootstrap confidence intervals "
            "across good/poor overlap, correct/misspecified models, and different sample sizes."
        ),
    )
    p_unc.add_argument("--config", required=True, help="Path to an experiment YAML file.")
    p_unc.add_argument(
        "--output-dir", default="runs", help="Directory in which a new run directory is created."
    )
    p_unc.add_argument(
        "--n-replications",
        type=int,
        default=None,
        help="Override the configured number of replications (recorded in the run config).",
    )
    p_unc.add_argument(
        "--n-bootstraps",
        type=int,
        default=None,
        help="Override the configured number of bootstrap draws.",
    )
    p_unc.set_defaults(func=_cmd_calibrate_uncertainty)

    p_ref = sub.add_parser(
        "compare-reference",
        help="Compare IPS/SNIPS with Open Bandit Pipeline on the small sample.",
        description="Load the small Open Bandit Dataset files named in the config, "
        "evaluate the named policies with ReversalLab and with OBP's IPW and SNIPW "
        "on the same arrays, and write the differences. Exits 1 if any difference "
        "exceeds the configured tolerance. Does not download the full dataset.",
    )
    p_ref.add_argument("--config", required=True, help="Path to an experiment YAML file.")
    p_ref.add_argument(
        "--output-dir", default="runs", help="Directory in which a new run directory is created."
    )
    p_ref.set_defaults(func=_cmd_compare_reference)

    p_rev = sub.add_parser(
        "run-reversal",
        help="Run decision-reversal research prototype (experiment 005).",
        description=(
            "Evaluate paired policy-value differences, baseline deployment decisions, "
            "and 1-D context mixture perturbation search across synthetic scenarios."
        ),
    )
    p_rev.add_argument("--config", required=True, help="Path to an experiment YAML file.")
    p_rev.add_argument(
        "--output-dir", default="runs", help="Directory in which a new run directory is created."
    )
    p_rev.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the configured random seed.",
    )
    p_rev.set_defaults(func=_cmd_run_reversal)

    p_rel = sub.add_parser(
        "run-reliability",
        help="Run decision-reversal reliability benchmark (experiment 006).",
        description=(
            "Multi-replication Monte Carlo comparison of 5 decision rules across scenarios, "
            "evaluating false recommendations, withholding rates, and search exactness."
        ),
    )
    p_rel.add_argument("--config", required=True, help="Path to an experiment YAML file.")
    p_rel.add_argument(
        "--output-dir", default="runs", help="Directory in which a new run directory is created."
    )
    p_rel.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the configured random seed.",
    )
    p_rel.add_argument(
        "--n-replications",
        type=int,
        default=None,
        help="Override the configured number of replications.",
    )
    p_rel.set_defaults(func=_cmd_run_reliability)

    p_ind = sub.add_parser(
        "run-independent-reliability",
        help="Run independent reliability evaluation benchmark (experiment 007, M4C).",
        description=(
            "Preregistered Monte Carlo benchmark comparing 6 decision methods (A-F), "
            "evaluating simultaneous coverage, false recommendation rates, and duality."
        ),
    )
    p_ind.add_argument("--config", required=True, help="Path to an experiment YAML file.")
    p_ind.add_argument(
        "--output-dir", default="runs", help="Directory in which a new run directory is created."
    )
    p_ind.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the configured random seed.",
    )
    p_ind.add_argument(
        "--n-replications",
        type=int,
        default=None,
        help="Override the configured number of replications.",
    )
    p_ind.set_defaults(func=_cmd_run_independent_reliability)

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
