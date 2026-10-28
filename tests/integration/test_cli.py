from __future__ import annotations

import json
from pathlib import Path

import pytest

from reversallab.cli import main

CONFIG = """
name: cli_smoke
seed: 7
n_replications: 20
reference_n_contexts: 20000
estimators: [ips, snips]
environment: {n_actions: 4, context_dim: 2, reward_scale: 1.0}
scenarios:
  - name: uniform_logging
    n_samples: [50, 200]
    logging_policy: {kind: uniform}
    evaluation_policy: {kind: softmax, inverse_temperature: 2.0, scores: reward}
  - name: greedy_candidate
    n_samples: [20]
    logging_policy: {kind: softmax, inverse_temperature: 3.0, scores: random, score_seed: 1}
    evaluation_policy: {kind: epsilon_greedy, epsilon: 0.0, scores: reward}
"""


def _run(tmp_path: Path, config_text: str, out: str) -> dict[str, object]:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(config_text, encoding="utf-8")
    code = main(
        [
            "--log-level",
            "WARNING",
            "validate-estimators",
            "--config",
            str(cfg),
            "--output-dir",
            str(tmp_path / out),
        ]
    )
    assert code == 0
    (run_dir,) = (tmp_path / out).iterdir()
    payload: dict[str, object] = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    return payload


def test_end_to_end_run_writes_complete_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _run(tmp_path, CONFIG, "runs")
    for key in (
        "run_id",
        "config",
        "config_sha256",
        "results",
        "results_sha256",
        "warnings",
        "runtime",
        "provenance",
        "reproduction_command",
    ):
        assert key in run
    results = run["results"]
    assert isinstance(results, dict)
    scenarios = results["scenarios"]
    assert [s["name"] for s in scenarios] == ["uniform_logging", "greedy_candidate"]
    cell = scenarios[0]["cells"][0]
    assert cell["estimators"]["ips"]["n_successful"] == 20
    assert 0.0 <= cell["estimators"]["ips"]["ci_coverage"] <= 1.0
    assert "Reproduce:" in capsys.readouterr().out


def test_identical_config_gives_identical_results(tmp_path: Path) -> None:
    a = _run(tmp_path, CONFIG, "a")
    b = _run(tmp_path, CONFIG, "b")
    assert a["results"] == b["results"]
    assert a["results_sha256"] == b["results_sha256"]
    assert a["config_sha256"] == b["config_sha256"]


def test_scenario_results_independent_of_other_scenarios(tmp_path: Path) -> None:
    full = _run(tmp_path, CONFIG, "full")
    only_first = CONFIG.split("  - name: greedy_candidate")[0]
    partial = _run(tmp_path, only_first, "partial")
    assert full["results"]["scenarios"][0] == partial["results"]["scenarios"][0]  # type: ignore[index]


def test_report_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = _run(tmp_path, CONFIG, "runs")
    capsys.readouterr()
    assert main(["report", "--run-dir", str(tmp_path / "runs" / str(run["run_id"]))]) == 0
    assert "uniform_logging" in capsys.readouterr().out


def test_invalid_config_reports_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        CONFIG.replace("estimators: [ips, snips]", "estimators: [magic]"), encoding="utf-8"
    )
    assert main(["validate-estimators", "--config", str(cfg), "--output-dir", str(tmp_path)]) == 2
    assert "unknown estimator" in capsys.readouterr().err


def test_missing_run_dir_reports_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["report", "--run-dir", str(tmp_path / "nope")]) == 2
    assert "No run.json" in capsys.readouterr().err
