"""Integration test verifying that the tutorial notebook executes end-to-end without errors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def test_quickstart_tutorial_notebook_executes(tmp_path: Path) -> None:
    notebook_path = Path("notebooks/quickstart_decision_reversal.ipynb")
    assert notebook_path.exists(), "Notebook file missing."

    content = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert content["nbformat"] == 4
    assert len(content["cells"]) > 0

    code_cells = [cell for cell in content["cells"] if cell.get("cell_type") == "code"]
    assert len(code_cells) >= 7

    # Execute all code cells sequentially in a shared namespace
    global_namespace: dict[str, Any] = {"__file__": str(notebook_path)}
    for idx, cell in enumerate(code_cells, start=1):
        code_text = "".join(cell.get("source", []))
        try:
            exec(code_text, global_namespace)
        except Exception as exc:  # pragma: no cover
            msg = f"Notebook code cell {idx} failed with error: {exc}\nCode:\n{code_text}"
            raise AssertionError(msg) from exc

    # Verify that expected variables exist in the execution namespace and match verified results
    assert "dataset" in global_namespace
    assert global_namespace["dataset"].n_samples == 2500
    assert global_namespace["dataset"].n_actions == 3
    assert global_namespace["dataset"].context_dim == 2

    assert "comparison_report" in global_namespace
    comp = global_namespace["comparison_report"]
    assert comp.schema_version == "1.0.0"
    assert comp.decision["outcome"] == "RECOMMEND"
    assert comp.diagnostics["gate_passed"] is True
    assert comp.reversal_search["status"] == "NO_REVERSAL_FOUND_WITHIN_BOUNDS"
    assert comp.robust_mixture["outcome"] == "RECOMMEND"

    assert "gated_comparison" in global_namespace
    gated = global_namespace["gated_comparison"]
    assert gated.decision["outcome"] == "INCONCLUSIVE"
    assert gated.diagnostics["gate_passed"] is False
    assert len(gated.diagnostics["rejection_reasons"]) > 0

    # Clean up written artifact if present
    report_file = Path("runs/tutorial_audit_report.json")
    if report_file.exists():
        report_file.unlink()
