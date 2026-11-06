"""Integration test verifying end-to-end execution of the tutorial notebook.

Executes using nbclient with a genuine Jupyter kernel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import nbformat
from nbclient import NotebookClient


def test_quickstart_tutorial_notebook_jupyter_kernel_execution(tmp_path: Path) -> None:
    """Execute the tutorial notebook using nbclient with a real python3 kernel."""
    notebook_path = Path("notebooks/quickstart_decision_reversal.ipynb")
    assert notebook_path.exists(), f"Notebook file not found at {notebook_path}"

    nb = nbformat.read(str(notebook_path), as_version=4)  # type: ignore[no-untyped-call]

    # Initialize genuine NotebookClient
    client = NotebookClient(
        nb,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(Path.cwd())}},
    )

    # Execute all cells in a live kernel
    client.execute()

    code_cells = [cell for cell in nb.cells if cell.get("cell_type") == "code"]
    assert len(code_cells) >= 8, f"Expected at least 8 code cells, found {len(code_cells)}"

    # Check for execution errors in any cell
    for idx, cell in enumerate(code_cells, start=1):
        assert cell.get("execution_count") is not None, f"Code cell {idx} was not executed"
        outputs = cast(list[dict[str, Any]], cell.get("outputs", []))
        for output in outputs:
            assert output.get("output_type") != "error", (
                f"Error in cell {idx}: {output.get('ename')}: {output.get('evalue')}"
            )

    # Verify key output content from the executed notebook
    all_stream_text = " ".join(
        output.get("text", "")
        for cell in code_cells
        for output in cast(list[dict[str, Any]], cell.get("outputs", []))
        if output.get("output_type") == "stream"
    )

    assert "N = 2500" in all_stream_text
    assert "RECOMMEND" in all_stream_text
    assert "INCONCLUSIVE" in all_stream_text
    assert "Active propensity deficit" in all_stream_text

    # Clean up generated tutorial report if produced during execution
    report_file = Path("runs/tutorial_audit_report.json")
    if report_file.exists():
        report_file.unlink()
