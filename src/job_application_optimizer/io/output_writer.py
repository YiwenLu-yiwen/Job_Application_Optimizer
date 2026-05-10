"""Output artifact writers."""

import json
from pathlib import Path
from typing import Any

from job_application_optimizer.io.csv_writer import append_completed_all_csv, write_completed_csv, write_ranked_csv


def write_output(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def flush_progress(
    out_dir: Path,
    completed_csv_path: Path,
    completed_all_csv_path: Path,
    summary_rows: list[dict[str, Any]],
    completed_rows: list[dict[str, Any]],
    latest_completed_row: dict[str, Any] | None = None,
) -> None:
    write_output(out_dir / "batch_summary.json", json.dumps(summary_rows, ensure_ascii=False, indent=2))
    write_completed_csv(completed_csv_path, completed_rows)
    write_ranked_csv(out_dir / "ranked_jobs.csv", completed_rows)
    if latest_completed_row is not None:
        append_completed_all_csv(completed_all_csv_path, latest_completed_row)


__all__ = ["flush_progress", "write_output"]
