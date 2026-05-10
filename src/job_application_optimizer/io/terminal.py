"""Terminal progress display helpers."""

import os
import sys
from datetime import datetime

from job_application_optimizer.models import JobRecord


RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"


def _use_color() -> bool:
    return os.getenv("NO_COLOR") is None and (sys.stdout.isatty() or os.getenv("FORCE_COLOR") == "1")


def _color(text: str, color_code: str) -> str:
    if not _use_color():
        return text
    return f"{color_code}{text}{RESET}"


def _clean(value: object) -> str:
    text = str(value or "-").strip()
    text = " ".join(text.replace("|", "/").split())
    return text or "-"


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {remaining:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {remaining:02d}s"


def format_score(score: object, target_score: float) -> str:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "-"
    text = f"{value:.1f}"
    return _color(text, GREEN if value >= target_score else RED)


def _format_status(status: str) -> str:
    status = _clean(status)
    if status in {"completed", "written", "ready"}:
        return _color(status, GREEN)
    if status in {"failed", "error"}:
        return _color(status, RED)
    return status


def _print_row(values: list[object]) -> None:
    print(" | ".join(_clean(value) for value in values), flush=True)


def current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_batch_job_start(index: int, total_jobs: int, job: JobRecord) -> None:
    _print_row([current_timestamp(), "start job", f"{index}/{total_jobs}", job.job_id, job.url])


def print_interview_job_start(index: int, total_jobs: int, job: JobRecord, folder: object) -> None:
    _print_row([current_timestamp(), "start interview", f"{index}/{total_jobs}", job.job_id, job.company, job.role, folder])


def print_batch_header(total_jobs: int) -> None:
    print(f"JDs to process: {total_jobs}", flush=True)
    print("date | id | company | job | ats_scores | screening_recommendation | status | time", flush=True)


def print_batch_job_result(
    run_date: str,
    job: JobRecord,
    ats_score: object,
    screening_recommendation: str,
    elapsed_seconds: float,
    target_score: float,
    status: str = "completed",
) -> None:
    _print_row(
        [
            run_date,
            job.job_id,
            job.company,
            job.role,
            format_score(ats_score, target_score),
            screening_recommendation,
            _format_status(status),
            format_duration(elapsed_seconds),
        ]
    )


def print_interview_header(total_jobs: int) -> None:
    print(f"Interview prep jobs to process: {total_jobs}", flush=True)
    print("date | id | company | job | status | time", flush=True)


def print_interview_job_result(run_date: str, job: JobRecord, elapsed_seconds: float, status: str = "written") -> None:
    _print_row(
        [
            run_date,
            job.job_id,
            job.company,
            job.role,
            _format_status(status),
            format_duration(elapsed_seconds),
        ]
    )


def print_run_footer(label: str, total_jobs: int, elapsed_seconds: float) -> None:
    print(f"{label} finished: {total_jobs} job(s) in {format_duration(elapsed_seconds)}", flush=True)


__all__ = [
    "current_timestamp",
    "format_duration",
    "format_score",
    "print_batch_header",
    "print_batch_job_start",
    "print_batch_job_result",
    "print_interview_header",
    "print_interview_job_start",
    "print_interview_job_result",
    "print_run_footer",
]
