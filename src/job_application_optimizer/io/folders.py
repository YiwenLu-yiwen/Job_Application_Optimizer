"""Output folder naming helpers."""

import re

from job_application_optimizer.models import JobRecord


def _sanitize_folder_part(value: str, fallback: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        cleaned = fallback
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", cleaned)
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = cleaned.strip("._")
    return cleaned or fallback


def build_job_folder_name(job: JobRecord, used_names: set[str]) -> str:
    company = _sanitize_folder_part(job.company, "UnknownCompany")
    role = _sanitize_folder_part(job.role, "UnknownRole")
    base = f"{company}_{role}"

    if base not in used_names:
        used_names.add(base)
        return base

    idx = 2
    while f"{base}_{idx}" in used_names:
        idx += 1
    unique_name = f"{base}_{idx}"
    used_names.add(unique_name)
    return unique_name


__all__ = ["_sanitize_folder_part", "build_job_folder_name"]
