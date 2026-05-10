"""Shared data models."""

from dataclasses import dataclass


@dataclass
class JobRecord:
    job_id: str
    url: str
    company: str
    role: str
    location: str


COMPLETED_FIELDNAMES = [
    "run_date",
    "job_id",
    "url",
    "company",
    "role",
    "location",
    "original_score",
    "optimized_score",
    "target_score",
    "meets_target",
    "accepted_resume_version",
    "gap_type",
    "gap_summary",
    "screening_recommendation",
    "status",
    "generation_mode",
    "output_folder",
    "error",
]


__all__ = ["COMPLETED_FIELDNAMES", "JobRecord"]
