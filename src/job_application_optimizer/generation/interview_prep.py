"""Generate interview prep from an existing job output folder.

This module is intentionally separate from the batch pipeline. It only reads
existing artifacts and writes `interview_prep.md`; it does not fetch job pages,
score ATS fit, regenerate resumes, or rewrite cover letters.
"""

import argparse
import csv
import os
import textwrap
from datetime import datetime
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

from job_application_optimizer.config import _prompt_text
from job_application_optimizer.llm.client import require_llm_generate, require_openai_client
from job_application_optimizer.io.run_logger import log_event, setup_run_logger
from job_application_optimizer.io.terminal import (
    format_duration,
    print_interview_header,
    print_interview_job_result,
    print_interview_job_start,
    print_run_footer,
)
from job_application_optimizer.models import JobRecord


load_dotenv()


def infer_job_from_folder(job_folder: Path) -> JobRecord:
    summary_path = job_folder / "analysis_summary.csv"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            row = rows[0]
            return JobRecord(
                job_id=job_folder.name,
                url=row.get("url", ""),
                company=row.get("company", "") or "Unknown Company",
                role=row.get("role", "") or "Unknown Role",
                location=row.get("location", "") or "Unknown",
            )

    parts = job_folder.name.split("_", 1)
    company = parts[0] if parts and parts[0] else "Unknown Company"
    role = parts[1] if len(parts) > 1 and parts[1] else "Unknown Role"
    return JobRecord(job_id=job_folder.name, url="", company=company, role=role, location="Unknown")


def build_existing_folder_interview_prompt(
    job: JobRecord,
    job_text: str,
    tailored_resume: str,
    cv_understanding: str,
) -> str:
    job_excerpt = _prompt_text(job_text)
    resume_excerpt = _prompt_text(tailored_resume)
    cv_understanding_excerpt = _prompt_text(cv_understanding)

    return textwrap.dedent(
        f"""
        Company: {job.company}
        Role: {job.role}
        Location: {job.location}

        Create a comprehensive interview prep pack for this already-tailored application.
        Use the tailored resume as the primary candidate evidence and the JD as the role context.

        REQUIRED OUTPUT STRUCTURE:
        1) Role Fit Snapshot
           - 4 to 6 strengths and 2 to 3 risks for this role.
           - Every point must be grounded in the tailored resume or CV understanding.

        2) Why This Company + Motivation Narrative
           - A polished answer to "Why do you want to join {job.company}?" (180 to 260 words).
           - Include:
             a) What specifically about the company and role is attractive.
             b) Why the candidate is motivated beyond compensation/title.
             c) What impact the candidate wants to create in the first 12 months.
           - Add a short "Evidence to mention" list (3 to 5 bullets).

        3) Core Questions
           - Question 1: Why do you want to join {job.company}?
             Provide a high-quality sample answer and 3 concise talking-point variants.
           - Question 2: Tell me about yourself.
             Provide a 2-minute narrative answer plus a 30-second version.

        4) Additional High-Probability Questions
           - Provide 6 role-relevant questions with STAR-style answer frameworks.
           - For each: include "What interviewer is testing" and "How to answer".

        5) Online Assessment Preparation
           - List likely assessment themes for this role and sample question styles.

        6) Final Coaching Notes
           - Common pitfalls to avoid.
           - Final personalization tips before interview.

        Constraints:
        - Use only facts supported by the tailored resume and CV understanding.
        - Do not fabricate employers, projects, metrics, leadership scope, or tech stack.
        - Keep the tone practical and interview-ready.
        - Do not use generic labels like "strong fit"; show fit through specific evidence.

        JD:
        {job_excerpt}

        Tailored resume:
        {resume_excerpt}

        Candidate deep understanding:
        {cv_understanding_excerpt}
        """
    ).strip()


def generate_interview_for_folder(
    job_folder: Path,
    cv_understanding_path: Path,
    model: str,
    overwrite: bool = False,
) -> Path:
    job_folder = job_folder.resolve()
    output_path = job_folder / "interview_prep.md"
    if output_path.exists() and not overwrite:
        return output_path

    job_text_path = job_folder / "job_text.txt"
    resume_path = job_folder / "tailored_resume.md"
    if not job_text_path.exists():
        raise FileNotFoundError(f"Missing required file: {job_text_path}")
    if not resume_path.exists():
        raise FileNotFoundError(f"Missing required file: {resume_path}")
    if not cv_understanding_path.exists():
        raise FileNotFoundError(f"Missing CV understanding file: {cv_understanding_path}")

    job = infer_job_from_folder(job_folder)
    prompt = build_existing_folder_interview_prompt(
        job=job,
        job_text=job_text_path.read_text(encoding="utf-8"),
        tailored_resume=resume_path.read_text(encoding="utf-8"),
        cv_understanding=cv_understanding_path.read_text(encoding="utf-8"),
    )
    client = require_openai_client()
    interview_pack = require_llm_generate(client, model, prompt, "interview prep", temperature=0.3)
    output_path.write_text(interview_pack, encoding="utf-8")
    return output_path


def iter_job_folders(run_dir: Path) -> list[Path]:
    run_dir = run_dir.resolve()
    return sorted(
        path
        for path in run_dir.iterdir()
        if path.is_dir() and (path / "job_text.txt").exists() and (path / "tailored_resume.md").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate interview_prep.md from existing job output folders")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job-folder", help="Existing job output folder containing job_text.txt and tailored_resume.md")
    group.add_argument("--run-dir", help="Run directory containing multiple job output folders, e.g. outputs/2026-05-06")
    parser.add_argument(
        "--cv-understanding",
        default="data/cv_deep_understanding.md",
        help="Path to CV deep-understanding cache file",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate interview_prep.md even if it already exists")
    args = parser.parse_args()

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    cv_understanding_path = Path(args.cv_understanding).resolve()

    logger, log_path = setup_run_logger("interview_prep")
    print(f"Log file: {log_path}", flush=True)
    run_date = datetime.now().strftime("%Y-%m-%d")

    if args.job_folder:
        folder = Path(args.job_folder)
        print_interview_header(1)
        run_started_at = perf_counter()
        job_started_at = perf_counter()
        output_path = folder.resolve() / "interview_prep.md"
        status = "skipped" if output_path.exists() and not args.overwrite else "written"
        job = infer_job_from_folder(folder)
        print_interview_job_start(1, 1, job, folder)
        log_event(
            logger,
            "interview_job_start",
            index=1,
            total=1,
            job_id=job.job_id,
            url=job.url,
            company=job.company,
            role=job.role,
            folder=folder,
            overwrite=args.overwrite,
        )
        output = generate_interview_for_folder(
            folder,
            cv_understanding_path=cv_understanding_path,
            model=model,
            overwrite=args.overwrite,
        )
        elapsed = perf_counter() - job_started_at
        log_event(logger, "interview_job_finished", job_id=job.job_id, status=status, output=output, duration=format_duration(elapsed))
        print_interview_job_result(run_date, job, elapsed, status=status)
        total_elapsed = perf_counter() - run_started_at
        log_event(logger, "interview_run_finished", total_jobs=1, duration=format_duration(total_elapsed))
        print_run_footer("Interview prep", 1, total_elapsed)
        print(f"Interview prep ready: {output}")
        print(f"Log file: {log_path}")
        return

    folders = iter_job_folders(Path(args.run_dir))
    print_interview_header(len(folders))
    run_started_at = perf_counter()
    log_event(logger, "interview_run_start", run_dir=args.run_dir, total_jobs=len(folders), overwrite=args.overwrite)
    generated = []
    skipped = []
    for index, folder in enumerate(folders, start=1):
        job_started_at = perf_counter()
        job = infer_job_from_folder(folder)
        print_interview_job_start(index, len(folders), job, folder)
        log_event(
            logger,
            "interview_job_start",
            index=index,
            total=len(folders),
            job_id=job.job_id,
            url=job.url,
            company=job.company,
            role=job.role,
            folder=folder,
            overwrite=args.overwrite,
        )
        output = folder / "interview_prep.md"
        if output.exists() and not args.overwrite:
            skipped.append(output)
            elapsed = perf_counter() - job_started_at
            log_event(logger, "interview_job_finished", index=index, job_id=job.job_id, status="skipped", output=output, duration=format_duration(elapsed))
            print_interview_job_result(run_date, job, elapsed, status="skipped")
            continue
        generated_output = generate_interview_for_folder(
            folder,
            cv_understanding_path=cv_understanding_path,
            model=model,
            overwrite=args.overwrite,
        )
        generated.append(generated_output)
        elapsed = perf_counter() - job_started_at
        log_event(logger, "interview_job_finished", index=index, job_id=job.job_id, status="written", output=generated_output, duration=format_duration(elapsed))
        print_interview_job_result(run_date, job, elapsed, status="written")

    total_elapsed = perf_counter() - run_started_at
    log_event(
        logger,
        "interview_run_finished",
        total_jobs=len(folders),
        generated=len(generated),
        skipped=len(skipped),
        duration=format_duration(total_elapsed),
    )
    print_run_footer("Interview prep", len(folders), total_elapsed)
    print(f"Generated: {len(generated)}")
    print(f"Skipped existing: {len(skipped)}")
    for output in generated:
        print(f"- {output}")
    print(f"Log file: {log_path}")


if __name__ == "__main__":
    main()


__all__ = [
    "build_existing_folder_interview_prompt",
    "generate_interview_for_folder",
    "infer_job_from_folder",
    "iter_job_folders",
    "main",
]
