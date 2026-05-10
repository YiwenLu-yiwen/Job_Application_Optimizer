"""Batch job application pipeline."""

import argparse
import json
import logging
import os
from datetime import datetime
from time import perf_counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from job_application_optimizer.generation.prompt_bundle import build_prompt_bundle
from job_application_optimizer.io.csv_writer import (
    write_analysis_summary_csv,
    write_completed_csv,
    write_evaluation_history_csv,
    write_ranked_csv,
    write_requirements_evidence_csv,
)
from job_application_optimizer.io.folders import build_job_folder_name
from job_application_optimizer.io.output_writer import flush_progress, write_output
from job_application_optimizer.io.run_logger import log_event, setup_run_logger
from job_application_optimizer.io.terminal import (
    format_duration,
    print_batch_header,
    print_batch_job_result,
    print_batch_job_start,
    print_run_footer,
)
from job_application_optimizer.jobs.fetcher import fetch_job_page
from job_application_optimizer.jobs.metadata import infer_job_meta_from_html, llm_extract_job_metadata
from job_application_optimizer.jobs.parser import extract_clean_job_text, parse_urls_file
from job_application_optimizer.jobs.requirements import llm_extract_job_requirements
from job_application_optimizer.llm.client import require_llm_generate, require_openai_client
from job_application_optimizer.models import JobRecord
from job_application_optimizer.resume.optimizer import optimize_resume_content
from job_application_optimizer.resume.reader import read_resume_text
from job_application_optimizer.resume.understanding import load_or_generate_cv_understanding
from job_application_optimizer.scoring.ats import ats_stop_condition_met, llm_ats_score
from job_application_optimizer.scoring.evidence import llm_map_resume_evidence
from job_application_optimizer.scoring.gap import build_gap_summary

load_dotenv()


def run_batch(
    jobs: list[JobRecord],
    resume_path: Path,
    out_dir: Path,
    completed_csv_path: Path,
    completed_all_csv_path: Path,
    cv_understanding_path: Path,
    run_date: str,
    include_interview_prep: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    processable_jobs = [job for job in jobs if job.url]
    print_batch_header(len(processable_jobs))
    run_started_at = perf_counter()
    log_event(
        logger,
        "batch_start",
        total_jobs=len(processable_jobs),
        resume_path=resume_path,
        out_dir=out_dir,
        completed_csv=completed_csv_path,
        include_interview_prep=include_interview_prep,
    )

    stage_started_at = perf_counter()
    resume_text = read_resume_text(resume_path)
    log_event(logger, "resume_read", path=resume_path, chars=len(resume_text), duration=format_duration(perf_counter() - stage_started_at))

    client = require_openai_client()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    target_score = float(os.getenv("TARGET_RESUME_SCORE", "80"))
    stage_started_at = perf_counter()
    cv_understanding = load_or_generate_cv_understanding(resume_path, client, model, cv_understanding_path)
    log_event(
        logger,
        "cv_understanding_ready",
        path=cv_understanding_path,
        chars=len(cv_understanding),
        duration=format_duration(perf_counter() - stage_started_at),
    )

    summary_rows = []
    completed_rows = []
    used_folder_names: set[str] = set()
    for index, job in enumerate(processable_jobs, start=1):
        job_started_at = perf_counter()
        print_batch_job_start(index, len(processable_jobs), job)
        log_event(logger, "job_start", index=index, total=len(processable_jobs), job_id=job.job_id, url=job.url)
        output_folder = ""
        try:
            stage_started_at = perf_counter()
            raw_html = fetch_job_page(job.url)
            log_event(
                logger,
                "stage_done",
                job_id=job.job_id,
                stage="fetch_job_page",
                chars=len(raw_html),
                duration=format_duration(perf_counter() - stage_started_at),
            )
            inferred_role, inferred_company = infer_job_meta_from_html(raw_html, job.role, job.company)
            if not job.role:
                job.role = inferred_role
            if not job.company:
                job.company = inferred_company
            if not job.location:
                job.location = "Unknown"

            stage_started_at = perf_counter()
            job_text = extract_clean_job_text(raw_html)
            log_event(
                logger,
                "stage_done",
                job_id=job.job_id,
                stage="extract_clean_job_text",
                chars=len(job_text),
                duration=format_duration(perf_counter() - stage_started_at),
            )

            stage_started_at = perf_counter()
            job = llm_extract_job_metadata(client, model, job_text, job)
            log_event(
                logger,
                "stage_done",
                job_id=job.job_id,
                stage="llm_extract_job_metadata",
                company=job.company,
                role=job.role,
                location=job.location,
                duration=format_duration(perf_counter() - stage_started_at),
            )

            stage_started_at = perf_counter()
            requirements = llm_extract_job_requirements(client, model, job, job_text)
            log_event(
                logger,
                "stage_done",
                job_id=job.job_id,
                stage="llm_extract_job_requirements",
                requirements_count=len(requirements.get("requirements", []) or []),
                duration=format_duration(perf_counter() - stage_started_at),
            )

            stage_started_at = perf_counter()
            evidence_map = llm_map_resume_evidence(client, model, job, resume_text, cv_understanding, requirements)
            log_event(
                logger,
                "stage_done",
                job_id=job.job_id,
                stage="llm_map_resume_evidence",
                duration=format_duration(perf_counter() - stage_started_at),
            )

            stage_started_at = perf_counter()
            baseline_analysis = llm_ats_score(
                client,
                model,
                job,
                job_text,
                resume_text,
                requirements=requirements,
                evidence_map=evidence_map,
            )
            log_event(
                logger,
                "stage_done",
                job_id=job.job_id,
                stage="baseline_ats_score",
                score=baseline_analysis.get("score", ""),
                recommendation=baseline_analysis.get("screening_recommendation", ""),
                duration=format_duration(perf_counter() - stage_started_at),
            )

            folder_name = build_job_folder_name(job, used_folder_names)
            job_folder = out_dir / folder_name
            output_folder = str(job_folder)
            job_folder.mkdir(parents=True, exist_ok=True)
            log_event(logger, "job_folder_ready", job_id=job.job_id, output_folder=job_folder)

            write_output(job_folder / "job_text.txt", job_text)
            log_event(logger, "artifact_written", job_id=job.job_id, artifact="job_text.txt", path=job_folder / "job_text.txt")
            write_requirements_evidence_csv(job_folder / "requirements_evidence_matrix.csv", requirements, evidence_map)
            log_event(
                logger,
                "artifact_written",
                job_id=job.job_id,
                artifact="requirements_evidence_matrix.csv",
                path=job_folder / "requirements_evidence_matrix.csv",
            )

            stage_started_at = perf_counter()
            (
                tailored_resume,
                optimized_analysis,
                generation_mode,
                evaluation_history,
                accepted_resume_version,
                resume_edit_log,
            ) = optimize_resume_content(
                job,
                job_text,
                resume_text,
                baseline_analysis,
                cv_understanding,
                requirements,
                evidence_map,
                client,
                model,
                target_score,
            )
            log_event(
                logger,
                "stage_done",
                job_id=job.job_id,
                stage="optimize_resume_content",
                generation_mode=generation_mode,
                optimized_score=optimized_analysis.get("score", ""),
                duration=format_duration(perf_counter() - stage_started_at),
            )

            prompts = build_prompt_bundle(
                job,
                job_text,
                resume_text,
                baseline_analysis,
                cv_understanding,
                requirements=requirements,
                evidence_map=evidence_map,
            )
            stage_started_at = perf_counter()
            cover_letter = require_llm_generate(client, model, prompts["cover_letter"], "cover letter", temperature=0.4)
            log_event(
                logger,
                "stage_done",
                job_id=job.job_id,
                stage="generate_cover_letter",
                chars=len(cover_letter),
                duration=format_duration(perf_counter() - stage_started_at),
            )

            gap_diagnosis = ""
            if optimized_analysis["score"] < target_score:
                stage_started_at = perf_counter()
                gap_diagnosis = require_llm_generate(client, model, prompts["gap"], "gap diagnosis", temperature=0.2)
                log_event(
                    logger,
                    "stage_done",
                    job_id=job.job_id,
                    stage="generate_gap_diagnosis",
                    chars=len(gap_diagnosis),
                    duration=format_duration(perf_counter() - stage_started_at),
                )

            meets_target = ats_stop_condition_met(optimized_analysis, target_score)
            gap_summary = build_gap_summary(optimized_analysis, target_score)
            write_analysis_summary_csv(
                job_folder / "analysis_summary.csv",
                job,
                baseline_analysis,
                optimized_analysis,
                target_score,
                meets_target,
                generation_mode,
                gap_summary,
                requirements,
                evidence_map,
                evaluation_history,
                accepted_resume_version,
            )
            write_evaluation_history_csv(job_folder / "evaluation_history.csv", evaluation_history)

            write_output(job_folder / "tailored_resume.md", tailored_resume)
            write_output(job_folder / "resume_edit_log.md", resume_edit_log)
            write_output(job_folder / "cover_letter.md", cover_letter)
            if include_interview_prep:
                stage_started_at = perf_counter()
                interview_pack = require_llm_generate(client, model, prompts["interview"], "interview pack", temperature=0.3)
                write_output(job_folder / "interview_prep.md", interview_pack)
                log_event(
                    logger,
                    "stage_done",
                    job_id=job.job_id,
                    stage="generate_interview_prep",
                    chars=len(interview_pack),
                    duration=format_duration(perf_counter() - stage_started_at),
                )
            if gap_diagnosis:
                gap_filename = f"ats_{optimized_analysis['score']:.1f}_gap_diagnosis.md"
                write_output(job_folder / gap_filename, gap_diagnosis)

            summary_rows.append(
                {
                    "job_id": job.job_id,
                    "company": job.company,
                    "role": job.role,
                    "original_score": baseline_analysis["score"],
                    "optimized_score": optimized_analysis["score"],
                    "meets_target": meets_target,
                    "accepted_resume_version": accepted_resume_version,
                    "gap_type": optimized_analysis.get("gap_type", ""),
                    "gap_summary": gap_summary,
                    "screening_recommendation": optimized_analysis.get("screening_recommendation", ""),
                    "output_folder": output_folder,
                }
            )

            completed_row = {
                "run_date": run_date,
                "job_id": job.job_id,
                "url": job.url,
                "company": job.company,
                "role": job.role,
                "location": job.location,
                "original_score": baseline_analysis["score"],
                "optimized_score": optimized_analysis["score"],
                "target_score": target_score,
                "meets_target": meets_target,
                "accepted_resume_version": accepted_resume_version,
                "gap_type": optimized_analysis.get("gap_type", ""),
                "gap_summary": gap_summary,
                "screening_recommendation": optimized_analysis.get("screening_recommendation", ""),
                "status": "completed",
                "generation_mode": generation_mode,
                "output_folder": output_folder,
                "error": "",
            }
            completed_rows.append(completed_row)
            flush_progress(out_dir, completed_csv_path, completed_all_csv_path, summary_rows, completed_rows, completed_row)
            log_event(
                logger,
                "job_completed",
                index=index,
                total=len(processable_jobs),
                job_id=job.job_id,
                company=job.company,
                role=job.role,
                ats_score=optimized_analysis.get("score", ""),
                screening_recommendation=optimized_analysis.get("screening_recommendation", ""),
                output_folder=output_folder,
                duration=format_duration(perf_counter() - job_started_at),
            )
            print_batch_job_result(
                run_date,
                job,
                optimized_analysis.get("score", ""),
                optimized_analysis.get("screening_recommendation", ""),
                perf_counter() - job_started_at,
                target_score,
                status="completed",
            )
        except Exception as exc:
            completed_row = {
                "run_date": run_date,
                "job_id": job.job_id,
                "url": job.url,
                "company": job.company,
                "role": job.role,
                "location": job.location,
                "original_score": "",
                "optimized_score": "",
                "target_score": target_score,
                "meets_target": False,
                "accepted_resume_version": "",
                "gap_type": "failed",
                "gap_summary": "Failed before ATS gap analysis completed.",
                "screening_recommendation": "",
                "status": "failed",
                "generation_mode": "",
                "output_folder": output_folder,
                "error": str(exc),
            }
            completed_rows.append(completed_row)
            summary_rows.append(
                {
                    "job_id": job.job_id,
                    "company": job.company,
                    "role": job.role,
                    "original_score": "",
                    "optimized_score": "",
                    "meets_target": False,
                    "accepted_resume_version": "",
                    "gap_type": completed_row["gap_type"],
                    "gap_summary": completed_row["gap_summary"],
                    "screening_recommendation": "",
                    "output_folder": output_folder,
                }
            )
            flush_progress(out_dir, completed_csv_path, completed_all_csv_path, summary_rows, completed_rows, completed_row)
            print_batch_job_result(
                run_date,
                job,
                completed_row["optimized_score"],
                completed_row["screening_recommendation"],
                perf_counter() - job_started_at,
                target_score,
                status="failed",
            )
            log_event(
                logger,
                "job_failed",
                index=index,
                total=len(processable_jobs),
                job_id=job.job_id,
                url=job.url,
                error=exc,
                duration=format_duration(perf_counter() - job_started_at),
            )
            if logger is not None:
                logger.exception("job_failed_traceback | job_id=%s | url=%s", job.job_id, job.url)
            print(f"Error: {exc}", flush=True)

    write_output(out_dir / "batch_summary.json", json.dumps(summary_rows, ensure_ascii=False, indent=2))
    write_completed_csv(completed_csv_path, completed_rows)
    write_ranked_csv(out_dir / "ranked_jobs.csv", completed_rows)
    total_duration = perf_counter() - run_started_at
    log_event(logger, "batch_finished", total_jobs=len(processable_jobs), duration=format_duration(total_duration), out_dir=out_dir)
    print_run_footer("Batch", len(processable_jobs), total_duration)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch job application optimizer")
    parser.add_argument("--urls", required=True, help="Path to URL list file (one URL per line)")
    parser.add_argument("--resume", required=True, help="Path to resume file (.pdf/.docx/.txt/.md)")
    parser.add_argument("--out", default="outputs", help="Output directory")
    parser.add_argument(
        "--completed-csv",
        default="completed_job.csv",
        help="Path to completed job report CSV",
    )
    parser.add_argument(
        "--completed-all-csv",
        default="completed_jobs_all.csv",
        help="Path to cumulative completed jobs CSV (append mode)",
    )
    parser.add_argument(
        "--cv-understanding",
        default="cv_deep_understanding.md",
        help="Path to CV deep-understanding cache file",
    )
    parser.add_argument(
        "--interview-prep",
        action="store_true",
        help="Also generate interview_prep.md for each job. Disabled by default to reduce cost and runtime.",
    )
    args = parser.parse_args()

    resume_path = Path(args.resume).resolve()
    run_date = datetime.now().strftime("%Y-%m-%d")
    out_root = Path(args.out).resolve()
    out_dir = out_root / run_date

    completed_csv_base = Path(args.completed_csv).resolve()
    completed_csv_name = f"{completed_csv_base.stem}_{run_date}{completed_csv_base.suffix or '.csv'}"
    completed_csv_path = completed_csv_base.parent / completed_csv_name
    completed_all_csv_path = Path(args.completed_all_csv).resolve()
    cv_understanding_path = Path(args.cv_understanding).resolve()

    urls_file = Path(args.urls).resolve()
    if not urls_file.exists():
        raise FileNotFoundError(f"urls file not found: {urls_file}")
    jobs = parse_urls_file(urls_file)

    if not resume_path.exists():
        raise FileNotFoundError(f"resume file not found: {resume_path}")

    logger, log_path = setup_run_logger("batch")
    print(f"Log file: {log_path}", flush=True)
    log_event(logger, "batch_cli_start", urls_file=urls_file, resume_path=resume_path, out_dir=out_dir)

    run_batch(
        jobs,
        resume_path,
        out_dir,
        completed_csv_path,
        completed_all_csv_path,
        cv_understanding_path,
        run_date,
        include_interview_prep=args.interview_prep,
        logger=logger,
    )
    print(f"Done. Outputs at: {out_dir}")
    print(f"Completed jobs report (dated): {completed_csv_path}")
    print(f"Completed jobs report (all runs): {completed_all_csv_path}")
    print(f"Log file: {log_path}")


__all__ = ["main", "run_batch"]
