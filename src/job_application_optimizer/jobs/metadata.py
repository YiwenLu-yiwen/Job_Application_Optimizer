"""Job metadata inference and extraction."""

import textwrap

from bs4 import BeautifulSoup
from openai import OpenAI

from job_application_optimizer.config import _prompt_text
from job_application_optimizer.llm.client import require_llm_generate
from job_application_optimizer.llm.json_parser import parse_llm_json
from job_application_optimizer.models import JobRecord


def infer_job_meta_from_html(html: str, fallback_role: str = "", fallback_company: str = "") -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title else ""
    if not title:
        return fallback_role or "Unknown Role", fallback_company or "Unknown Company"

    separators = [" - ", " | ", " @ ", " at "]
    for sep in separators:
        if sep in title:
            left, right = title.split(sep, 1)
            role = (fallback_role or left).strip() or "Unknown Role"
            company = (fallback_company or right).strip() or "Unknown Company"
            return role, company

    return fallback_role or title, fallback_company or "Unknown Company"


def build_job_metadata_prompt(job_text: str, url: str, fallback: JobRecord) -> str:
    job_excerpt = _prompt_text(job_text)
    return textwrap.dedent(
        f"""
        Extract canonical job metadata from this job posting.
        Prefer the posting content over the URL or page title. Use the fallback values only when the posting does not contain the field.

        URL:
        {url}

        Fallback metadata:
        company: {fallback.company}
        role: {fallback.role}
        location: {fallback.location}

        Job posting text:
        {job_excerpt}

        Return JSON only:
        {{
          "company": "canonical employer name",
          "role": "canonical job title",
          "location": "canonical location or work arrangement",
          "confidence": "high | medium | low"
        }}
        """
    ).strip()


def llm_extract_job_metadata(client: OpenAI, model: str, job_text: str, job: JobRecord) -> JobRecord:
    prompt = build_job_metadata_prompt(job_text, job.url, job)
    try:
        payload = parse_llm_json(require_llm_generate(client, model, prompt, "job metadata", temperature=0.0))
    except Exception:
        return job

    company = str(payload.get("company") or job.company or "").strip()
    role = str(payload.get("role") or job.role or "").strip()
    location = str(payload.get("location") or job.location or "").strip()

    return JobRecord(
        job_id=job.job_id,
        url=job.url,
        company=company or "Unknown Company",
        role=role or "Unknown Role",
        location=location or "Unknown",
    )


__all__ = ["build_job_metadata_prompt", "infer_job_meta_from_html", "llm_extract_job_metadata"]
