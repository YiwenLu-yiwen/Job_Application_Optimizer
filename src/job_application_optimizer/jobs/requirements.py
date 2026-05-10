"""Job requirement extraction."""

import textwrap
from typing import Any

from openai import OpenAI

from job_application_optimizer.config import _prompt_text
from job_application_optimizer.llm.client import require_llm_generate
from job_application_optimizer.llm.json_parser import parse_llm_json
from job_application_optimizer.models import JobRecord


def build_requirement_extraction_prompt(job: JobRecord, job_text: str) -> str:
    job_excerpt = _prompt_text(job_text)
    return textwrap.dedent(
        f"""
        Extract the real job requirements from this JD for resume tailoring and ATS/recruiter screening.
        Be role-agnostic and company-agnostic. Do not hardcode assumptions. Ignore employer branding, benefits, legal boilerplate, and repeated company slogans unless they describe actual selection criteria.

        Target role: {job.role}
        Company: {job.company}

        Job description:
        {job_excerpt}

        Return JSON only:
        {{
          "requirements": [
            {{
              "id": "R1",
              "requirement": "specific requirement",
              "category": "technical_skill | domain_context | responsibility | tool | methodology | soft_skill | qualification | delivery_evidence",
              "importance": "must_have | important | nice_to_have",
              "ats_keywords": ["JD terms or close variants that should appear only if truthfully supported"],
              "evidence_expected": "what proof a recruiter would expect to see"
            }}
          ],
          "role_summary": "1-2 sentence summary of what this role is really hiring for",
          "screening_priorities": ["highest-signal requirements in order"]
        }}
        """
    ).strip()


def llm_extract_job_requirements(client: OpenAI, model: str, job: JobRecord, job_text: str) -> dict[str, Any]:
    prompt = build_requirement_extraction_prompt(job, job_text)
    payload = parse_llm_json(require_llm_generate(client, model, prompt, "job requirements", temperature=0.1))
    requirements = payload.get("requirements")
    if not isinstance(requirements, list):
        payload["requirements"] = []
    return payload


__all__ = ["build_requirement_extraction_prompt", "llm_extract_job_requirements"]
