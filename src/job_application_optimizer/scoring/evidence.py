"""Resume evidence mapping."""

import json
import textwrap
from typing import Any

from openai import OpenAI

from job_application_optimizer.config import _prompt_text
from job_application_optimizer.llm.client import require_llm_generate
from job_application_optimizer.llm.json_parser import parse_llm_json
from job_application_optimizer.models import JobRecord


def build_evidence_mapping_prompt(
    job: JobRecord,
    resume_text: str,
    cv_understanding: str,
    requirements: dict[str, Any],
) -> str:
    resume_excerpt = _prompt_text(resume_text)
    cv_understanding_excerpt = _prompt_text(cv_understanding)
    requirements_json = json.dumps(requirements, ensure_ascii=False, indent=2)
    return textwrap.dedent(
        f"""
        Map the candidate's resume evidence to each extracted job requirement.
        Do not write or rewrite resume content. Do not invent evidence.
        For every requirement, decide whether the resume evidence is strong, partial, adjacent, or missing.

        Target role: {job.role}
        Company: {job.company}

        Extracted requirements:
        {requirements_json}

        Original resume:
        {resume_excerpt}

        CV deep understanding:
        {cv_understanding_excerpt}

        Return JSON only:
        {{
          "requirement_map": [
            {{
              "id": "R1",
              "coverage": "strong | partial | adjacent | missing",
              "resume_evidence": ["specific resume-backed evidence or metric"],
              "safe_resume_terms": ["JD-aligned terms that can be used truthfully"],
              "missing_evidence": ["what is not supported and should not be claimed"],
              "factual_risk": "none | low | medium | high",
              "positioning_advice": "how to position this requirement truthfully"
            }}
          ],
          "strongest_matches": ["requirements with strongest evidence"],
          "weakest_gaps": ["requirements with weak or missing evidence"],
          "safe_positioning_strategy": "brief strategy for the baseline resume"
        }}
        """
    ).strip()


def llm_map_resume_evidence(
    client: OpenAI,
    model: str,
    job: JobRecord,
    resume_text: str,
    cv_understanding: str,
    requirements: dict[str, Any],
) -> dict[str, Any]:
    prompt = build_evidence_mapping_prompt(job, resume_text, cv_understanding, requirements)
    payload = parse_llm_json(require_llm_generate(client, model, prompt, "resume evidence map", temperature=0.1))
    requirement_map = payload.get("requirement_map")
    if not isinstance(requirement_map, list):
        payload["requirement_map"] = []
    return payload


__all__ = ["build_evidence_mapping_prompt", "llm_map_resume_evidence"]
