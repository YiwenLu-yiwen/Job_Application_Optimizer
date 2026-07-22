"""Resume evidence mapping."""

import json
import textwrap
from typing import Any

from job_application_optimizer.config import _prompt_text
from job_application_optimizer.llm.client import LLMRouter, ModelRole
from job_application_optimizer.llm.schemas import EvidenceMappingResult
from job_application_optimizer.llm.structured import generate_structured
from job_application_optimizer.models import JobRecord

VALID_COVERAGE_VALUES = {"strong", "weakly_present", "partial", "adjacent", "missing"}


def build_evidence_mapping_prompt(
    job: JobRecord,
    resume_text: str,
    cv_understanding: str,
    requirements: dict[str, Any],
    capability_inventory: str = "",
) -> str:
    resume_excerpt = _prompt_text(resume_text)
    cv_understanding_excerpt = _prompt_text(cv_understanding)
    capability_inventory_excerpt = _prompt_text(capability_inventory) if capability_inventory else "No verified capability inventory provided."
    requirements_json = json.dumps(requirements, ensure_ascii=False, indent=2)
    return textwrap.dedent(
        f"""
        Map the candidate's resume evidence to each extracted job requirement.
        Do not write or rewrite resume content. Do not invent evidence.
        For every requirement, decide whether the resume evidence is strong, weakly_present, partial, adjacent, or missing.
        If a verified capability inventory is provided, use only entries with verification_status: verified.
        Ignore needs_review and rejected inventory entries as resume evidence.

        Target role: {job.role}
        Company: {job.company}

        Extracted requirements:
        {requirements_json}

        Original resume:
        {resume_excerpt}

        CV deep understanding:
        {cv_understanding_excerpt}

        Verified capability inventory:
        {capability_inventory_excerpt}

        Return JSON only:
        {{
          "requirement_map": [
            {{
              "id": "R1",
              "coverage": "strong | weakly_present | partial | adjacent | missing",
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
    llm: LLMRouter,
    job: JobRecord,
    resume_text: str,
    cv_understanding: str,
    requirements: dict[str, Any],
    capability_inventory: str = "",
) -> dict[str, Any]:
    prompt = build_evidence_mapping_prompt(
        job,
        resume_text,
        cv_understanding,
        requirements,
        capability_inventory=capability_inventory,
    )
    result = generate_structured(
        llm,
        ModelRole.REASONING,
        prompt,
        "resume evidence map",
        EvidenceMappingResult,
        temperature=0.1,
    )
    return result.model_dump()


__all__ = ["VALID_COVERAGE_VALUES", "build_evidence_mapping_prompt", "llm_map_resume_evidence"]
