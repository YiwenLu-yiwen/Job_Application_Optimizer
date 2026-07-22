import json

import pytest

from job_application_optimizer.jobs.metadata import llm_extract_job_metadata
from job_application_optimizer.jobs.requirements import llm_extract_job_requirements
from job_application_optimizer.llm.client import ModelRole
from job_application_optimizer.llm.schemas import JobMetadataResult
from job_application_optimizer.llm.structured import StructuredOutputError, generate_structured
from job_application_optimizer.models import JobRecord
from job_application_optimizer.resume.capability_inventory import generate_capability_inventory_draft
from job_application_optimizer.scoring.evidence import llm_map_resume_evidence
from job_application_optimizer.scoring.ats import llm_ats_score


class SequenceLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def generate(
        self,
        role: object,
        prompt: str,
        artifact_name: str,
        temperature: float = 0.2,
    ) -> str:
        self.prompts.append(prompt)
        return next(self.responses)


def test_job_metadata_retries_with_validation_feedback() -> None:
    llm = SequenceLLM(
        [
            json.dumps(
                {
                    "company": "Acme",
                    "role": "Engineer",
                    "location": "Remote",
                    "confidence": "certain",
                }
            ),
            json.dumps(
                {
                    "company": "Acme",
                    "role": "Engineer",
                    "location": "Remote",
                    "confidence": "high",
                }
            ),
        ]
    )
    fallback = JobRecord("job_001", "https://example.com", "", "", "")

    result = llm_extract_job_metadata(llm, "job posting", fallback)

    assert result.company == "Acme"
    assert result.role == "Engineer"
    assert len(llm.prompts) == 2
    assert "confidence" in llm.prompts[1]
    assert "validation" in llm.prompts[1].lower()


def test_job_requirements_retry_when_a_required_field_is_missing() -> None:
    incomplete_requirement = {
        "id": "R1",
        "requirement": "Build Python services",
        "category": "technical_skill",
        "importance": "must_have",
        "ats_keywords": ["Python"],
    }
    complete_requirement = {
        **incomplete_requirement,
        "evidence_expected": "Production Python delivery",
    }
    llm = SequenceLLM(
        [
            json.dumps(
                {
                    "requirements": [incomplete_requirement],
                    "role_summary": "Python delivery role",
                    "screening_priorities": ["Python"],
                }
            ),
            json.dumps(
                {
                    "requirements": [complete_requirement],
                    "role_summary": "Python delivery role",
                    "screening_priorities": ["Python"],
                }
            ),
        ]
    )
    job = JobRecord("job_001", "https://example.com", "Acme", "Engineer", "Remote")

    result = llm_extract_job_requirements(llm, job, "job posting")

    assert result["requirements"][0]["evidence_expected"] == "Production Python delivery"
    assert len(llm.prompts) == 2


def test_evidence_mapping_retries_invalid_coverage_values() -> None:
    def response(coverage: str) -> str:
        return json.dumps(
            {
                "requirement_map": [
                    {
                        "id": "R1",
                        "coverage": coverage,
                        "resume_evidence": ["Built Python services"],
                        "safe_resume_terms": ["Python"],
                        "missing_evidence": [],
                        "factual_risk": "none",
                        "positioning_advice": "Lead with production delivery",
                    }
                ],
                "strongest_matches": ["Python"],
                "weakest_gaps": [],
                "safe_positioning_strategy": "Use direct evidence",
            }
        )

    llm = SequenceLLM([response("complete"), response("strong")])
    job = JobRecord("job_001", "https://example.com", "Acme", "Engineer", "Remote")

    result = llm_map_resume_evidence(
        llm,
        job,
        "resume",
        "cv understanding",
        {"requirements": []},
    )

    assert result["requirement_map"][0]["coverage"] == "strong"
    assert len(llm.prompts) == 2


def test_ats_scoring_retries_non_numeric_scores() -> None:
    def response(score: object) -> str:
        return json.dumps(
            {
                "score": score,
                "category_scores": {
                    "core_requirements": 24,
                    "technical_fit": 20,
                    "evidence_strength": 16,
                    "domain_fit": 8,
                    "ats_readability": 9,
                    "risk_adjustment": 5,
                },
                "matched_keywords": [{"keyword": "Python", "weight": 3}],
                "missing_keywords": [],
                "resume_safe_improvements": [],
                "evidence_gaps": [],
                "ats_pass_benchmarks": [],
                "weak_sections": [],
                "weak_bullets": [],
                "factual_risks": [],
                "editor_instructions": [],
                "gap_type": "strong_match",
                "gap_summary": "Strong evidence alignment",
                "rationale": "The resume demonstrates the core requirements.",
                "screening_recommendation": "strong_interview",
            }
        )

    llm = SequenceLLM([response("82"), response(82)])
    job = JobRecord("job_001", "https://example.com", "Acme", "Engineer", "Remote")

    result = llm_ats_score(llm, job, "job posting", "resume")

    assert result["score"] == 82.0
    assert len(llm.prompts) == 2


def test_structured_generation_fails_after_corrective_retries_are_exhausted() -> None:
    llm = SequenceLLM(["not json", "still not json"])

    with pytest.raises(StructuredOutputError, match="after 2 attempts"):
        generate_structured(
            llm,
            ModelRole.METADATA,
            "Extract metadata",
            "job metadata",
            JobMetadataResult,
            max_attempts=2,
        )

    assert len(llm.prompts) == 2
    assert "Previous invalid response" in llm.prompts[1]


def test_capability_inventory_retries_entries_that_are_not_reviewable_drafts() -> None:
    def response(verification_status: str) -> str:
        return json.dumps(
            {
                "version": 1,
                "status": "draft",
                "instructions": "Review manually before use.",
                "capabilities": [
                    {
                        "name": "Python delivery",
                        "category": "technical_delivery",
                        "evidence": ["Built Python services"],
                        "metrics": [],
                        "tools_used": ["Python"],
                        "exact_terms_allowed": ["Python"],
                        "equivalent_terms_allowed": [],
                        "exposure_terms": [],
                        "forbidden_terms": [],
                        "safe_sentence_templates": [],
                        "confidence": "high",
                        "verification_status": verification_status,
                        "notes": "",
                    }
                ],
            }
        )

    llm = SequenceLLM([response("verified"), response("needs_review")])

    draft = generate_capability_inventory_draft("resume", "cv understanding", llm)

    assert json.loads(draft)["capabilities"][0]["verification_status"] == "needs_review"
    assert len(llm.prompts) == 2
