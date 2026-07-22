"""Strict schemas for structured LLM responses."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictLLMModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class JobMetadataResult(StrictLLMModel):
    company: str = Field(min_length=1)
    role: str = Field(min_length=1)
    location: str = Field(min_length=1)
    confidence: Literal["high", "medium", "low"]


class CapabilityInventoryEntry(StrictLLMModel):
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    evidence: list[str]
    metrics: list[str]
    tools_used: list[str]
    exact_terms_allowed: list[str]
    equivalent_terms_allowed: list[str]
    exposure_terms: list[str]
    forbidden_terms: list[str]
    safe_sentence_templates: list[str]
    confidence: Literal["high", "medium", "low"]
    verification_status: Literal["needs_review"]
    notes: str


class CapabilityInventoryDraft(StrictLLMModel):
    version: Literal[1]
    status: Literal["draft"]
    instructions: str = Field(min_length=1)
    capabilities: list[CapabilityInventoryEntry] = Field(min_length=1)


class JobRequirement(StrictLLMModel):
    id: str = Field(min_length=1)
    requirement: str = Field(min_length=1)
    category: Literal[
        "technical_skill",
        "domain_context",
        "responsibility",
        "tool",
        "methodology",
        "soft_skill",
        "qualification",
        "delivery_evidence",
    ]
    importance: Literal["must_have", "important", "nice_to_have"]
    ats_keywords: list[str]
    evidence_expected: str = Field(min_length=1)


class JobRequirementsResult(StrictLLMModel):
    requirements: list[JobRequirement] = Field(min_length=1)
    role_summary: str = Field(min_length=1)
    screening_priorities: list[str]


class RequirementEvidence(StrictLLMModel):
    id: str = Field(min_length=1)
    coverage: Literal["strong", "weakly_present", "partial", "adjacent", "missing"]
    resume_evidence: list[str]
    safe_resume_terms: list[str]
    missing_evidence: list[str]
    factual_risk: Literal["none", "low", "medium", "high"]
    positioning_advice: str


class EvidenceMappingResult(StrictLLMModel):
    requirement_map: list[RequirementEvidence]
    strongest_matches: list[str]
    weakest_gaps: list[str]
    safe_positioning_strategy: str


class ATSCategoryScores(StrictLLMModel):
    core_requirements: float = Field(ge=0, le=30)
    technical_fit: float = Field(ge=0, le=25)
    evidence_strength: float = Field(ge=0, le=20)
    domain_fit: float = Field(ge=0, le=10)
    ats_readability: float = Field(ge=0, le=10)
    risk_adjustment: float = Field(ge=0, le=5)


class ATSKeyword(StrictLLMModel):
    keyword: str = Field(min_length=1)
    weight: int = Field(ge=1)


class ATSPassBenchmark(StrictLLMModel):
    requirement_id: str
    requirement: str = Field(min_length=1)
    jd_priority: Literal["must_have", "important", "nice_to_have"]
    coverage: Literal["missing", "adjacent", "partial", "weakly_present", "strong"]
    screening_gate: bool
    current_resume_signal: str
    current_candidate_gap: str
    high_score_benchmark: str = Field(min_length=1)
    why_it_matters: str = Field(min_length=1)
    gap_size: Literal["small", "medium", "large"]
    can_be_improved_by_rewrite: bool
    requires_new_experience: bool
    safe_positioning: str


class ATSWeakSection(StrictLLMModel):
    section: str = Field(min_length=1)
    issue: str = Field(min_length=1)
    priority: Literal["high", "medium", "low"]


class ATSWeakBullet(StrictLLMModel):
    section: str = Field(min_length=1)
    current_bullet_excerpt: str = Field(min_length=1)
    issue: str = Field(min_length=1)
    fix_direction: str = Field(min_length=1)


class ATSFactualRisk(StrictLLMModel):
    claim: str = Field(min_length=1)
    risk: str = Field(min_length=1)
    action: Literal["remove", "soften", "verify"]


class ATSScoreResult(StrictLLMModel):
    score: float = Field(ge=0, le=100)
    category_scores: ATSCategoryScores
    matched_keywords: list[ATSKeyword]
    missing_keywords: list[ATSKeyword]
    resume_safe_improvements: list[str]
    evidence_gaps: list[str]
    ats_pass_benchmarks: list[ATSPassBenchmark]
    weak_sections: list[ATSWeakSection]
    weak_bullets: list[ATSWeakBullet]
    factual_risks: list[ATSFactualRisk]
    editor_instructions: list[str]
    gap_type: Literal[
        "strong_match",
        "wording_gap",
        "skill_gap",
        "domain_gap",
        "seniority_gap",
        "role_mismatch",
    ]
    gap_summary: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    screening_recommendation: Literal[
        "strong_interview",
        "interview_possible",
        "borderline",
        "unlikely",
    ]


__all__ = [
    "ATSScoreResult",
    "CapabilityInventoryDraft",
    "EvidenceMappingResult",
    "JobMetadataResult",
    "JobRequirementsResult",
    "StrictLLMModel",
]
