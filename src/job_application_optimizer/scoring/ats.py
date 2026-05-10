"""LLM ATS scoring."""

import json
import textwrap
from typing import Any

from openai import OpenAI

from job_application_optimizer.config import _prompt_text
from job_application_optimizer.llm.client import require_llm_generate
from job_application_optimizer.llm.json_parser import normalize_llm_ats_result, parse_llm_json
from job_application_optimizer.models import JobRecord


def build_llm_ats_score_prompt(
    job: JobRecord,
    job_text: str,
    resume_text: str,
    requirements: dict[str, Any] | None = None,
    evidence_map: dict[str, Any] | None = None,
) -> str:
    job_excerpt = _prompt_text(job_text)
    resume_excerpt = _prompt_text(resume_text)
    requirements_json = json.dumps(requirements or {}, ensure_ascii=False, indent=2)
    evidence_map_json = json.dumps(evidence_map or {}, ensure_ascii=False, indent=2)
    return textwrap.dedent(
        f"""
        You are the Evaluator. Simulate an ATS + technical recruiter screen for this resume against this job description.
        You must score, diagnose, and identify weak sections only. You must NOT rewrite the resume.
        This must be a general ATS simulation rubric that works across roles and companies, not a one-off keyword counter.

        Target role: {job.role}
        Company: {job.company}

        Job description:
        {job_excerpt}

        Resume:
        {resume_excerpt}

        Extracted job requirements:
        {requirements_json}

        Resume evidence mapped to requirements:
        {evidence_map_json}

        Scoring rubric:
        - 0-100 overall score.
        - Evaluate semantic fit, not only exact keyword overlap.
        - Ignore employer branding, benefits, DEI language, legal boilerplate, company slogans, and repeated company names unless they describe real role requirements.
        - Reward exact JD terminology only when it is supported by evidence in the resume.
        - Penalize unsupported claims, missing must-have tools, missing domain requirements, and lack of production evidence.
        - Distinguish missing wording from missing evidence.
        - Do not require every bonus skill if core evidence is strong.
        - Identify weak sections/bullets that an Editor can locally revise.
        - Identify factual risks separately from wording gaps. Factual risk means a claim in the resume may not be supported by the original resume facts or evidence map.
        - A realistic ATS/recruiter threshold is:
          85-100 excellent match, 70-84 plausible interview, 55-69 partial/adjacent match, 40-54 weak match, below 40 unlikely.

        Category weights:
        1) Core role requirements and responsibilities: 30
        2) Technical skills, tools, and architecture fit: 25
        3) Evidence strength: production scope, metrics, ownership, seniority: 20
        4) Domain/business context fit: 10
        5) ATS readability and keyword coverage: 10
        6) Risk factors: overclaiming, missing must-haves, unclear evidence: 5

        Return JSON only with this schema:
        {{
          "score": 0,
          "category_scores": {{
            "core_requirements": 0,
            "technical_fit": 0,
            "evidence_strength": 0,
            "domain_fit": 0,
            "ats_readability": 0,
            "risk_adjustment": 0
          }},
          "matched_keywords": [
            {{"keyword": "keyword or capability", "weight": 1}}
          ],
          "missing_keywords": [
            {{"keyword": "keyword or capability", "weight": 1}}
          ],
          "resume_safe_improvements": [
            "JD-aligned wording that can be added truthfully based on the resume"
          ],
          "evidence_gaps": [
            "real missing experience/tool/domain evidence that should not be claimed"
          ],
          "ats_pass_benchmarks": [
            {{
              "requirement_id": "R1",
              "requirement": "JD requirement or capability area",
              "jd_priority": "must_have | important | nice_to_have",
              "coverage": "missing | adjacent | partial | weakly_present | strong",
              "screening_gate": true,
              "current_resume_signal": "what the current resume already proves for this requirement",
              "current_candidate_gap": "how the current resume falls short, or 'already covered' if strong",
              "high_score_benchmark": "concrete resume evidence a high-scoring candidate would likely show",
              "why_it_matters": "why this evidence would improve ATS/recruiter screen pass likelihood for this JD",
              "gap_size": "small | medium | large",
              "can_be_improved_by_rewrite": true,
              "requires_new_experience": false,
              "safe_positioning": "truthful positioning language if rewrite can help; otherwise empty"
            }}
          ],
          "weak_sections": [
            {{"section": "Professional Summary | Core Skills | Experience: Company Name | Education | Selected Research", "issue": "what is weak", "priority": "high | medium | low"}}
          ],
          "weak_bullets": [
            {{"section": "Experience: Company Name", "current_bullet_excerpt": "short excerpt", "issue": "why weak", "fix_direction": "specific local edit direction"}}
          ],
          "factual_risks": [
            {{"claim": "unsupported or risky claim", "risk": "why it may be unsupported", "action": "remove | soften | verify"}}
          ],
          "editor_instructions": [
            "specific local edit instruction for the Editor; no full CV restructuring"
          ],
          "gap_type": "strong_match | wording_gap | skill_gap | domain_gap | seniority_gap | role_mismatch",
          "gap_summary": "one concise sentence explaining whether the gap is mainly wording, missing skills/tools, domain mismatch, seniority mismatch, or overall role mismatch",
          "rationale": "2-4 sentences explaining the score",
          "screening_recommendation": "strong_interview | interview_possible | borderline | unlikely"
        }}

        For ats_pass_benchmarks:
        - Provide 3-5 high-signal examples only for the most important unresolved or weakly evidenced requirements.
        - Anchor each item to extracted job requirements and resume evidence map where possible: use requirement_id, jd_priority, and coverage.
        - Prefer must_have and screening-priority requirements over nice-to-have items.
        - Set screening_gate to true only when the requirement is likely to materially affect ATS/recruiter screen selection.
        - Use gap_size small when the evidence exists but wording is weak, medium when evidence is adjacent/partial, and large when real evidence is missing.
        - Set can_be_improved_by_rewrite to true only when the original resume already contains enough evidence to improve the signal.
        - Set requires_new_experience to true when the gap should not be claimed without real additional experience.
        - These are benchmark examples of what an ATS-passing/high-screening resume would show, not rewrite instructions.
        - Do not imply the current candidate has benchmark evidence unless it is explicitly present in the resume/evidence map.
        """
    ).strip()


def llm_ats_score(
    client: OpenAI,
    model: str,
    job: JobRecord,
    job_text: str,
    resume_text: str,
    requirements: dict[str, Any] | None = None,
    evidence_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt = build_llm_ats_score_prompt(job, job_text, resume_text, requirements=requirements, evidence_map=evidence_map)
    content = require_llm_generate(client, model, prompt, "LLM ATS score", temperature=0.1)
    payload = parse_llm_json(content)
    return normalize_llm_ats_result(payload)


def has_factual_risk(analysis: dict[str, Any]) -> bool:
    risks = analysis.get("factual_risks", [])
    return isinstance(risks, list) and any(str(item).strip() for item in risks)


def ats_stop_condition_met(analysis: dict[str, Any], target_score: float) -> bool:
    return float(analysis.get("score", 0) or 0) >= target_score


__all__ = ["ats_stop_condition_met", "build_llm_ats_score_prompt", "has_factual_risk", "llm_ats_score"]
