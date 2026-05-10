"""Resume optimization loop."""

import difflib
import json
import os
import textwrap
from typing import Any

from openai import OpenAI

from job_application_optimizer.config import _prompt_text
from job_application_optimizer.generation.prompt_bundle import build_prompt_bundle
from job_application_optimizer.llm.client import require_llm_generate
from job_application_optimizer.models import JobRecord
from job_application_optimizer.scoring.ats import ats_stop_condition_met, llm_ats_score


def _analysis_score(analysis: dict[str, Any]) -> float:
    return float(analysis.get("score", 0) or 0)


def _factual_risk_count(analysis: dict[str, Any]) -> int:
    risks = analysis.get("factual_risks", [])
    if not isinstance(risks, list):
        return 0
    return len([item for item in risks if str(item).strip()])


def _rewrite_lift_threshold() -> float:
    try:
        return float(os.getenv("REWRITE_LIFT_THRESHOLD", "3"))
    except (TypeError, ValueError):
        return 3.0


def _acceptable_factual_risk_count() -> int:
    try:
        return max(int(os.getenv("ACCEPTABLE_FACTUAL_RISK_COUNT", "0")), 0)
    except (TypeError, ValueError):
        return 0


def _version_label(index: int) -> str:
    return f"version{index}"


def _short_list(items: Any, limit: int = 4) -> list[str]:
    if not isinstance(items, list):
        return []
    values = []
    for item in items[:limit]:
        if isinstance(item, dict):
            values.append("; ".join(f"{key}: {value}" for key, value in item.items() if value))
        else:
            values.append(str(item))
    return [value for value in values if value]


def _resume_diff(before: str, after: str, before_label: str, after_label: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=before_label,
        tofile=after_label,
        lineterm="",
    )
    return "\n".join(diff)


def _append_analysis_notes(lines: list[str], analysis: dict[str, Any], limit: int = 3) -> None:
    factual_risks = _short_list(analysis.get("factual_risks", []), limit=limit)
    weak_sections = _short_list(analysis.get("weak_sections", []), limit=limit)
    weak_bullets = _short_list(analysis.get("weak_bullets", []), limit=limit)
    safe_improvements = _short_list(analysis.get("resume_safe_improvements", []), limit=limit)
    evidence_gaps = _short_list(analysis.get("evidence_gaps", []), limit=limit)
    ats_pass_benchmarks = _short_list(analysis.get("ats_pass_benchmarks", []), limit=limit)
    if factual_risks:
        lines.append("- Factual risks:")
        lines.extend(f"  - {item}" for item in factual_risks)
    if weak_sections:
        lines.append("- Weak sections:")
        lines.extend(f"  - {item}" for item in weak_sections)
    if weak_bullets:
        lines.append("- Weak bullets:")
        lines.extend(f"  - {item}" for item in weak_bullets)
    if safe_improvements:
        lines.append("- Resume-safe improvements:")
        lines.extend(f"  - {item}" for item in safe_improvements)
    if evidence_gaps:
        lines.append("- Evidence gaps:")
        lines.extend(f"  - {item}" for item in evidence_gaps)
    if ats_pass_benchmarks:
        lines.append("- ATS-pass benchmarks:")
        lines.extend(f"  - {item}" for item in ats_pass_benchmarks)


def _coverage_items(evidence_map: dict[str, Any], coverage_values: set[str]) -> list[dict[str, Any]]:
    items = evidence_map.get("requirement_map", [])
    if not isinstance(items, list):
        return []
    matched = []
    for item in items:
        if not isinstance(item, dict):
            continue
        coverage = str(item.get("coverage") or "").strip().lower()
        if coverage in coverage_values:
            matched.append(item)
    return matched


def _gap_summary_items(items: list[dict[str, Any]], limit: int = 6) -> list[str]:
    rows = []
    for item in items[:limit]:
        req_id = str(item.get("id") or "").strip()
        advice = str(item.get("positioning_advice") or "").strip()
        terms = item.get("safe_resume_terms", [])
        terms_text = ", ".join(str(term) for term in terms[:4]) if isinstance(terms, list) else ""
        parts = [part for part in (req_id, terms_text, advice) if part]
        if parts:
            rows.append(" | ".join(parts))
    return rows


def estimate_rewrite_opportunity(
    baseline_analysis: dict[str, Any],
    evidence_map: dict[str, Any],
    target_score: float,
    lift_threshold: float = 3.0,
) -> dict[str, Any]:
    original_score = _analysis_score(baseline_analysis)
    supported = _coverage_items(evidence_map, {"strong"})
    partial = _coverage_items(evidence_map, {"partial"})
    adjacent = _coverage_items(evidence_map, {"adjacent"})
    missing = _coverage_items(evidence_map, {"missing"})

    wording_gap = str(baseline_analysis.get("gap_type") or "").strip() == "wording_gap"
    safe_improvements = baseline_analysis.get("resume_safe_improvements", [])
    weak_bullets = baseline_analysis.get("weak_bullets", [])
    factual_risks = _factual_risk_count(baseline_analysis)

    estimated_lift = 0.0
    if wording_gap:
        estimated_lift += 2.0
    if isinstance(safe_improvements, list):
        estimated_lift += min(len(safe_improvements), 4) * 0.75
    if isinstance(weak_bullets, list):
        estimated_lift += min(len(weak_bullets), 4) * 0.5
    estimated_lift += min(len(partial), 5) * 0.8
    estimated_lift += min(len(adjacent), 3) * 0.25
    estimated_lift -= min(len(missing), 5) * 0.4
    estimated_lift -= factual_risks * 0.5
    estimated_lift = round(max(0.0, estimated_lift), 2)

    should_rewrite = estimated_lift >= lift_threshold
    if original_score >= target_score and factual_risks == 0 and estimated_lift < lift_threshold + 2:
        should_rewrite = False

    if should_rewrite:
        reason = "estimated_lift_meets_threshold"
    elif original_score >= target_score and factual_risks == 0:
        reason = "original_already_meets_target_without_risk"
    else:
        reason = "estimated_lift_below_threshold"

    return {
        "should_rewrite": should_rewrite,
        "estimated_lift": estimated_lift,
        "lift_threshold": lift_threshold,
        "reason": reason,
        "gap_classification": {
            "supported_by_resume_evidence": _gap_summary_items(supported),
            "partially_supported_can_reframe": _gap_summary_items(partial),
            "adjacent_only_soft_positioning": _gap_summary_items(adjacent),
            "not_supported_cannot_add": _gap_summary_items(missing),
        },
    }


def build_resume_edit_log(
    job: JobRecord,
    target_score: float,
    original_score: float,
    generation_mode: str,
    accepted_resume_version: str,
    evaluation_history: list[dict[str, Any]],
    edit_attempts: list[dict[str, Any]],
    original_analysis: dict[str, Any] | None = None,
    rewrite_decision: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# Resume Edit Log",
        "",
        f"- Company: {job.company}",
        f"- Role: {job.role}",
        f"- Accepted resume version: {accepted_resume_version}",
        f"- Generation mode: {generation_mode}",
        f"- Original resume score: {original_score:g}",
        f"- Target score: {target_score:g}",
        "",
        "## Original Resume Baseline",
        "",
        f"- Score: {original_score:g}",
        f"- Stop condition met: {ats_stop_condition_met(original_analysis or {}, target_score)}",
        f"- Stop reason: {(original_analysis or {}).get('stop_reason', 'none') or 'none'}",
    ]
    if original_analysis:
        _append_analysis_notes(lines, original_analysis, limit=4)
    lines.extend(
        [
            "",
            "## Rewrite Decision",
            "",
        ]
    )
    if rewrite_decision:
        lines.extend(
            [
                f"- Should rewrite: {rewrite_decision.get('should_rewrite')}",
                f"- Estimated lift: {rewrite_decision.get('estimated_lift')}",
                f"- Lift threshold: {rewrite_decision.get('lift_threshold')}",
                f"- Reason: {rewrite_decision.get('reason')}",
            ]
        )
        gap_classification = rewrite_decision.get("gap_classification", {})
        if isinstance(gap_classification, dict):
            labels = [
                ("Supported by resume evidence", "supported_by_resume_evidence"),
                ("Partially supported; can soften/reframe", "partially_supported_can_reframe"),
                ("Adjacent only; soft positioning", "adjacent_only_soft_positioning"),
                ("Not supported; cannot add", "not_supported_cannot_add"),
            ]
            for label, key in labels:
                values = gap_classification.get(key, [])
                if isinstance(values, list) and values:
                    lines.append(f"- {label}:")
                    lines.extend(f"  - {item}" for item in values[:6])
    else:
        lines.append("- No rewrite decision was recorded.")
    lines.extend(
        [
            "",
            "## Evaluation Versions",
            "",
        ]
    )

    for index, item in enumerate(evaluation_history, start=1):
        analysis = item.get("analysis", {}) if isinstance(item, dict) else {}
        version = _version_label(index)
        score = _analysis_score(analysis) if isinstance(analysis, dict) else 0
        stop_reason = analysis.get("stop_reason", "") if isinstance(analysis, dict) else ""
        mode = item.get("mode", "") if isinstance(item, dict) else ""
        accepted_marker = "accepted final" if version == accepted_resume_version else "not final"
        if generation_mode == "original-kept":
            accepted_marker = "rejected; original resume kept"
        lines.extend(
            [
                f"### {version}",
                "",
                f"- Mode: {mode}",
                f"- Score: {score:g}",
                f"- Status: {accepted_marker}",
                f"- Stop reason: {stop_reason or 'none'}",
            ]
        )
        if isinstance(analysis, dict):
            _append_analysis_notes(lines, analysis, limit=3)
        lines.append("")

    lines.extend(["## Editor Attempts", ""])
    if not edit_attempts:
        if generation_mode == "original-kept" and not evaluation_history:
            lines.append("No tailored rewrite was generated because the estimated evidence-backed lift was below the rewrite threshold.")
        elif generation_mode == "original-kept":
            lines.append("No editor attempts were run because the first generated tailored resume did not improve on the original resume.")
        else:
            lines.append("No editor attempts were run; the first generated tailored resume was accepted.")
        return "\n".join(lines).strip() + "\n"

    for attempt in edit_attempts:
        decision = "accepted" if attempt.get("accepted") else "rejected"
        delta = attempt.get("score_delta", "")
        lines.extend(
            [
                f"### Attempt {attempt.get('attempt')}: {attempt.get('from_version')} -> {attempt.get('to_version')}",
                "",
                f"- Previous score: {attempt.get('previous_score')}",
                f"- Candidate score: {attempt.get('candidate_score')}",
                f"- Score delta: {delta}",
                f"- Decision: {decision}",
                f"- Reason: {attempt.get('reason', '')}",
                "",
            ]
        )
        diff = str(attempt.get("diff", "")).strip()
        if diff:
            lines.extend(["```diff", diff, "```", ""])

    return "\n".join(lines).strip() + "\n"


def build_section_editor_prompt(
    job: JobRecord,
    job_text: str,
    resume_text: str,
    current_resume: str,
    optimized_analysis: dict[str, Any],
    requirements: dict[str, Any],
    evidence_map: dict[str, Any],
    attempt_number: int,
) -> str:
    missing = ", ".join(k for k, _ in optimized_analysis["missing_keywords"][:20]) or "none"
    safe_improvements = "\n".join(f"- {item}" for item in optimized_analysis.get("resume_safe_improvements", [])[:10]) or "- None provided"
    evidence_gaps = "\n".join(f"- {item}" for item in optimized_analysis.get("evidence_gaps", [])[:10]) or "- None provided"
    weak_sections = json.dumps(optimized_analysis.get("weak_sections", []), ensure_ascii=False, indent=2)
    weak_bullets = json.dumps(optimized_analysis.get("weak_bullets", []), ensure_ascii=False, indent=2)
    factual_risks = json.dumps(optimized_analysis.get("factual_risks", []), ensure_ascii=False, indent=2)
    editor_instructions = "\n".join(f"- {item}" for item in optimized_analysis.get("editor_instructions", [])[:12]) or "- None provided"
    requirements_json = json.dumps(requirements, ensure_ascii=False, indent=2)
    evidence_map_json = json.dumps(evidence_map, ensure_ascii=False, indent=2)
    resume_excerpt = _prompt_text(resume_text)
    job_excerpt = _prompt_text(job_text)
    current_resume_excerpt = _prompt_text(current_resume)
    return textwrap.dedent(
        f"""
        You are the Editor. Apply a targeted local edit to the current tailored resume for {job.role} at {job.company}.
        This is local edit attempt {attempt_number}. The Evaluator scored the current resume {optimized_analysis['score']}.

        Critical role separation:
        - The Evaluator found the issues below.
        - You are not allowed to re-evaluate, invent new gaps, or rebuild the resume from scratch.
        - You must revise only the weak sections or weak bullets identified by the Evaluator.
        - Preserve all other sections and bullets as close to verbatim as possible.
        - Output the full resume after local edits so it can be re-scored.

        Constraints:
        1) Preserve truthfulness to the original resume facts.
        2) Revise only weak sections/bullets listed below. Do not restructure the whole CV.
        3) If factual risks are listed, remove, soften, or verify those exact claims using only source evidence.
        4) Use resume-safe improvements and missing keyword guidance only where supported by original facts and evidence map.
        5) Translate demonstrated experience into equivalent JD terminology, but do not introduce a skill, tool, framework, security practice, or architecture pattern unless the original resume explicitly supports it.
           Keyword replacement must feel organic inside the bullet. Do not append awkward keyword lists or repeat the same JD term unnaturally.
        6) If a JD keyword is adjacent but not proven, omit it from the resume.
        7) Keep the existing resume format and section order unless the Evaluator specifically flags a section-order issue.
        8) Keep output ATS-friendly: simple section headings, plain text bullets, no tables, no graphics, no meta commentary.
        9) Preserve the experience bullet counts:
            - Most recent experience: exactly 5 bullets.
            - Second most recent experience: exactly 4 bullets.
            - Every remaining experience (if any): exactly 3 bullets each.
        10) If editing the Professional Summary, keep it concise: 45-70 words, 2-3 natural sentences.
            It should sound like a human-written senior technical summary, not a keyword list.
            Build it intelligently:
            - Sentence 1: identity positioning + strongest role-relevant experience.
            - Sentence 2: core technical skill stack selected from resume-backed evidence and JD priorities.
            - Sentence 3: system impact and/or business impact using supported outcomes, metrics, scale, reliability, latency, cost, or operational value.
        11) If editing an experience bullet, keep it local and improve action verb, system/model/data/product context, technical depth, and quantified result where supported.
        12) Avoid comma-separated keyword stuffing.
        13) Improve distinctive candidate strengths and role alignment only through evidence-backed wording.
           Do NOT use generic claims like "application strong fit", "strong fit", "perfect fit", or "ideal candidate".
        14) Do not include optimization notes, explanations, comments, target role alignment sections, or any text outside the resume.

        Evaluator weak sections:
        {weak_sections}

        Evaluator weak bullets:
        {weak_bullets}

        Evaluator factual risks:
        {factual_risks}

        Evaluator editor instructions:
        {editor_instructions}

        Top remaining keyword gaps:
        {missing}

        Resume-safe improvements from the ATS simulation:
        {safe_improvements}

        Evidence gaps that must NOT be claimed unless supported by the original resume:
        {evidence_gaps}

        Extracted job requirements:
        {requirements_json}

        Resume evidence map:
        {evidence_map_json}

        Original resume facts:
        {resume_excerpt}

        Job description:
        {job_excerpt}

        Current tailored resume to edit locally:
        {current_resume_excerpt}
        """
    ).strip()


def optimize_resume_content(
    job: JobRecord,
    job_text: str,
    base_resume_text: str,
    baseline_analysis: dict[str, Any],
    cv_understanding: str,
    requirements: dict[str, Any],
    evidence_map: dict[str, Any],
    client: OpenAI,
    model: str,
    target_score: float,
    max_retries: int = 3,
) -> tuple[str, dict[str, Any], str, list[dict[str, Any]], str, str]:
    generation_mode = "llm"
    edit_attempts: list[dict[str, Any]] = []
    original_score = _analysis_score(baseline_analysis)
    rewrite_decision = estimate_rewrite_opportunity(
        baseline_analysis,
        evidence_map,
        target_score,
        lift_threshold=_rewrite_lift_threshold(),
    )
    if not rewrite_decision["should_rewrite"]:
        baseline_analysis["stop_reason"] = str(rewrite_decision["reason"])
        resume_edit_log = build_resume_edit_log(
            job,
            target_score,
            original_score,
            "original-kept",
            "original",
            [],
            edit_attempts,
            baseline_analysis,
            rewrite_decision,
        )
        return base_resume_text, baseline_analysis, "original-kept", [], "original", resume_edit_log

    prompts = build_prompt_bundle(
        job,
        job_text,
        base_resume_text,
        baseline_analysis,
        cv_understanding,
        requirements=requirements,
        evidence_map=evidence_map,
    )
    tailored_resume = require_llm_generate(client, model, prompts["resume"], "tailored resume")

    optimized_analysis = llm_ats_score(
        client,
        model,
        job,
        job_text,
        tailored_resume,
        requirements=requirements,
        evidence_map=evidence_map,
    )
    evaluation_history = [
        {
            "attempt": 0,
            "mode": "baseline_full_generation",
            "analysis": optimized_analysis,
        }
    ]
    generated_score = _analysis_score(optimized_analysis)
    original_risk_count = _factual_risk_count(baseline_analysis)
    generated_risk_count = _factual_risk_count(optimized_analysis)
    acceptable_risk_count = _acceptable_factual_risk_count()
    baseline_readability = float((baseline_analysis.get("category_scores", {}) or {}).get("ats_readability", 0) or 0)
    generated_readability = float((optimized_analysis.get("category_scores", {}) or {}).get("ats_readability", 0) or 0)
    readability_not_worse = generated_readability >= baseline_readability
    factual_risk_acceptable = generated_risk_count <= acceptable_risk_count and generated_risk_count <= original_risk_count
    if generated_score <= original_score or not factual_risk_acceptable or not readability_not_worse:
        if generated_score <= original_score:
            optimized_analysis["stop_reason"] = "rejected_no_score_improvement"
        elif not factual_risk_acceptable:
            optimized_analysis["stop_reason"] = "rejected_factual_risk_increase"
        else:
            optimized_analysis["stop_reason"] = "rejected_readability_regression"
        generation_mode = "original-kept"
        accepted_resume_version = "original"
        baseline_analysis["stop_reason"] = "kept_original_after_rejected_rewrite"
        resume_edit_log = build_resume_edit_log(
            job,
            target_score,
            original_score,
            generation_mode,
            accepted_resume_version,
            evaluation_history,
            edit_attempts,
            baseline_analysis,
            rewrite_decision,
        )
        return base_resume_text, baseline_analysis, generation_mode, evaluation_history, accepted_resume_version, resume_edit_log

    retries = 0
    while not ats_stop_condition_met(optimized_analysis, target_score) and retries < max_retries:
        previous_resume = tailored_resume
        previous_analysis = optimized_analysis
        previous_score = _analysis_score(optimized_analysis)
        previous_version = _version_label(len(evaluation_history))
        candidate_version = _version_label(len(evaluation_history) + 1)
        editor_prompt = build_section_editor_prompt(
            job,
            job_text,
            base_resume_text,
            tailored_resume,
            optimized_analysis,
            requirements,
            evidence_map,
            retries + 1,
        )
        edited_resume = require_llm_generate(client, model, editor_prompt, "section editor retry")

        tailored_resume = edited_resume
        optimized_analysis = llm_ats_score(
            client,
            model,
            job,
            job_text,
            tailored_resume,
            requirements=requirements,
            evidence_map=evidence_map,
        )
        evaluation_history.append(
            {
                "attempt": retries + 1,
                "mode": "local_section_edit",
                "analysis": optimized_analysis,
            }
        )
        generation_mode = "llm-edited"
        retries += 1
        current_score = _analysis_score(optimized_analysis)
        score_delta = round(current_score - previous_score, 2)
        current_risk_count = _factual_risk_count(optimized_analysis)
        previous_risk_count = _factual_risk_count(previous_analysis)
        acceptable_risk_count = _acceptable_factual_risk_count()
        previous_readability = float((previous_analysis.get("category_scores", {}) or {}).get("ats_readability", 0) or 0)
        current_readability = float((optimized_analysis.get("category_scores", {}) or {}).get("ats_readability", 0) or 0)
        factual_risk_acceptable = current_risk_count <= acceptable_risk_count and current_risk_count <= previous_risk_count
        accepted_edit = current_score > previous_score and factual_risk_acceptable and current_readability >= previous_readability
        if current_score <= previous_score:
            rejection_reason = "rejected_no_score_improvement"
        elif not factual_risk_acceptable:
            rejection_reason = "rejected_factual_risk_increase"
        elif current_readability < previous_readability:
            rejection_reason = "rejected_readability_regression"
        else:
            rejection_reason = "score_improved"
        edit_attempt = {
            "attempt": retries,
            "from_version": previous_version,
            "to_version": candidate_version,
            "previous_score": previous_score,
            "candidate_score": current_score,
            "score_delta": score_delta,
            "accepted": accepted_edit,
            "reason": rejection_reason,
            "diff": _resume_diff(previous_resume, edited_resume, f"{previous_version}.md", f"{candidate_version}.md"),
        }
        edit_attempts.append(edit_attempt)
        if not accepted_edit:
            optimized_analysis["stop_reason"] = rejection_reason
            previous_analysis["stop_reason"] = "kept_previous_after_no_score_improvement"
            tailored_resume = previous_resume
            optimized_analysis = previous_analysis
            break

    if "stop_reason" not in optimized_analysis:
        if ats_stop_condition_met(optimized_analysis, target_score):
            optimized_analysis["stop_reason"] = "target_met"
        elif retries >= max_retries:
            optimized_analysis["stop_reason"] = "max_local_edits_reached"
        else:
            optimized_analysis["stop_reason"] = "stopped"

    if generation_mode == "original-kept":
        accepted_resume_version = "original"
    elif evaluation_history and isinstance(evaluation_history[-1], dict):
        last_analysis = evaluation_history[-1].get("analysis", {})
        if last_analysis.get("stop_reason") == "rejected_no_score_improvement":
            accepted_resume_version = _version_label(max(len(evaluation_history) - 1, 1))
        else:
            accepted_resume_version = _version_label(len(evaluation_history))
    else:
        accepted_resume_version = "unknown"

    resume_edit_log = build_resume_edit_log(
        job,
        target_score,
        original_score,
        generation_mode,
        accepted_resume_version,
        evaluation_history,
        edit_attempts,
        baseline_analysis,
        rewrite_decision,
    )
    return tailored_resume, optimized_analysis, generation_mode, evaluation_history, accepted_resume_version, resume_edit_log


__all__ = ["build_resume_edit_log", "build_section_editor_prompt", "estimate_rewrite_opportunity", "optimize_resume_content"]
