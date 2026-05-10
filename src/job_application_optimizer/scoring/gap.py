"""Gap summary helpers."""

from typing import Any

from job_application_optimizer.scoring.ats import has_factual_risk


def build_gap_summary(analysis: dict[str, Any], target_score: float) -> str:
    score = float(analysis.get("score", 0) or 0)
    if score >= target_score and not has_factual_risk(analysis):
        return "Meets target: strong enough match for ATS/recruiter screen."
    if score >= target_score and has_factual_risk(analysis):
        return "factual_risk: score meets target, but evaluator found claims that need removal, softening, or verification."

    gap_type = str(analysis.get("gap_type") or "gap").strip()
    summary = str(analysis.get("gap_summary") or "").strip()
    if summary:
        return f"{gap_type}: {summary}"[:220]

    evidence_gaps = [str(item).strip() for item in analysis.get("evidence_gaps", []) if str(item).strip()]
    safe_improvements = [str(item).strip() for item in analysis.get("resume_safe_improvements", []) if str(item).strip()]
    missing = [keyword for keyword, _ in analysis.get("missing_keywords", [])[:3]]

    if evidence_gaps:
        return f"{gap_type}: real evidence gap - {evidence_gaps[0]}"[:220]
    if safe_improvements:
        return f"{gap_type}: mostly wording gap - {safe_improvements[0]}"[:220]
    if missing:
        return f"{gap_type}: missing or weakly evidenced areas - {', '.join(missing)}"[:220]
    return f"{gap_type}: below target; review category scores and evidence gaps."[:220]


__all__ = ["build_gap_summary"]
