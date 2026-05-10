"""LLM JSON parsing and normalization helpers."""

import json
import re
from typing import Any


def parse_llm_json(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON response must be an object")
    return parsed


def _keyword_pairs(values: Any) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    if not isinstance(values, list):
        return pairs

    for item in values:
        if isinstance(item, str):
            keyword = item.strip()
            weight = 1
        elif isinstance(item, dict):
            keyword = str(item.get("keyword") or item.get("term") or item.get("skill") or "").strip()
            raw_weight = item.get("weight") or item.get("importance") or 1
            try:
                weight = int(raw_weight)
            except (TypeError, ValueError):
                weight = 1
        elif isinstance(item, (list, tuple)) and item:
            keyword = str(item[0]).strip()
            try:
                weight = int(item[1]) if len(item) > 1 else 1
            except (TypeError, ValueError):
                weight = 1
        else:
            continue

        if keyword:
            pairs.append((keyword, max(weight, 1)))
    return pairs


def _bool_or_default(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1"}:
            return True
        if normalized in {"false", "no", "n", "0"}:
            return False
    return default


def _ats_pass_benchmarks(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in values:
        if not isinstance(item, dict):
            continue

        high_score_benchmark = str(item.get("high_score_benchmark") or "").strip()
        requirement = str(item.get("requirement") or "").strip()
        current_gap = str(item.get("current_candidate_gap") or "").strip()
        if not any((high_score_benchmark, requirement, current_gap)):
            continue

        requires_new_experience = _bool_or_default(item.get("requires_new_experience"), False)
        if "can_be_improved_by_rewrite" in item:
            can_be_improved_by_rewrite = _bool_or_default(item.get("can_be_improved_by_rewrite"), False)
        else:
            can_be_improved_by_rewrite = not requires_new_experience

        normalized.append(
            {
                "requirement_id": str(item.get("requirement_id") or item.get("id") or "").strip(),
                "requirement": requirement,
                "jd_priority": str(item.get("jd_priority") or item.get("importance") or "").strip(),
                "coverage": str(item.get("coverage") or "").strip(),
                "screening_gate": _bool_or_default(item.get("screening_gate"), True),
                "current_resume_signal": str(item.get("current_resume_signal") or "").strip(),
                "current_candidate_gap": current_gap,
                "high_score_benchmark": high_score_benchmark,
                "why_it_matters": str(item.get("why_it_matters") or "").strip(),
                "gap_size": str(item.get("gap_size") or "").strip(),
                "can_be_improved_by_rewrite": can_be_improved_by_rewrite,
                "requires_new_experience": requires_new_experience,
                "safe_positioning": str(item.get("safe_positioning") or "").strip(),
            }
        )
    return normalized


def normalize_llm_ats_result(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        score = float(payload.get("score", 0))
    except (TypeError, ValueError):
        score = 0.0
    score = round(max(0.0, min(100.0, score)), 2)

    return {
        "score": score,
        "matched_keywords": _keyword_pairs(payload.get("matched_keywords")),
        "missing_keywords": _keyword_pairs(payload.get("missing_keywords")),
        "category_scores": payload.get("category_scores", {}),
        "resume_safe_improvements": payload.get("resume_safe_improvements", []),
        "evidence_gaps": payload.get("evidence_gaps", []),
        "ats_pass_benchmarks": _ats_pass_benchmarks(payload.get("ats_pass_benchmarks")),
        "weak_sections": payload.get("weak_sections", []),
        "weak_bullets": payload.get("weak_bullets", []),
        "factual_risks": payload.get("factual_risks", []),
        "editor_instructions": payload.get("editor_instructions", []),
        "gap_type": payload.get("gap_type", ""),
        "gap_summary": payload.get("gap_summary", ""),
        "rationale": payload.get("rationale", ""),
        "screening_recommendation": payload.get("screening_recommendation", ""),
        "scoring_version": "llm_ats_sim_v1",
    }


__all__ = ["_ats_pass_benchmarks", "_keyword_pairs", "normalize_llm_ats_result", "parse_llm_json"]
