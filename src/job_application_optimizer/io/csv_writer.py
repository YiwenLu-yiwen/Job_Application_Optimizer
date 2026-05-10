"""CSV report writers and human-readable cell formatting."""

import csv
import json
import re
from pathlib import Path
from typing import Any

from job_application_optimizer.models import COMPLETED_FIELDNAMES, JobRecord


def inline_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(item for item in (inline_csv_value(item).strip() for item in value) if item)
    if isinstance(value, dict):
        return "; ".join(
            f"{key}: {inline_csv_value(item).strip()}"
            for key, item in value.items()
            if inline_csv_value(item).strip()
        )
    return str(value)


def csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, (list, tuple)):
        items = [inline_csv_value(item).strip() for item in value]
        items = [item for item in items if item]
        return "\n".join(f"- {item}" for item in items)
    if isinstance(value, dict):
        items = [
            f"{key}: {inline_csv_value(item).strip()}"
            for key, item in value.items()
            if inline_csv_value(item).strip()
        ]
        return "\n".join(f"- {item}" for item in items)
    return str(value)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key, "")) for key in fieldnames})


def text_tokens(value: Any) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z0-9+#.]+", str(value).lower()) if len(token) > 2}


def requirement_priority_ranks(requirements: dict[str, Any], requirement_items: list[Any]) -> dict[int, int]:
    valid_requirements = [
        (index, requirement)
        for index, requirement in enumerate(requirement_items, start=1)
        if isinstance(requirement, dict)
    ]
    screening_priorities = [
        str(item).strip()
        for item in requirements.get("screening_priorities", []) or []
        if str(item).strip()
    ]
    importance_rank = {"must_have": 0, "important": 1, "nice_to_have": 2}

    matched_priority_by_index: dict[int, int] = {}
    matched_requirements: set[int] = set()
    for priority_index, priority in enumerate(screening_priorities, start=1):
        priority_tokens = text_tokens(priority)
        if not priority_tokens:
            continue

        best_index = 0
        best_score = 0.0
        for requirement_index, requirement in valid_requirements:
            if requirement_index in matched_requirements:
                continue
            searchable_text = " ".join(
                [
                    str(requirement.get("requirement", "")),
                    str(requirement.get("category", "")),
                    csv_cell(requirement.get("ats_keywords", [])),
                    str(requirement.get("evidence_expected", "")),
                ]
            )
            requirement_tokens = text_tokens(searchable_text)
            if not requirement_tokens:
                continue
            overlap = len(priority_tokens & requirement_tokens)
            score = overlap / max(len(priority_tokens), len(requirement_tokens))
            if score > best_score:
                best_index = requirement_index
                best_score = score

        if best_index and best_score >= 0.25:
            matched_priority_by_index[best_index] = priority_index
            matched_requirements.add(best_index)

    sorted_requirements = sorted(
        valid_requirements,
        key=lambda item: (
            0 if item[0] in matched_priority_by_index else 1,
            matched_priority_by_index.get(item[0], 999),
            importance_rank.get(str(item[1].get("importance", "")).strip(), 3),
            item[0],
        ),
    )
    return {requirement_index: rank for rank, (requirement_index, _) in enumerate(sorted_requirements, start=1)}


def requirement_evidence_rows(requirements: dict[str, Any], evidence_map: dict[str, Any]) -> list[dict[str, Any]]:
    requirement_items = requirements.get("requirements", [])
    evidence_items = evidence_map.get("requirement_map", [])
    if not isinstance(requirement_items, list):
        requirement_items = []
    if not isinstance(evidence_items, list):
        evidence_items = []

    evidence_by_id = {
        str(item.get("id", "")).strip(): item
        for item in evidence_items
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    }
    coverage_score = {"strong": 100, "partial": 65, "adjacent": 40, "missing": 0}
    priority_rank_by_index = requirement_priority_ranks(requirements, requirement_items)

    rows: list[dict[str, Any]] = []
    for index, requirement in enumerate(requirement_items, start=1):
        if not isinstance(requirement, dict):
            continue
        req_id = str(requirement.get("id") or f"R{index}").strip()
        evidence = evidence_by_id.get(req_id, {})
        coverage = str(evidence.get("coverage") or "missing").strip()
        if coverage == "strong":
            action = "Feature prominently; evidence is strong."
        elif coverage == "partial":
            action = "Strengthen wording with supported evidence and JD terminology."
        elif coverage == "adjacent":
            action = "Position as transferable evidence; avoid claiming direct experience."
        else:
            action = "Do not claim; treat as a real evidence gap."

        rows.append(
            {
                "priority_rank": priority_rank_by_index.get(index, index),
                "requirement_id": req_id,
                "requirement": requirement.get("requirement", ""),
                "category": requirement.get("category", ""),
                "importance": requirement.get("importance", ""),
                "ats_keywords": requirement.get("ats_keywords", []),
                "evidence_expected": requirement.get("evidence_expected", ""),
                "coverage": coverage,
                "coverage_score": coverage_score.get(coverage, ""),
                "resume_evidence": evidence.get("resume_evidence", []),
                "safe_resume_terms": evidence.get("safe_resume_terms", []),
                "missing_evidence": evidence.get("missing_evidence", []),
                "factual_risk": evidence.get("factual_risk", ""),
                "positioning_advice": evidence.get("positioning_advice", ""),
                "recommended_action": action,
            }
        )
    return sorted(rows, key=lambda row: int(row.get("priority_rank", 999)))


def write_requirements_evidence_csv(path: Path, requirements: dict[str, Any], evidence_map: dict[str, Any]) -> None:
    fieldnames = [
        "priority_rank",
        "requirement_id",
        "requirement",
        "category",
        "importance",
        "ats_keywords",
        "evidence_expected",
        "coverage",
        "coverage_score",
        "resume_evidence",
        "safe_resume_terms",
        "missing_evidence",
        "factual_risk",
        "positioning_advice",
        "recommended_action",
    ]
    write_csv(path, fieldnames, requirement_evidence_rows(requirements, evidence_map))


def keyword_names(pairs: Any) -> list[str]:
    if not isinstance(pairs, list):
        return []
    names = []
    for item in pairs:
        if isinstance(item, (list, tuple)) and item:
            names.append(str(item[0]))
        elif isinstance(item, dict):
            names.append(str(item.get("keyword") or item.get("term") or item.get("skill") or ""))
        elif isinstance(item, str):
            names.append(item)
    return [name for name in names if name]


def write_completed_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, COMPLETED_FIELDNAMES, rows)


def append_completed_all_csv(path: Path, rows: list[dict[str, Any]] | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, dict):
        rows = [rows]
    file_exists = path.exists()

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMPLETED_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_cell(row.get(k, "")) for k in COMPLETED_FIELDNAMES})


def write_ranked_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "job_id",
        "company",
        "role",
        "original_score",
        "optimized_score",
        "target_score",
        "meets_target",
        "accepted_resume_version",
        "gap_type",
        "gap_summary",
        "screening_recommendation",
        "output_folder",
    ]
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            bool(row.get("meets_target")),
            float(row.get("optimized_score", 0) or 0),
            float(row.get("original_score", 0) or 0),
        ),
        reverse=True,
    )
    payloads = []
    for index, row in enumerate(sorted_rows, start=1):
        payload = {k: row.get(k, "") for k in fieldnames}
        payload["rank"] = index
        payloads.append(payload)
    write_csv(path, fieldnames, payloads)


def write_evaluation_history_csv(path: Path, evaluation_history: list[dict[str, Any]]) -> None:
    rows = []
    previous_score: float | None = None
    for index, item in enumerate(evaluation_history, start=1):
        analysis = item.get("analysis", {}) if isinstance(item, dict) else {}
        score = float(analysis.get("score", 0) or 0)
        score_delta = "" if previous_score is None else round(score - previous_score, 2)
        rows.append(
            {
                "version": f"version{index}",
                "attempt": item.get("attempt", index - 1),
                "mode": item.get("mode", ""),
                "score": score,
                "score_delta": score_delta,
                "improved": "" if previous_score is None else score > previous_score,
                "gap_type": analysis.get("gap_type", ""),
                "gap_summary": analysis.get("gap_summary", ""),
                "screening_recommendation": analysis.get("screening_recommendation", ""),
                "stop_reason": analysis.get("stop_reason", ""),
                "factual_risk_count": len(analysis.get("factual_risks", []) or []),
                "factual_risks": analysis.get("factual_risks", []),
                "category_scores": analysis.get("category_scores", {}),
                "matched_keywords": keyword_names(analysis.get("matched_keywords", [])),
                "missing_keywords": keyword_names(analysis.get("missing_keywords", [])),
                "weak_sections": analysis.get("weak_sections", []),
                "weak_bullets": analysis.get("weak_bullets", []),
                "resume_safe_improvements": analysis.get("resume_safe_improvements", []),
                "evidence_gaps": analysis.get("evidence_gaps", []),
                "ats_pass_benchmarks": analysis.get("ats_pass_benchmarks", []),
                "editor_instructions": analysis.get("editor_instructions", []),
                "rationale": analysis.get("rationale", ""),
            }
        )
        previous_score = score

    fieldnames = list(rows[0].keys()) if rows else ["version", "attempt", "mode", "score"]
    write_csv(path, fieldnames, rows)


def write_analysis_summary_csv(
    path: Path,
    job: JobRecord,
    baseline_analysis: dict[str, Any],
    optimized_analysis: dict[str, Any],
    target_score: float,
    meets_target: bool,
    generation_mode: str,
    gap_summary: str,
    requirements: dict[str, Any],
    evidence_map: dict[str, Any],
    evaluation_history: list[dict[str, Any]],
    accepted_resume_version: str,
) -> None:
    matrix_rows = requirement_evidence_rows(requirements, evidence_map)
    coverage_counts: dict[str, int] = {}
    for row in matrix_rows:
        coverage = str(row.get("coverage") or "missing")
        coverage_counts[coverage] = coverage_counts.get(coverage, 0) + 1

    original_score = float(baseline_analysis.get("score", 0) or 0)
    optimized_score = float(optimized_analysis.get("score", 0) or 0)
    row = {
        "company": job.company,
        "role": job.role,
        "location": job.location,
        "url": job.url,
        "original_score": original_score,
        "optimized_score": optimized_score,
        "score_delta": round(optimized_score - original_score, 2),
        "target_score": target_score,
        "meets_target": meets_target,
        "generation_mode": generation_mode,
        "accepted_resume_version": accepted_resume_version,
        "edit_versions": max(len(evaluation_history) - 1, 0),
        "stop_reason": optimized_analysis.get("stop_reason", ""),
        "gap_type": optimized_analysis.get("gap_type", ""),
        "gap_summary": gap_summary,
        "screening_recommendation": optimized_analysis.get("screening_recommendation", ""),
        "factual_risk_count": len(optimized_analysis.get("factual_risks", []) or []),
        "factual_risks": optimized_analysis.get("factual_risks", []),
        "top_missing_keywords": keyword_names(optimized_analysis.get("missing_keywords", []))[:8],
        "top_safe_improvements": (optimized_analysis.get("resume_safe_improvements", []) or [])[:5],
        "top_evidence_gaps": (optimized_analysis.get("evidence_gaps", []) or [])[:5],
        "top_ats_pass_benchmarks": (optimized_analysis.get("ats_pass_benchmarks", []) or [])[:5],
        "weak_sections": optimized_analysis.get("weak_sections", []),
        "weak_bullets": optimized_analysis.get("weak_bullets", []),
        "requirements_count": len(matrix_rows),
        "strong_coverage_count": coverage_counts.get("strong", 0),
        "partial_coverage_count": coverage_counts.get("partial", 0),
        "adjacent_coverage_count": coverage_counts.get("adjacent", 0),
        "missing_coverage_count": coverage_counts.get("missing", 0),
        "role_summary": requirements.get("role_summary", ""),
        "safe_positioning_strategy": evidence_map.get("safe_positioning_strategy", ""),
    }
    write_csv(path, list(row.keys()), [row])


__all__ = [
    "append_completed_all_csv",
    "csv_cell",
    "inline_csv_value",
    "keyword_names",
    "requirement_evidence_rows",
    "requirement_priority_ranks",
    "text_tokens",
    "write_analysis_summary_csv",
    "write_completed_csv",
    "write_csv",
    "write_evaluation_history_csv",
    "write_ranked_csv",
    "write_requirements_evidence_csv",
]
