"""Capability inventory generation and loading."""

import argparse
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from job_application_optimizer.config import _prompt_text
from job_application_optimizer.llm.client import LLMRouter, ModelRole, require_llm_router
from job_application_optimizer.llm.schemas import CapabilityInventoryDraft
from job_application_optimizer.llm.structured import generate_structured
from job_application_optimizer.resume.cache import read_fresh_cache, resume_sha256, write_cache
from job_application_optimizer.resume.understanding import read_resume

load_dotenv()


@dataclass(frozen=True)
class CapabilityInventoryStatus:
    path: Path
    draft_path: Path
    usable: bool
    generated_draft: bool
    content: str
    message: str


def default_draft_path(inventory_path: Path) -> Path:
    return inventory_path.with_name(f"{inventory_path.stem}.draft{inventory_path.suffix or '.yaml'}")


def build_capability_inventory_prompt(resume_text: str, cv_understanding: str) -> str:
    return f"""Create a draft capability inventory YAML from the provided resume and CV deep understanding.

This is a source-of-truth candidate evidence bank for resume tailoring. It must separate verified-looking facts from inferred positioning.

Rules:
- Output JSON only. JSON syntax is used because it is also valid YAML and can be saved as a .yaml file.
- Do not include markdown fences.
- Default every capability to verification_status: needs_review.
- Do not mark anything verified.
- Do not invent tools, metrics, companies, customers, frameworks, model methods, or domain experience.
- Use exact resume evidence where available.
- If a term is only adjacent or inferred, put it in equivalent_terms_allowed or exposure_terms, not exact_terms_allowed.
- Put direct tool names in forbidden_terms when the candidate has adjacent capability but no evidence of exact tool use.

Required schema:
{{
  "version": 1,
  "status": "draft",
  "instructions": "Review manually. Rename/copy to capability_inventory.yaml only after verification.",
  "capabilities": [
    {{
      "name": "",
      "category": "",
      "evidence": [""],
      "metrics": [""],
      "tools_used": [""],
      "exact_terms_allowed": [""],
      "equivalent_terms_allowed": [""],
      "exposure_terms": [""],
      "forbidden_terms": [""],
      "safe_sentence_templates": [""],
      "confidence": "high | medium | low",
      "verification_status": "needs_review",
      "notes": ""
    }}
  ]
}}

Category guidance:
- Use a concise free-text category inferred from the resume, not a fixed taxonomy.
- Examples: technical_delivery, product_strategy, customer_success, operations, research, leadership, compliance, analytics, design, writing, finance, domain_expertise.
- Do not force capabilities into any predefined domain; choose the smallest truthful category supported by the evidence.

Resume:
{_prompt_text(resume_text)}

CV deep understanding:
{_prompt_text(cv_understanding)}
"""


def generate_capability_inventory_draft(resume_text: str, cv_understanding: str, llm: LLMRouter) -> str:
    prompt = build_capability_inventory_prompt(resume_text, cv_understanding)
    result = generate_structured(
        llm,
        ModelRole.CV,
        prompt,
        "capability inventory draft",
        CapabilityInventoryDraft,
        temperature=0.1,
    )
    return f"{result.model_dump_json(indent=2)}\n"


def load_or_generate_capability_inventory(
    resume_path: Path,
    cv_understanding: str,
    inventory_path: Path,
    llm: LLMRouter,
) -> CapabilityInventoryStatus:
    inventory_path = inventory_path.resolve()
    draft_path = default_draft_path(inventory_path)
    resume_hash = resume_sha256(resume_path)
    verified_content = read_fresh_cache(inventory_path, resume_hash)
    if verified_content is not None:
        return CapabilityInventoryStatus(
            path=inventory_path,
            draft_path=draft_path,
            usable=True,
            generated_draft=False,
            content=verified_content,
            message=f"Using verified capability inventory: {inventory_path}",
        )
    draft_content = read_fresh_cache(draft_path, resume_hash)
    if draft_content is not None:
        return CapabilityInventoryStatus(
            path=inventory_path,
            draft_path=draft_path,
            usable=False,
            generated_draft=False,
            content="",
            message=f"Capability inventory draft exists but is not used until verified: {draft_path}",
        )

    resume_text = read_resume(resume_path)
    draft = generate_capability_inventory_draft(resume_text, cv_understanding, llm)
    write_cache(draft_path, draft, resume_hash, "yaml")
    stale_verified_note = " Existing verified inventory was preserved but is stale." if inventory_path.exists() else ""
    return CapabilityInventoryStatus(
        path=inventory_path,
        draft_path=draft_path,
        usable=False,
        generated_draft=True,
        content="",
        message=f"Generated capability inventory draft for review: {draft_path}.{stale_verified_note}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a draft capability inventory from resume and CV understanding")
    parser.add_argument("resume", help="Path to resume file (.pdf/.docx/.txt/.md)")
    parser.add_argument(
        "--cv-understanding",
        default="cv_deep_understanding.md",
        help="Path to existing CV deep-understanding markdown",
    )
    parser.add_argument(
        "--output",
        default="capability_inventory.draft.yaml",
        help="Draft YAML output path",
    )
    args = parser.parse_args()

    resume_path = Path(args.resume).resolve()
    cv_understanding_path = Path(args.cv_understanding).resolve()
    output_path = Path(args.output).resolve()
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume not found: {resume_path}")
    if not cv_understanding_path.exists():
        raise FileNotFoundError(f"CV understanding not found: {cv_understanding_path}")

    resume_hash = resume_sha256(resume_path)
    cv_understanding = read_fresh_cache(cv_understanding_path, resume_hash)
    if cv_understanding is None:
        raise RuntimeError(
            "CV understanding is missing a matching resume hash. Regenerate it before creating a capability inventory."
        )

    llm = require_llm_router()
    draft = generate_capability_inventory_draft(
        read_resume(resume_path),
        cv_understanding,
        llm,
    )
    write_cache(output_path, draft, resume_hash, "yaml")
    print(f"Saved draft capability inventory to: {output_path}")
    print("Review it manually, then copy/rename to capability_inventory.yaml when verified.")


__all__ = [
    "CapabilityInventoryStatus",
    "build_capability_inventory_prompt",
    "default_draft_path",
    "generate_capability_inventory_draft",
    "load_or_generate_capability_inventory",
    "main",
]


if __name__ == "__main__":
    main()
