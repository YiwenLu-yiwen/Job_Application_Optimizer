"""CV deep-understanding generation."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from docx import Document
from pypdf import PdfReader

from job_application_optimizer.llm.client import LLMRouter, ModelRole, require_llm_router

load_dotenv()


def read_resume(resume_path: Path) -> str:
    """Extract text from resume (PDF, DOCX, or TXT)."""
    suffix = resume_path.suffix.lower()
    
    if suffix == ".pdf":
        reader = PdfReader(resume_path)
        return "\n".join([page.extract_text() for page in reader.pages])
    elif suffix == ".docx":
        doc = Document(resume_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif suffix in [".txt", ".md"]:
        return resume_path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def generate_cv_understanding(resume_text: str, llm: LLMRouter) -> str:
    """
    Generate deep CV understanding analysis using LLM.
    """
    prompt = f"""Analyze this resume and generate a comprehensive "CV Deep Understanding" document for job application tailoring.

RESUME:
{resume_text}

---

INSTRUCTIONS: Generate a structured, actionable analysis (Markdown format) with these sections:

## 1. CORE PROFILE
- **Who is this person in one sentence?** (Capture their core identity and unique position)
- **Career arc/progression:** How did they get here? What's the trajectory?
- **What makes them different?** (Specific, not generic - avoid "hard-working" or "team player")

## 2. UNIQUE VALUE PROPOSITION (UVP)
Create a narrative table:
| Dimension | Their Strength | Evidence | Business Value |
|-----------|---|---|---|
| [e.g., Cost Optimization] | [What they did] | [Proof/metric] | [Why it matters] |

Include 5-6 rows covering their key differentiators. Then explain **their story** - why are they uniquely positioned?

## 3. KEY STRENGTHS BY CATEGORY
For **each major skill cluster** (5-7 categories based on resume):
- **What they built/did:** Specific projects or achievements
- **Core capabilities:** What can they actually do?
- **Keywords they own:** Industry terminology they've demonstrated
- **Impact metrics:** Quantifiable results

## 4. KEYWORD MAPPING: CV TERMS → JD ALTERNATIVES
Create a detailed mapping table:
| Their CV Term | Equivalent JD Terminology | Real Meaning | Their Proof | Evidence Level |
|---|---|---|---|---|
| [their phrasing] | [how JDs say it] | [what it means] | [evidence] | Explicit / Adjacent / Gap |

Include 12-15 mappings covering their main skills. This is used for resume tailoring.
Use Evidence Level carefully:
- **Explicit** = can be used directly in resume bullets or skills.
- **Adjacent** = can guide positioning, but should not be written as a claimed skill/tool unless the resume also supports it.
- **Gap** = relevant to target roles but not supported; keep out of the resume.

## 5. HIDDEN STRENGTHS (What CV Implies)
List 4-5 things NOT explicitly stated but clearly demonstrated:
- **Leadership & Ownership:** Evidence of initiative
- **Problem-Solving:** Innovation, optimization, efficiency gains
- **Business Impact:** Revenue, retention, cost savings
- **Reliability & Precision:** Quality focus, high standards
- **[Other based on resume]**

For each hidden strength, label it as either:
- **Resume-safe**: clearly demonstrated and safe to use in resume wording.
- **Positioning-only**: useful for strategy/interview framing, but too inferential for resume claims.

## 6. POSITIONING BY ROLE TYPE
For **3-4 most relevant role types** (inferred from their background):
- **Lead with:** [Most relevant value proposition]
- **Emphasize:** [Key strengths for this role]
- **Proof/Metrics:** [Specific evidence]

Example roles: "AI Architect", "ML Engineer", "Data Scientist", etc.

## 7. KEYWORD CLUSTERS FOR RESUME TAILORING
When matching to job descriptions, use this guide:
- **If JD says "X"** → Emphasize [Y from their resume]
- Include 8-10 such mappings

## 8. ACTIONABLE RESUME REFINEMENT PRINCIPLES
Provide 6-8 specific, actionable principles for this person to tailor their resume:
1. [Principle 1 with why and how]
2. [Principle 2 with why and how]
etc.

## 9. 30-SECOND PITCH
Write a compelling, confident 30-second introduction for interviews/opening paragraphs.
Should sound natural, mention key achievements and unique positioning.

## 10. SUMMARY
2-3 paragraph executive summary: Who are they? What's their unique value? How do they stand out?

---

CRITICAL REQUIREMENTS:
✓ Be SPECIFIC and GROUNDED - use actual resume details, not generic advice
✓ Use CONCRETE METRICS - "increased retention 15%" not "strong business acumen"
✓ DIFFERENTIATE them - what makes them uniquely qualified, not what any AI engineer has
✓ IDENTIFY REAL GAPS - mention what's not in their CV too (e.g., "no explicit team management")
✓ ACTIONABLE - every section should help with resume tailoring and job matching
✓ STRATEGIC - think like a hiring manager trying to understand what this person uniquely brings
✓ Use EXACT PHRASING from their resume when creating mappings
✓ Separate explicit resume evidence from adjacent/inferred positioning so later resume generation does not overclaim"""

    system_instruction = (
        "You are an elite career strategist and CV analyst with 15+ years experience helping "
        "top technical professionals land their dream roles. Provide deep, strategic insights grounded in actual content. "
        "Focus on what differentiates them, not generic strengths. Every statement must be specific and actionable."
    )
    
    return llm.generate(
        ModelRole.CV,
        f"{system_instruction}\n\n{prompt}",
        "CV deep understanding",
        temperature=0.2,
    )


def load_or_generate_cv_understanding(resume_path: Path, llm: LLMRouter, cache_path: Path) -> str:
    cache_path = cache_path.resolve()
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    resume_text = read_resume(resume_path)
    analysis = generate_cv_understanding(resume_text, llm)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(analysis, encoding="utf-8")
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Generate CV Deep Understanding Analysis")
    parser.add_argument("resume", help="Path to resume file (.pdf/.docx/.txt/.md)")
    parser.add_argument("--output", "-o", help="Output file path (default: cv_deep_understanding.md)")
    parser.add_argument("--json-output", help="Also save analysis summary as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print analysis to stdout")
    args = parser.parse_args()
    
    # Load resume
    resume_path = Path(args.resume).resolve()
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume not found: {resume_path}")
    
    print(f"📄 Reading resume: {resume_path}")
    resume_text = read_resume(resume_path)
    print(f"✓ Resume length: {len(resume_text)} characters")
    
    # Generate understanding
    print("\n🤖 Generating CV Deep Understanding using LLM...")
    llm = require_llm_router()

    analysis = generate_cv_understanding(resume_text, llm)
    
    # Save to file
    output_path = Path(args.output or "cv_deep_understanding.md")
    output_path.write_text(analysis, encoding="utf-8")
    print(f"✓ Saved to: {output_path}")
    
    # Optional: save JSON summary
    if args.json_output:
        summary = {
            "resume_path": str(resume_path),
            "resume_length": len(resume_text),
            "analysis_length": len(analysis),
            "timestamp": str(Path.cwd()),
        }
        json_path = Path(args.json_output)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"✓ Saved summary to: {json_path}")
    
    # Optional: print to stdout
    if args.verbose:
        print("\n" + "="*80)
        print("CV DEEP UNDERSTANDING ANALYSIS")
        print("="*80 + "\n")
        print(analysis)
    
    print("\n✅ Done!")


__all__ = [
    "generate_cv_understanding",
    "load_or_generate_cv_understanding",
    "main",
    "read_resume",
]


if __name__ == "__main__":
    main()
