"""Prompt bundle construction for generated artifacts."""

import json
import textwrap
from typing import Any

from job_application_optimizer.config import _prompt_text
from job_application_optimizer.models import JobRecord


def build_prompt_bundle(
    job: JobRecord,
    job_text: str,
    resume_text: str,
    analysis: dict[str, Any],
    cv_understanding: str,
    requirements: dict[str, Any] | None = None,
    evidence_map: dict[str, Any] | None = None,
) -> dict[str, str]:
    missing = ", ".join(k for k, _ in analysis["missing_keywords"][:20])
    matched = ", ".join(k for k, _ in analysis["matched_keywords"][:20])
    safe_improvements = "\n".join(f"- {item}" for item in analysis.get("resume_safe_improvements", [])[:10]) or "- None provided"
    evidence_gaps = "\n".join(f"- {item}" for item in analysis.get("evidence_gaps", [])[:10]) or "- None provided"
    ats_pass_benchmarks = (
        "\n".join(f"- {item}" for item in analysis.get("ats_pass_benchmarks", [])[:8]) or "- None provided"
    )
    requirements_json = json.dumps(requirements or {}, ensure_ascii=False, indent=2)
    evidence_map_json = json.dumps(evidence_map or {}, ensure_ascii=False, indent=2)
    job_excerpt = _prompt_text(job_text)
    resume_excerpt = _prompt_text(resume_text)
    cv_understanding_excerpt = _prompt_text(cv_understanding)

    resume_prompt = textwrap.dedent(
        f"""
        Target role: {job.role}
        Company: {job.company}

        Job description text:
        {job_excerpt}

        Original resume text:
        {resume_excerpt}

        CANDIDATE DEEP UNDERSTANDING (Use this to make intelligent enhancements):
        {cv_understanding_excerpt}

        EXTRACTED JOB REQUIREMENTS:
        {requirements_json}

        RESUME EVIDENCE MAP:
        {evidence_map_json}

        Matched keywords: {matched}
        Missing keywords: {missing}

        LLM ATS simulation - resume-safe improvements:
        {safe_improvements}

        LLM ATS simulation - real evidence gaps to avoid claiming:
        {evidence_gaps}

        TASK: Create a truthful, high-density technical resume tailored to the role while remaining ATS-friendly.
        This is the baseline full rewrite. Later iterations will only edit weak sections, so make the initial structure clean and stable.

        INSTRUCTIONS:
        1) Optimize for two readers at once:
           - ATS: include exact JD terminology only where supported by the candidate's evidence.
           - Recruiter/hiring manager: make ownership, role fit, technical depth, domain context, and measurable impact easy to scan.
        2) Reorder bullets to emphasize the most relevant and highest-signal experience first.
        3) Replace generic phrasing with specific, JD-aligned terminology where applicable.
           Use natural keyword replacement: prefer the JD's phrasing when it accurately describes the same work, but keep the sentence fluent and credible.
           Good: use a JD term when the resume supports the same capability, and tie it to a specific project, method, or outcome.
           Bad: append a broad list of loosely related JD keywords without evidence or sentence-level purpose.
        4) You may translate demonstrated experience into equivalent industry terminology, but do not introduce a skill, tool, framework, security practice, or architecture pattern unless the original resume explicitly supports it.
           If a JD keyword is adjacent but not proven, omit it from the resume and leave it for gap diagnosis instead.
        5) Expand bullet points only with context that clarifies actual experience. Do not inflate scope, seniority, team size, domain, or ownership beyond the provided facts.
        6) DO NOT fabricate projects, roles, companies, quantified metrics, tools, leadership scope, security practices, or domain expertise.
        7) Use ATS-friendly sections and formatting: simple headings, plain text bullets, no tables, no columns, no graphics, no icons.
        8) Target a one-page resume. Use these sections only:
           - Header
           - Professional Summary, 45-70 words
           - Core Skills, maximum 4 lines
           - Professional Experience
           - Education
           - Selected Research, only if directly relevant to the role
        9) Enforce experience bullet counts in final output:
           - Most recent experience: exactly 5 bullets.
           - Second most recent experience: exactly 4 bullets.
           - Every remaining experience (if any): exactly 3 bullets each.
        10) Keep only the most important and most JD-aligned achievements.
           You may merge overlapping bullets and rewrite/refine wording for clarity and impact.
        11) Each experience bullet must follow this standard:
            - Strong action verb
            - Specific system/model/data/product context
            - Technical depth or design decision
            - Quantified result where supported
        12) Avoid comma-separated keyword stuffing. Prefer fewer, stronger bullets over broad lists of tools.
        13) Highlight distinctive candidate strengths and role alignment using concrete evidence.
           Do NOT use generic claims like "application strong fit", "strong fit", "perfect fit", or "ideal candidate".
        14) Write the Professional Summary in this style:
            - Sentence 1: identity positioning + core experience most relevant to the JD.
            - Sentence 2: core skill stack, expressed naturally and only with resume-backed capabilities.
            - Sentence 3: system impact and/or business impact, preferably with supported metrics.
            Do not hardcode one reusable summary. Generate a role-aware summary from the JD, resume facts, and ATS simulation.
        15) Do not include Target Role Alignment, Optimization Notes, explanations, comments, or any text outside the resume.
        """
    ).strip()

    cover_prompt = textwrap.dedent(
        f"""
        Write a sharp, high-conviction cover letter for {job.role} at {job.company} ({job.location}).
        Use only the resume facts provided below. Do not fabricate employers, projects, impact, technologies, leadership scope, or domain expertise.

        Quality bar:
        1) Sound specific to this role and company, not generic.
        2) Open with concrete, resume-backed evidence of alignment to this role.
        3) Highlight 2 or 3 resume-backed strengths that map directly to the JD.
        4) Acknowledge one growth area only if it can be framed constructively.
        5) Keep the tone professional, credible, and concise.
        6) Avoid empty phrases like "I am passionate", "team player", "strong fit", "perfect fit", or generic enthusiasm without evidence.
        7) Output plain text only, 220-320 words.

        Structure:
        - Paragraph 1: why this role and strongest fit
        - Paragraph 2: most relevant achievements or experience evidence
        - Paragraph 3: close with value proposition and next-step intent

        Job description:
          {job_excerpt}

        Resume:
          {resume_excerpt}
        """
    ).strip()

    interview_prompt = textwrap.dedent(
        f"""
        Company: {job.company}
        Role: {job.role}
          Create a comprehensive interview prep pack that is specific, strategic, and evidence-based.

          REQUIRED OUTPUT STRUCTURE:
          1) Role Fit Snapshot
              - 4 to 6 strengths and 2 to 3 risks for this role (all resume-backed).

          2) Why This Company + Motivation Narrative
              - A polished answer to "Why do you want to join {job.company}?" (180 to 260 words).
              - Include:
                 a) What specifically about the company and role is attractive.
                 b) Why the candidate is motivated beyond compensation/title.
                 c) What impact the candidate wants to create in first 12 months.
              - Add a short "Evidence to mention" list (3 to 5 bullets) drawn from resume facts.

          3) Core Questions (must include both)
              - Question 1: Why do you want to join {job.company}?
                 Provide a high-quality sample answer and 3 concise talking-point variants.
              - Question 2: Tell me about yourself.
                 Provide a 2-minute narrative answer (about 280 to 360 words) plus a 30-second version.

          4) Additional High-Probability Questions
              - Provide 6 role-relevant questions with STAR-style answer frameworks.
              - For each: include "What interviewer is testing" and "How to answer".

          5) Online Assessment Preparation
              - List likely assessment themes for this role and sample question styles.

          6) Final Coaching Notes
              - Common pitfalls to avoid, and final personalization tips before interview.

          Constraints:
          - Use only facts supported by resume and CV understanding.
          - Do not fabricate employers, projects, metrics, leadership scope, or tech stack.
          - Keep the tone practical and interview-ready (not generic textbook advice).
          - Do not use generic labels like "strong fit"; show fit via specific achievements and capabilities.

        JD:
          {job_excerpt}

        Resume:
          {resume_excerpt}

        Candidate deep understanding:
          {cv_understanding_excerpt}
        """
    ).strip()

    gap_prompt = textwrap.dedent(
        f"""
        The candidate's tailored resume for {job.role} at {job.company} is still below the target ATS score.
        Produce a precise gap diagnosis based only on the JD and original resume facts.

        Output requirements:
        1) Explain why the score is still below target.
        2) Separate missing evidence from missing wording.
        3) List the top 5 unresolved JD keywords or capability areas.
        4) State which gaps can be fixed by rewriting and which require real experience.
        5) Include ATS-pass benchmark examples for the JD-priority weak points. Clearly separate:
           - what the current resume already proves,
           - what a high-screening resume would prove,
           - whether the gap is a rewrite issue or requires real new experience.
        6) Give a short action plan for improving future applications.
        7) Output plain text only.

        Current analysis:
        Matched keywords: {matched}
        Missing keywords: {missing}
        Resume-safe improvements: {safe_improvements}
        Evidence gaps: {evidence_gaps}
        ATS-pass benchmarks: {ats_pass_benchmarks}

        Job description:
        {job_excerpt}

        Resume:
        {resume_excerpt}
        """
    ).strip()

    return {
        "resume": resume_prompt,
        "cover_letter": cover_prompt,
        "interview": interview_prompt,
        "gap": gap_prompt,
    }


__all__ = ["build_prompt_bundle"]
