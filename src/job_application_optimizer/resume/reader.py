"""Resume file readers."""

from pathlib import Path


def read_resume_text(resume_path: Path) -> str:
    suffix = resume_path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(resume_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

    if suffix == ".docx":
        from docx import Document

        doc = Document(str(resume_path))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs).strip()

    if suffix in {".txt", ".md"}:
        return resume_path.read_text(encoding="utf-8")

    raise ValueError(f"Unsupported resume format: {resume_path}")


__all__ = ["read_resume_text"]
