"""Job page and URL parsing helpers."""

from pathlib import Path

from bs4 import BeautifulSoup

from job_application_optimizer.models import JobRecord


def extract_clean_job_text(html: str) -> str:
    try:
        import trafilatura
    except ModuleNotFoundError:
        extracted = None
    else:
        extracted = trafilatura.extract(html, include_comments=False, include_tables=True)
    if extracted:
        return extracted

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def parse_urls_file(path: Path) -> list[JobRecord]:
    jobs: list[JobRecord] = []
    index = 1
    for raw in path.read_text(encoding="utf-8").splitlines():
        url = raw.strip()
        if not url or url.startswith("#"):
            continue
        jobs.append(JobRecord(job_id=f"job_{index:03d}", url=url, company="", role="", location=""))
        index += 1
    return jobs


__all__ = ["extract_clean_job_text", "parse_urls_file"]
