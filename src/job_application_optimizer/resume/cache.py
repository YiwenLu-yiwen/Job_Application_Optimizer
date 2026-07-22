"""Resume-derived cache markers and freshness checks."""

import hashlib
import re
from pathlib import Path
from typing import Literal


CACHE_MARKER = "job-application-optimizer-cache: resume_sha256="
_MARKER_RE = re.compile(
    rf"^(?:<!--\s*|#\s*){re.escape(CACHE_MARKER)}([0-9a-f]{{64}})(?:\s*-->)?\s*\n?"
)


def resume_sha256(resume_path: Path) -> str:
    digest = hashlib.sha256()
    with resume_path.open("rb") as resume_file:
        for chunk in iter(lambda: resume_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_fresh_cache(cache_path: Path, resume_hash: str) -> str | None:
    if not cache_path.exists():
        return None
    cached = cache_path.read_text(encoding="utf-8")
    marker = _MARKER_RE.match(cached)
    if marker is None or marker.group(1) != resume_hash:
        return None
    return cached[marker.end() :]


def write_cache(
    cache_path: Path,
    content: str,
    resume_hash: str,
    style: Literal["markdown", "yaml"],
) -> None:
    if style == "markdown":
        marker = f"<!-- {CACHE_MARKER}{resume_hash} -->\n"
    else:
        marker = f"# {CACHE_MARKER}{resume_hash}\n"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(f"{marker}{content}", encoding="utf-8")


__all__ = ["read_fresh_cache", "resume_sha256", "write_cache"]
