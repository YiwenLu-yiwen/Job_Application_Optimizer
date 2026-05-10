"""Runtime configuration helpers."""

import os


def _read_int_env(name: str, default: int) -> int:
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value


# 0 means no truncation (full text included in prompts).
PROMPT_CHAR_LIMIT = _read_int_env("PROMPT_CHAR_LIMIT", 0)


def _prompt_text(text: str) -> str:
    if PROMPT_CHAR_LIMIT <= 0:
        return text
    return text[:PROMPT_CHAR_LIMIT]


__all__ = ["PROMPT_CHAR_LIMIT", "_prompt_text", "_read_int_env"]
