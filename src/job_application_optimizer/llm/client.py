"""OpenAI client and generation helpers."""

import os

from openai import APIConnectionError, OpenAI


def maybe_get_openai_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    return OpenAI(api_key=api_key, base_url=base_url)


def require_openai_client() -> OpenAI:
    client = maybe_get_openai_client()
    if client is None:
        raise RuntimeError("LLM generation is required. Set OPENAI_API_KEY before running this script.")
    return client


def llm_generate(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float = 0.2,
) -> str:
    system_instruction = (
        "You are a precise career assistant. Never fabricate experience. "
        "Optimize wording only based on provided resume facts. "
        "Avoid generic claims like 'strong fit', 'great fit', 'perfect fit', or 'ideal candidate'. "
        "Show fit through concrete, resume-backed evidence and distinctive strengths."
    )
    combined_input = f"{system_instruction}\n\n{prompt}"
    response = client.responses.create(
        model=model,
        input=combined_input,
    )
    return (response.output_text or "").strip()


def safe_llm_generate(
    client: OpenAI | None,
    model: str,
    prompt: str,
    temperature: float = 0.2,
) -> str | None:
    if client is None:
        return None

    try:
        return llm_generate(client, model, prompt, temperature=temperature)
    except APIConnectionError:
        return None
    except Exception:
        return None


def require_llm_generate(
    client: OpenAI,
    model: str,
    prompt: str,
    artifact_name: str,
    temperature: float = 0.2,
) -> str:
    content = safe_llm_generate(client, model, prompt, temperature=temperature)
    if content is None or not content.strip():
        raise RuntimeError(f"LLM generation failed for {artifact_name}. Check API connectivity and model settings.")
    return content


__all__ = [
    "llm_generate",
    "maybe_get_openai_client",
    "require_llm_generate",
    "require_openai_client",
    "safe_llm_generate",
]
