"""LLM client, endpoint, and routing helpers."""

import os
from dataclasses import dataclass
from enum import Enum

from openai import APIConnectionError, OpenAI


class ModelRole(str, Enum):
    DEFAULT = "default"
    METADATA = "metadata"
    REASONING = "reasoning"
    WRITER = "writer"
    INTERVIEW = "interview"
    CV = "cv"


@dataclass(frozen=True)
class LLMEndpoint:
    """One OpenAI-compatible model endpoint."""

    model: str
    api_key: str
    base_url: str | None = None
    api_mode: str = "responses"

    def client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key, base_url=self.base_url)


def _read_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _role_prefix(role: ModelRole | str) -> str:
    value = role.value if isinstance(role, ModelRole) else str(role)
    return value.upper().replace("-", "_")


class LLMRouter:
    """Routes pipeline stages to configurable OpenAI-compatible endpoints."""

    def __init__(self, endpoints: dict[ModelRole, LLMEndpoint]) -> None:
        self._endpoints = endpoints
        self._clients: dict[ModelRole, OpenAI] = {}

    @classmethod
    def from_env(cls) -> "LLMRouter":
        default_api_key = _read_env("LLM_API_KEY")
        if not default_api_key:
            raise RuntimeError("LLM generation is required. Set LLM_API_KEY before running this script.")

        default_model = _read_env("LLM_MODEL") or "gpt-4o-mini"
        default_base_url = _read_env("LLM_BASE_URL") or None
        default_api_mode = (_read_env("LLM_API_MODE") or "responses").lower()

        endpoints: dict[ModelRole, LLMEndpoint] = {}
        for role in ModelRole:
            prefix = _role_prefix(role)
            model = _read_env(f"LLM_{prefix}_MODEL") or default_model
            api_key = _read_env(f"LLM_{prefix}_API_KEY") or default_api_key
            base_url = _read_env(f"LLM_{prefix}_BASE_URL") or default_base_url
            api_mode = (_read_env(f"LLM_{prefix}_API_MODE") or default_api_mode).lower()
            endpoints[role] = LLMEndpoint(model=model, api_key=api_key, base_url=base_url, api_mode=api_mode)

        return cls(endpoints)

    def endpoint(self, role: ModelRole | str = ModelRole.DEFAULT) -> LLMEndpoint:
        normalized = ModelRole(role)
        return self._endpoints.get(normalized) or self._endpoints[ModelRole.DEFAULT]

    def client(self, role: ModelRole | str = ModelRole.DEFAULT) -> OpenAI:
        normalized = ModelRole(role)
        if normalized not in self._clients:
            self._clients[normalized] = self.endpoint(normalized).client()
        return self._clients[normalized]

    def generate(
        self,
        role: ModelRole | str,
        prompt: str,
        artifact_name: str,
        temperature: float = 0.2,
    ) -> str:
        content = safe_endpoint_generate(self.endpoint(role), prompt, temperature=temperature, client=self.client(role))
        if content is None or not content.strip():
            raise RuntimeError(f"LLM generation failed for {artifact_name}. Check API connectivity and model settings.")
        return content


def require_llm_router() -> LLMRouter:
    return LLMRouter.from_env()


def endpoint_generate(
    endpoint: LLMEndpoint,
    prompt: str,
    temperature: float = 0.2,
    client: OpenAI | None = None,
) -> str:
    system_instruction = (
        "You are a precise career assistant. Never fabricate experience. "
        "Optimize wording only based on provided resume facts. "
        "Avoid generic claims like 'strong fit', 'great fit', 'perfect fit', or 'ideal candidate'. "
        "Show fit through concrete, resume-backed evidence and distinctive strengths."
    )
    client = client or endpoint.client()
    if endpoint.api_mode == "chat":
        response = client.chat.completions.create(
            model=endpoint.model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()

    combined_input = f"{system_instruction}\n\n{prompt}"
    response = client.responses.create(
        model=endpoint.model,
        input=combined_input,
    )
    return (response.output_text or "").strip()


def safe_endpoint_generate(
    endpoint: LLMEndpoint,
    prompt: str,
    temperature: float = 0.2,
    client: OpenAI | None = None,
) -> str | None:
    try:
        return endpoint_generate(endpoint, prompt, temperature=temperature, client=client)
    except APIConnectionError:
        return None
    except Exception:
        return None


__all__ = [
    "LLMEndpoint",
    "LLMRouter",
    "ModelRole",
    "endpoint_generate",
    "require_llm_router",
    "safe_endpoint_generate",
]
