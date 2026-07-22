"""Validated structured-output generation with corrective retries."""

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from job_application_optimizer.llm.client import LLMRouter, ModelRole
from job_application_optimizer.llm.json_parser import parse_llm_json


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """Raised when an LLM cannot produce a valid structured response."""


def _validation_feedback(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return json.dumps(exc.errors(include_url=False, include_input=False), ensure_ascii=False)
    return str(exc)


def generate_structured(
    llm: LLMRouter,
    role: ModelRole,
    prompt: str,
    artifact_name: str,
    schema: type[SchemaT],
    temperature: float = 0.2,
    max_attempts: int = 3,
) -> SchemaT:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    current_prompt = prompt
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        content = llm.generate(role, current_prompt, artifact_name, temperature=temperature)
        try:
            return schema.model_validate(parse_llm_json(content))
        except (ValueError, ValidationError) as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            feedback = _validation_feedback(exc)
            current_prompt = (
                f"{prompt}\n\n"
                "Your previous response failed JSON schema validation. Return one corrected JSON object only. "
                "Do not add fields or include markdown fences.\n\n"
                f"Validation errors:\n{feedback}\n\n"
                f"Previous invalid response:\n{content[:4000]}"
            )

    raise StructuredOutputError(
        f"LLM generation failed schema validation for {artifact_name} after {max_attempts} attempts: "
        f"{_validation_feedback(last_error or ValueError('unknown validation error'))}"
    ) from last_error


__all__ = ["StructuredOutputError", "generate_structured"]
