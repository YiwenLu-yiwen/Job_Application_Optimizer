import json
from pathlib import Path

from job_application_optimizer.resume.capability_inventory import (
    load_or_generate_capability_inventory,
)
from job_application_optimizer.resume.understanding import (
    load_or_generate_cv_understanding,
)


class RecordingLLM:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *args: object, **kwargs: object) -> str:
        self.calls += 1
        return f"generated-content-{self.calls}"


class RecordingCapabilityLLM(RecordingLLM):
    def generate(self, *args: object, **kwargs: object) -> str:
        self.calls += 1
        return json.dumps(
            {
                "version": 1,
                "status": "draft",
                "instructions": "Review manually before use.",
                "capabilities": [
                    {
                        "name": f"generated-content-{self.calls}",
                        "category": "technical_delivery",
                        "evidence": [],
                        "metrics": [],
                        "tools_used": [],
                        "exact_terms_allowed": [],
                        "equivalent_terms_allowed": [],
                        "exposure_terms": [],
                        "forbidden_terms": [],
                        "safe_sentence_templates": [],
                        "confidence": "medium",
                        "verification_status": "needs_review",
                        "notes": "",
                    }
                ],
            }
        )


def test_cv_understanding_cache_is_reused_until_resume_changes(tmp_path: Path) -> None:
    resume_path = tmp_path / "resume.txt"
    cache_path = tmp_path / "cv_deep_understanding.md"
    resume_path.write_text("first resume", encoding="utf-8")
    cache_path.write_text("legacy cache without a resume hash", encoding="utf-8")
    llm = RecordingLLM()

    first = load_or_generate_cv_understanding(resume_path, llm, cache_path)
    second = load_or_generate_cv_understanding(resume_path, llm, cache_path)

    assert first == "generated-content-1"
    assert second == first
    assert llm.calls == 1

    resume_path.write_text("updated resume", encoding="utf-8")

    refreshed = load_or_generate_cv_understanding(resume_path, llm, cache_path)

    assert refreshed == "generated-content-2"
    assert llm.calls == 2
    assert "generated-content-2" in cache_path.read_text(encoding="utf-8")


def test_stale_verified_inventory_is_preserved_and_replaced_by_a_fresh_draft(tmp_path: Path) -> None:
    resume_path = tmp_path / "resume.txt"
    inventory_path = tmp_path / "capability_inventory.yaml"
    resume_path.write_text("first resume", encoding="utf-8")
    llm = RecordingCapabilityLLM()

    initial = load_or_generate_capability_inventory(
        resume_path,
        "cv understanding",
        inventory_path,
        llm,
    )
    inventory_path.write_text(initial.draft_path.read_text(encoding="utf-8"), encoding="utf-8")

    verified = load_or_generate_capability_inventory(
        resume_path,
        "cv understanding",
        inventory_path,
        llm,
    )

    assert verified.usable is True
    assert json.loads(verified.content)["capabilities"][0]["name"] == "generated-content-1"
    assert llm.calls == 1

    resume_path.write_text("updated resume", encoding="utf-8")

    refreshed = load_or_generate_capability_inventory(
        resume_path,
        "updated cv understanding",
        inventory_path,
        llm,
    )

    assert refreshed.usable is False
    assert refreshed.generated_draft is True
    assert llm.calls == 2
    assert inventory_path.exists()
    assert "generated-content-1" in inventory_path.read_text(encoding="utf-8")
    assert "generated-content-2" in refreshed.draft_path.read_text(encoding="utf-8")
