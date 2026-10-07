"""TrainingSample: the one format for data sent to models for fine-tuning or evaluation.

A sample is a chat transcript (system / user / assistant) whose parts are text or *references* to
evidence images (never inline bytes), plus the structured target, the label, and full provenance.
Converting to a framework format (HF ``messages`` with ``{"type": "image"}`` parts, OpenAI-style
``image_url``) is a mechanical step done by the training script; the canonical file stays this one.

Rules enforced here (``docs/DATA_INTERCHANGE_STANDARD.md`` §4):

* the last message is the assistant target; the assistant never sees images;
* every image part is an :class:`ArtifactRef` pinned by SHA-256;
* every referenced artifact belongs to the sample's case (no cross-case mixing inside a sample);
* only VERIFIED values may appear in an assistant target, except in ``evidence_status_classification``
  where the target *is* the status label.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from src.schemas.common import SCHEMA_VERSION, ArtifactRef, Split


class TaskType(str, Enum):
    IMAGE_TEXT_GROUNDING = "image_text_grounding"      # image -> visible text that a database corroborates
    FINDING_GENERATION = "finding_generation"          # raw artifact fields -> citable Finding JSON
    EVIDENCE_STATUS = "evidence_status_classification"  # finding + evidence -> verified | rejected | uncorroborated


class ContentPart(BaseModel):
    type: Literal["text", "image"]
    text: Optional[str] = None
    image: Optional[ArtifactRef] = None

    @model_validator(mode="after")
    def _one_payload(self) -> "ContentPart":
        if self.type == "text" and (self.text is None or self.image is not None):
            raise ValueError("text part needs 'text' and no 'image'")
        if self.type == "image" and (self.image is None or self.text is not None):
            raise ValueError("image part needs 'image' and no 'text'")
        if self.type == "image" and not self.image.sha256:  # type: ignore[union-attr]
            raise ValueError("image parts must be pinned by sha256")
        return self


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: list[ContentPart]

    @classmethod
    def text(cls, role: str, text: str) -> "Message":
        return cls(role=role, content=[ContentPart(type="text", text=text)])  # type: ignore[arg-type]


class Provenance(BaseModel):
    case_id: str
    evidence_id: Optional[str] = None
    run_id: Optional[str] = None
    finding_ids: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list, description="every evidence file the sample uses")
    generator: str = Field(description="exporter that built the sample, e.g. 'finetune.image_text_grounding'")
    dataset: Optional[str] = Field(default=None, description="source export, e.g. data/datasets/20261006")
    created_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TrainingSample(BaseModel):
    schema_version: str = SCHEMA_VERSION
    sample_id: str = ""
    task: TaskType
    split: Split
    messages: list[Message]
    target: Any = Field(default=None, description="structured form of the assistant answer")
    label: Optional[str] = None
    target_status: Literal["verified", "rejected", "uncorroborated"] = Field(
        description="verification status of the content the target is built from")
    weight: float = 1.0
    provenance: Provenance

    @model_validator(mode="after")
    def _rules(self) -> "TrainingSample":
        if not self.messages or self.messages[-1].role != "assistant":
            raise ValueError("the last message must be the assistant target")
        for m in self.messages:
            if m.role == "assistant" and any(p.type == "image" for p in m.content):
                raise ValueError("assistant messages cannot contain images")
            for p in m.content:
                if p.type == "image" and p.image.case_id != self.provenance.case_id:  # type: ignore[union-attr]
                    raise ValueError("image from a different case than provenance.case_id")
        for a in self.provenance.artifacts:
            if a.case_id != self.provenance.case_id:
                raise ValueError("provenance.artifacts must all belong to provenance.case_id")
        if self.task is not TaskType.EVIDENCE_STATUS and self.target_status != "verified":
            raise ValueError(f"task {self.task.value} may only train on VERIFIED content")
        if self.task is TaskType.EVIDENCE_STATUS and self.label not in ("verified", "rejected", "uncorroborated"):
            raise ValueError("evidence_status_classification needs label verified|rejected|uncorroborated")
        if not self.sample_id:
            self.sample_id = self.content_hash()
        return self

    def content_hash(self) -> str:
        """Stable id from task + inputs + target (identical samples dedupe to one id)."""
        body = json.dumps([self.task.value, [m.model_dump(mode="json") for m in self.messages], self.target],
                          sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()[:20]

    def image_hashes(self) -> set[str]:
        return {p.image.sha256 for m in self.messages for p in m.content if p.type == "image" and p.image}  # type: ignore[misc]
