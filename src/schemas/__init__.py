"""ForensiAgent Data Interchange (FDX) contracts.

One set of typed records shared by the forensic bridge, the agents, the dataset export and
fine-tuning. The human-readable specification is ``docs/DATA_INTERCHANGE_STANDARD.md``; the
field-level reference is ``docs/DATA_DICTIONARY.md``. ``python main.py schemas`` writes the JSON
Schemas of these models to ``docs/schemas/``.
"""
from src.schemas.bridge import BridgeRecord
from src.schemas.common import (SCHEMA_VERSION, ArtifactRef, EpochKind, EvidenceKind, FileCategory, ForensicTimestamp,
                                Platform, Split, to_timestamp)
from src.schemas.training import ContentPart, Message, Provenance, TaskType, TrainingSample

__all__ = [
    "SCHEMA_VERSION", "ArtifactRef", "BridgeRecord", "ContentPart", "EpochKind", "EvidenceKind", "FileCategory",
    "ForensicTimestamp", "Message", "Platform", "Provenance", "Split", "TaskType", "TrainingSample", "to_timestamp",
]
