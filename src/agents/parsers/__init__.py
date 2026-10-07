"""Artifact parsers used by the AppArtifactAgent (one module per family; see ``base.py``)."""
from src.agents.parsers import android, browsers, ios, messengers
from src.agents.parsers.base import Col, Lookup, ParsedRecord, TableSpec, detect, parse_table

ALL_SPECS: list[TableSpec] = [*android.SPECS, *ios.SPECS, *browsers.SPECS, *messengers.SPECS]

__all__ = ["ALL_SPECS", "Col", "Lookup", "ParsedRecord", "TableSpec", "detect", "parse_table"]
