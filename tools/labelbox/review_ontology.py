"""Building blocks of the Labelbox review forms: the column review (`column_review.py`) and the view and type
review (`view_type_review.py`).

Each form is laid out exactly as the Labelbox SDK's `OntologyBuilder.asdict()` writes it, so a round trip through
`OntologyBuilder.from_dict` changes nothing (each form's tests check it when the SDK is installed). Nothing here talks
to Labelbox.
"""

from __future__ import annotations


def _classification(kind: str, name: str, instructions: str, *, required: bool, options=(), top_level=False) -> dict:
    """One classification, with the SDK's keys in the SDK's order (nested ones have no scope)."""
    out = {"type": kind, "instructions": instructions, "name": name, "required": required, "options": list(options),
           "schemaNodeId": None, "featureSchemaId": None, "attributes": None}
    if top_level:
        out["scope"] = "global"
    return out


def _option(value: str, label: str, nested=()) -> dict:
    return {"schemaNodeId": None, "featureSchemaId": None, "label": label, "value": value, "options": list(nested)}
