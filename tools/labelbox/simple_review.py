"""Build a small proposal form; no Labelbox resources or clinical uploads are created.

Every row supplies at most five numbered, structured candidate claims as review
context. Slot questions are generic so one form can review all twenty view classes.
"""

import json
from pathlib import Path

from app.radar.ontology import VIEWS
from tools.labelbox.review_ontology import _classification, _option

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "configs/radar/simple_review_proposal_v1.json"
FORM = ROOT / "configs/radar/labelbox/simple_review_proposal_v1.json"


def build_form():
    answers = [
        _option("yes", "Yes, supported"),
        _option("no", "No, wrong or unsupported"),
        _option("cannot_assess", "Cannot assess"),
    ]
    questions = [
        _classification(
            "radio",
            "view",
            "What is the view?",
            required=True,
            options=[_option(view, view) for view in VIEWS],
            top_level=True,
        ),
        _classification(
            "radio",
            "modality_ok",
            "Is the acquisition mode correct?",
            required=True,
            options=answers,
            top_level=True,
        ),
    ]
    for slot in range(1, 6):
        questions.append(
            _classification(
                "radio",
                f"claim_{slot}_ok",
                f"Is claim {slot} supported in the model input?",
                required=True,
                options=answers + [_option("no_claim", "No claim in this slot")],
                top_level=True,
            )
        )
    questions.append(
        _classification(
            "radio",
            "important_missing",
            "Is important information missing?",
            required=True,
            options=[
                _option("no", "No"),
                _option(
                    "yes",
                    "Yes",
                    [_classification("text", "missing_detail", "Specify the missing information", required=True)],
                ),
            ],
            top_level=True,
        )
    )
    questions.append(
        _classification(
            "text",
            "optional_correction",
            "Optional correction (requires adjudication)",
            required=False,
            top_level=True,
        )
    )
    return {"tools": [], "classifications": questions}


def score_review(candidate, answers):
    """Score submitted answers against the original slot claims, never against report context.

    A rejected claim is dropped, not converted to a clinical negative. Free-text
    corrections stay outside the verified dataset until separately adjudicated.
    """
    claims = candidate.get("claims", [])
    if len(claims) > 5:
        raise ValueError("At most five claims per item")
    spec = json.loads(SPEC.read_text())
    for claim in claims:
        if not isinstance(claim, dict) or claim.get("concept") not in spec["fields"]:
            raise ValueError("Every claim needs a known concept")
        if claim.get("value") not in spec["fields"][claim["concept"]]["values"]:
            raise ValueError("Every claim needs an allowed value")
    if candidate.get("view") not in spec["views"]:
        raise ValueError("Candidate has an invalid proposed view")
    allowed = {"yes", "no", "cannot_assess"}
    reviewed_view = answers.get("view")
    if reviewed_view not in spec["views"]:
        raise ValueError("Missing or invalid view")
    if answers.get("modality_ok") not in allowed:
        raise ValueError("Missing or invalid modality_ok")
    accepted, rejected, unresolved = [], [], []
    for i in range(5):
        answer = answers.get(f"claim_{i + 1}_ok")
        if i >= len(claims):
            if answer != "no_claim":
                raise ValueError("Unused slots require no_claim")
            continue
        if answer not in allowed:
            raise ValueError("A populated slot needs yes, no or cannot_assess")
        {"yes": accepted, "no": rejected, "cannot_assess": unresolved}[answer].append(claims[i])
    if answers.get("important_missing") not in {"yes", "no"}:
        raise ValueError("Missing or invalid important_missing")
    if answers["important_missing"] == "yes" and not str(answers.get("missing_detail", "")).strip():
        raise ValueError("Missing information requires a detail")
    has_provenance = candidate.get("input_provenance_verified") is True
    route = spec["views"][reviewed_view]
    acquisition = candidate.get("modality")
    input_ok = (
        has_provenance
        and acquisition in {"bmode", "color_flow"}
        and answers["modality_ok"] == "yes"
    )
    allowed_claims = (
        route["bmode_priority"] + route.get("bmode_optional", [])
        if acquisition == "bmode"
        else route["color_flow_priority"]
    )
    if any(claim["concept"] not in allowed_claims for claim in accepted):
        input_ok = False
    if route.get("requires_target_confirmation") and candidate.get("target_confirmed") is not True:
        input_ok = False
    if reviewed_view == "SUBCOSTAL":
        target = candidate.get("target")
        if target not in {"ivc", "four_chamber"}:
            input_ok = False
        for claim in accepted:
            ivc_claim = claim["concept"] in {"ivc_size", "ivc_collapse"}
            if ivc_claim != (target == "ivc"):
                input_ok = False
    for claim in accepted:
        required = spec["fields"][claim["concept"]].get("required_modality")
        if required and required != acquisition:
            input_ok = False
    if reviewed_view == "OTHER":
        input_ok = False
    # Keep supported claims as a partial caption only after text/claims are rebuilt
    # and verified. Neither report EF nor rejected claims enter this list.
    eligible = (
        bool(accepted) and input_ok and answers["important_missing"] == "no" and not answers.get("optional_correction")
    )
    return {
        "proposed_view": candidate.get("view"),
        "reviewed_view": reviewed_view,
        "view_corrected": reviewed_view != candidate.get("view"),
        "supported": accepted,
        "rejected": rejected,
        "cannot_assess": unresolved,
        "candidate_claims": len(claims),
        "supported_count": len(accepted),
        "resolved_count": len(accepted) + len(rejected),
        "all_supported": bool(claims) and len(accepted) == len(claims) and input_ok,
        "partial_caption_ready_for_validation": eligible,
    }


if __name__ == "__main__":
    FORM.write_text(json.dumps(build_form(), ensure_ascii=False, indent=2) + "\n")
