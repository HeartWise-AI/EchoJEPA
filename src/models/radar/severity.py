# src/models/radar/severity.py

"""Severity weights of RADAR captions (`loss.severity_weights`, off by default), after DeepCORO_CLIP's positive
severity weights (`dataloaders/siglip_support.py`: normal < mild < moderate < severe).

Each caption gets one grade, from the caption table's structured columns, never from its text:

- `normality` normal -> `normal`, unknown -> `unknown`;
- `normality` abnormal -> the highest grade named by the categorical values of its `sources` column (the merged
  builder's `[[column, value, source], ...]`, e.g. `mildly_reduced`, `moderately_reduced`, `severely_reduced`,
  `akinetic_or_severely_hypokinetic`): a value token `severe`/`severely` is `severe`, `moderate`/`moderately`
  `moderate`, `mild`/`mildly` `mild` (`mild_to_moderate` is therefore `moderate`). An abnormal caption without a
  graded value (no `sources` column, as in the v0 builder, or only ungraded values such as `dilated`) is `abnormal`.

The weight of a grade comes from `DEFAULT_WEIGHTS`, updated by the config's mapping. The loss looks weights up by
caption id (`CaptionWeights`), so every rank derives them from the gathered ids. One caption id with two grades (the
same text built from different sources) takes the larger weight; `coverage` counts those ids.
"""

import json
import re

import torch

GRADES = ("normal", "mild", "moderate", "severe", "abnormal", "unknown")
DEFAULT_WEIGHTS = {"normal": 1.0, "mild": 1.5, "moderate": 2.5, "severe": 3.0, "abnormal": 1.5, "unknown": 1.0}
_RANK = {"mild": 1, "moderate": 2, "severe": 3}
_TOKEN_GRADE = {"mild": "mild", "mildly": "mild", "moderate": "moderate", "moderately": "moderate",
                "severe": "severe", "severely": "severe"}


def resolve_weights(config):
    """The grade -> weight map of `loss.severity_weights`: true (the defaults) or a mapping that updates them."""
    if config is True:
        return dict(DEFAULT_WEIGHTS)
    if not isinstance(config, dict):
        raise TypeError("loss.severity_weights must be true or a mapping of grade -> weight.")
    unknown = sorted(set(config) - set(GRADES))
    if unknown:
        raise ValueError(f"loss.severity_weights names unknown grades {unknown}; grades are {GRADES}.")
    weights = {**DEFAULT_WEIGHTS, **{k: float(v) for k, v in config.items()}}
    if any(not w > 0 for w in weights.values()):
        raise ValueError("loss.severity_weights must be positive.")
    return weights


def value_grade(value):
    """`mild`, `moderate`, `severe` or None for one categorical source value (e.g. `moderately_reduced`)."""
    grades = [_TOKEN_GRADE[t] for t in re.split(r"[^a-z]+", str(value).lower()) if t in _TOKEN_GRADE]
    return max(grades, key=_RANK.get) if grades else None


def caption_grade(normality, sources=None):
    """The grade of one caption row (see the module docstring). `sources`: the JSON of the `sources` column."""
    if normality == "normal":
        return "normal"
    if normality != "abnormal":
        return "unknown"
    graded = []
    if isinstance(sources, str) and sources.strip():
        for entry in json.loads(sources):
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                grade = value_grade(entry[1])
                if grade:
                    graded.append(grade)
    return max(graded, key=_RANK.get) if graded else "abnormal"


def caption_weights(captions, config):
    """({caption_id: weight}, coverage) for a caption table (columns structure, caption_id, normality and, when
    present, sources). Coverage: rows per grade and structure, the weights used and the ids with two grades; no
    text."""
    weights = resolve_weights(config)
    has_sources = "sources" in captions.columns
    out, grade_of, conflicts = {}, {}, set()
    rows = {}
    for r in captions.itertuples():
        grade = caption_grade(r.normality, getattr(r, "sources", None) if has_sources else None)
        rows.setdefault(r.structure, {g: 0 for g in GRADES})[grade] += 1
        cid = int(r.caption_id)
        if cid in grade_of and grade_of[cid] != grade:
            conflicts.add(cid)
        if weights[grade] >= out.get(cid, 0.0):
            out[cid], grade_of[cid] = weights[grade], grade
    return out, {"weights": weights, "graded_from_sources": has_sources, "rows": rows,
                 "caption_ids_with_two_grades": len(conflicts)}


class CaptionWeights:
    """A caption id -> weight lookup on any device; ids not in the table get `default`."""

    def __init__(self, weights, default=1.0):
        ids = sorted(weights)
        self.default = float(default)
        self._table = (torch.tensor(ids, dtype=torch.int64),
                       torch.tensor([weights[i] for i in ids], dtype=torch.float32))
        self._device = None

    def __call__(self, caption_ids):
        if self._device is None or self._device[0].device != caption_ids.device:
            self._device = tuple(x.to(caption_ids.device) for x in self._table)
        keys, values = self._device
        out = torch.full(caption_ids.shape, self.default, dtype=torch.float32, device=caption_ids.device)
        if keys.numel() == 0:
            return out
        caption_ids = caption_ids.contiguous().to(torch.int64)
        pos = torch.searchsorted(keys, caption_ids).clamp(max=keys.numel() - 1)
        return torch.where(keys[pos] == caption_ids, values[pos], out)
