# src/models/radar/routing.py

"""The RADAR structure routing registry (`configs/radar/radar_routing_v0.yaml`).

For each cardiac structure the registry lists the view classes in which it may be visible, the report findings
column that gives its caption, how the view set was obtained (`strict`: the source specification's level-free
views; `weak`: a named override, reported as exploratory) and whether a query is trained for it.

Four flags are kept apart, per study and structure:

- `has_eligible_clip`: the study's clips include one whose view is in the structure's views;
- `has_training_caption`: the structure's caption is present and allowed (set by the caption builder);
- `contrastive_trained`: the structure has a trained query (`Structure.train`);
- `prediction_supported`: `contrastive_trained` and `has_eligible_clip`; otherwise the prediction is missing.

A clip of an eligible view never gives a prediction to a structure without a trained query.

The registry also holds `view_families`, which only steers which clips enter a study's bag (see
`src/datasets/radar_study_dataset.py`) and plays no part in eligibility.
"""

import hashlib
import os
from dataclasses import dataclass, field

import torch
import yaml

ROUTING_SCHEMA = "radar-structure-routing"
ROUTINGS = ("strict", "weak")
UNCLASSIFIED_VIEW = "OTHER"
DEFAULT_ROUTING = os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", "radar",
                               "radar_routing_v0.yaml")


@dataclass(frozen=True)
class Structure:
    name: str
    label: str
    report_column: str | None
    views: tuple[str, ...]
    routing: str
    train: bool
    note: str = ""


@dataclass(frozen=True)
class Routing:
    version: int
    sha256: str
    view_classes: tuple[str, ...]
    structures: tuple[Structure, ...]
    view_families: dict[str, str] = field(default_factory=dict, hash=False)

    @property
    def trained(self) -> tuple[Structure, ...]:
        """The structures with a trained query, in registry order (the order of the model's queries)."""
        return tuple(s for s in self.structures if s.train)

    def structure(self, name: str) -> Structure:
        for s in self.structures:
            if s.name == name:
                return s
        raise KeyError(f"No structure {name!r} in the routing registry.")

    def view_index(self, view: str) -> int:
        """The class id of a view name; -1 for an unknown or missing view (never eligible)."""
        try:
            return self.view_classes.index(view)
        except ValueError:
            return -1

    def eligibility(self, names=None) -> torch.Tensor:
        """[num_view_classes + 1, S] bool: row v says which structures a clip of view class v may show. The extra
        last row stands for an unknown view or a padded slot (view id -1) and is all False. `names` defaults to the
        trained structures."""
        names = [s.name for s in self.trained] if names is None else list(names)
        table = torch.zeros(len(self.view_classes) + 1, len(names), dtype=torch.bool)
        for j, name in enumerate(names):
            for view in self.structure(name).views:
                table[self.view_classes.index(view), j] = True
        return table

    def clip_eligibility(self, view_ids: torch.Tensor, clip_valid: torch.Tensor, names=None) -> torch.Tensor:
        """[B, S, C] bool: structure s may pool clip c. `view_ids` [B, C] holds class ids (-1 for unknown or
        padded), `clip_valid` [B, C] is False for padded slots."""
        table = self.eligibility(names).to(view_ids.device)
        ids = torch.where(view_ids < 0, torch.full_like(view_ids, len(self.view_classes)), view_ids)
        return (table[ids] & clip_valid[..., None]).permute(0, 2, 1)


def _problems(raw) -> list[str]:
    problems = []
    if not isinstance(raw, dict):
        return ["the registry is not a mapping"]
    if raw.get("schema") != ROUTING_SCHEMA:
        problems.append(f"schema must be {ROUTING_SCHEMA!r}")
    if not isinstance(raw.get("version"), int) or isinstance(raw.get("version"), bool):
        problems.append("version must be an integer")
    views = raw.get("view_classes")
    if not isinstance(views, list) or not views or len(set(views)) != len(views):
        problems.append("view_classes must be a nonempty list of distinct names")
        views = []
    families = raw.get("view_families", {})
    if not isinstance(families, dict):
        problems.append("view_families must be a mapping of view to family")
    else:
        unknown = sorted({str(v) for pair in families.items() for v in pair} - set(views))
        if unknown:
            problems.append(f"view_families names unknown views {unknown}")
        chained = sorted(set(families) & set(families.values()))
        if chained:
            problems.append(f"view_families maps families to other families: {chained}")
    structures = raw.get("structures")
    if not isinstance(structures, list) or not structures:
        return problems + ["structures must be a nonempty list"]
    names = [s.get("name") if isinstance(s, dict) else None for s in structures]
    if len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
        problems.append("structure names must be distinct nonempty strings")
    for s in structures:
        if not isinstance(s, dict):
            problems.append("every structure must be a mapping")
            continue
        where = f"structure {s.get('name')!r}"
        if s.get("routing") not in ROUTINGS:
            problems.append(f"{where}: routing must be one of {ROUTINGS}")
        if not isinstance(s.get("train"), bool):
            problems.append(f"{where}: train must be true or false")
        own = s.get("views")
        if not isinstance(own, list) or len(set(own)) != len(own):
            problems.append(f"{where}: views must be a list of distinct names")
            continue
        unknown = [v for v in own if v not in views]
        if unknown:
            problems.append(f"{where}: unknown views {unknown}")
        if UNCLASSIFIED_VIEW in own:
            problems.append(f"{where}: {UNCLASSIFIED_VIEW} carries no structure and cannot be routed")
        if s.get("train") is True and (not own or not s.get("report_column")):
            problems.append(f"{where}: a trained structure needs views and a report_column")
    return problems


def load_routing(path: str = DEFAULT_ROUTING) -> Routing:
    """Read and validate a routing registry. Raises ValueError listing every problem."""
    with open(path, "rb") as f:
        data = f.read()
    raw = yaml.safe_load(data)
    problems = _problems(raw)
    if problems:
        raise ValueError(f"Invalid routing registry {path}: " + "; ".join(problems))
    structures = tuple(
        Structure(name=s["name"], label=s.get("label", s["name"]), report_column=s.get("report_column"),
                  views=tuple(s["views"]), routing=s["routing"], train=s["train"], note=s.get("note", ""))
        for s in raw["structures"]
    )
    return Routing(version=raw["version"], sha256=hashlib.sha256(data).hexdigest(),
                   view_classes=tuple(raw["view_classes"]), structures=structures,
                   view_families=dict(raw.get("view_families", {})))
