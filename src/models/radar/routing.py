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

Version 1 registries (`configs/radar/radar_routing_v1.yaml`) may also route Doppler stills: `still_structs` maps a
still's structure tag (from the sonographer's measurement label, `data/build_still_manifest.py`) to the structures it
may feed, and `still_types` names the still types (`cw_AV`, `pw_LVOT`, ...) that get an embedding. A structure fed
only by stills may have no views, and one with `caption_source: structured` (Hemo) takes a templated caption instead
of a report column. A v0 registry has neither key and behaves exactly as before.
"""

import hashlib
import os
from dataclasses import dataclass, field

import torch
import yaml

ROUTING_SCHEMA = "radar-structure-routing"
ROUTINGS = ("strict", "weak")
CAPTION_SOURCES = ("report", "structured")
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
    caption_source: str = "report"
    # a structured caption's fields and the still tags each rests on, in registry order: ((field, (tag, ...)), ...)
    caption_evidence: tuple = ()
    # a report caption's Doppler findings and the still tags that show each: ((finding, (tag, ...)), ...)
    finding_support: tuple = ()

    def supported_fields(self, tags) -> tuple[str, ...]:
        """The caption fields whose every still tag is among `tags` (the tags of the stills in a bag)."""
        tags = set(tags)
        return tuple(name for name, needed in self.caption_evidence if set(needed) <= tags)


    def supported_findings(self, tags) -> tuple[str, ...]:
        """The Doppler findings that at least one of `tags` (the tags of the stills in a bag) shows."""
        tags = set(tags)
        return tuple(name for name, showing in self.finding_support if set(showing) & tags)


def finding_scope(base, findings) -> str:
    """The caption scope of a report caption (`base`: full or doppler) whose Doppler clauses state only
    `findings`: `full:<finding>+<finding>`, sorted."""
    return f"{base}:" + "+".join(sorted(findings))


def evidence_scope(fields) -> str:
    """The caption scope of a structured caption that states exactly `fields`: `ev:<field>+<field>`, sorted."""
    return "ev:" + "+".join(sorted(fields))


@dataclass(frozen=True)
class Routing:
    version: int
    sha256: str
    view_classes: tuple[str, ...]
    structures: tuple[Structure, ...]
    view_families: dict[str, str] = field(default_factory=dict, hash=False)
    still_structs: dict[str, tuple[str, ...]] = field(default_factory=dict, hash=False)
    still_types: tuple[str, ...] = ()

    @property
    def still_tags(self) -> tuple[str, ...]:
        """The still structure tags, in registry order (the ids of `still_eligibility`)."""
        return tuple(self.still_structs)

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

    def still_tag_index(self, tag: str) -> int:
        """The id of a still structure tag; -1 for an unknown or missing tag (never eligible)."""
        try:
            return self.still_tags.index(tag)
        except ValueError:
            return -1

    def still_type_index(self, still_type: str) -> int:
        """The id of a still type; -1 for an unknown one (its embedding is the shared unknown row)."""
        try:
            return self.still_types.index(still_type)
        except ValueError:
            return -1

    def still_table(self, names=None) -> torch.Tensor:
        """[num_still_tags + 1, S] bool: row k says which structures a still of tag k may feed. The last row stands
        for an unknown tag or a padded slot and is all False."""
        names = [s.name for s in self.trained] if names is None else list(names)
        table = torch.zeros(len(self.still_tags) + 1, len(names), dtype=torch.bool)
        for k, tag in enumerate(self.still_tags):
            for j, name in enumerate(names):
                table[k, j] = name in self.still_structs[tag]
        return table

    def still_eligibility(self, tag_ids: torch.Tensor, still_valid: torch.Tensor, names=None) -> torch.Tensor:
        """[B, S, K] bool: structure s may pool still k. `tag_ids` [B, K] (-1 unknown or padded)."""
        table = self.still_table(names).to(tag_ids.device)
        ids = torch.where(tag_ids < 0, torch.full_like(tag_ids, len(self.still_tags)), tag_ids)
        return (table[ids] & still_valid[..., None]).permute(0, 2, 1)

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
    still_structs = raw.get("still_structs", {})
    fed_by_stills = set()
    if not isinstance(still_structs, dict):
        problems.append("still_structs must be a mapping of tag to structures")
    else:
        for tag, targets in still_structs.items():
            if not isinstance(tag, str) or not isinstance(targets, list) or len(set(targets)) != len(targets):
                problems.append(f"still_structs[{tag!r}] must be a list of distinct structure names")
                continue
            unknown = [t for t in targets if t not in names]
            if unknown:
                problems.append(f"still_structs[{tag!r}] names unknown structures {unknown}")
            fed_by_stills |= set(targets)
    still_types = raw.get("still_types", [])
    if not isinstance(still_types, list) or len(set(still_types)) != len(still_types):
        problems.append("still_types must be a list of distinct names")
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
        source = s.get("caption_source", "report")
        if source not in CAPTION_SOURCES:
            problems.append(f"{where}: caption_source must be one of {CAPTION_SOURCES}")
        evidence = s.get("caption_evidence")
        if evidence is not None:
            feeding = {tag for tag, targets in still_structs.items() if s.get("name") in (targets or [])} \
                if isinstance(still_structs, dict) else set()
            if source != "structured" or not isinstance(evidence, dict) or not evidence:
                problems.append(f"{where}: caption_evidence is a nonempty mapping, for caption_source: structured")
            elif any(not isinstance(tags, list) or not tags or set(tags) - feeding for tags in evidence.values()):
                problems.append(f"{where}: each caption_evidence field needs still tags that feed the structure")
        elif source == "structured" and s.get("train") is True:
            problems.append(f"{where}: a trained structured caption needs caption_evidence (which stills support "
                            "each field)")
        support = s.get("finding_support")
        if support is not None:
            feeding = {tag for tag, targets in still_structs.items() if s.get("name") in (targets or [])} \
                if isinstance(still_structs, dict) else set()
            if source != "report" or not isinstance(support, dict) or not support:
                problems.append(f"{where}: finding_support is a nonempty mapping, for caption_source: report")
            elif any(not isinstance(tags, list) or not tags or set(tags) - feeding for tags in support.values()):
                problems.append(f"{where}: each finding_support finding needs still tags that feed the structure")
        has_input = bool(own) or s.get("name") in fed_by_stills
        has_caption = bool(s.get("report_column")) or source == "structured"
        if s.get("train") is True and (not has_input or not has_caption):
            problems.append(f"{where}: a trained structure needs views and a report_column (a v1 registry: still "
                            "routing and caption_source: structured may stand in for them)")
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
                  views=tuple(s["views"]), routing=s["routing"], train=s["train"], note=s.get("note", ""),
                  caption_source=s.get("caption_source", "report"),
                  caption_evidence=tuple((str(k), tuple(v)) for k, v in (s.get("caption_evidence") or {}).items()),
                  finding_support=tuple((str(k), tuple(v)) for k, v in (s.get("finding_support") or {}).items()))
        for s in raw["structures"]
    )
    return Routing(version=raw["version"], sha256=hashlib.sha256(data).hexdigest(),
                   view_classes=tuple(raw["view_classes"]), structures=structures,
                   view_families=dict(raw.get("view_families", {})),
                   still_structs={str(k): tuple(v) for k, v in (raw.get("still_structs") or {}).items()},
                   still_types=tuple(raw.get("still_types", [])))
