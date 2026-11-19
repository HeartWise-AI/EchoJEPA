# src/models/radar/view_prefix.py

"""Captions with a view prefix (`data.view_prefix: true`; off by default).

RADAR's published captions start with the view the image was taken in. Here a caption belongs to a (study,
structure) and the structure's embedding is pooled over every clip of the study's bag that the routing registry lets
it read, so the prefix names the views that feed the structure in that bag: the distinct phrases of their view
families, in the fixed order of the phrase file (`configs/radar/view_phrases_v0.yaml`), then the caption:

    "Apical four-chamber view. <caption>"
    "Parasternal long-axis and apical four-chamber views. <caption>"
    "Parasternal long-axis, apical four-chamber and apical two-chamber views. <caption>"

A clip of an unphrased view (OTHER) adds nothing. A structure that no clip feeds gets no prefix (it is never trained
or scored in that bag). The unit of the model is unchanged: one embedding and one caption per (study, structure).

Exact-caption positives (A1): the prefixed row's caption id is derived from the prefix and the builder's caption id
(`prefixed_caption_id`), so two rows are the same caption only when both the views and the caption match.

A frozen caption encoder needs every caption it will meet encoded before the first step. `possible_prefixes` lists
the prefixes a study's bags can give a structure: with view-balanced bags and no more view families than clips per bag
(or no more clips than the budget), every family of the study is in every bag, and only a family whose clips are not
all readable by the structure (A2C_LV is an LV view, not an LA one) may or may not feed it; otherwise every family
that can feed it may be absent. The list holds every prefix that a bag can give (more only when a mixed family always
gets more than one slot), so a bag never meets an unencoded caption.
"""

import hashlib
import itertools
import json
import os

import yaml

DEFAULT_VIEW_PHRASES = os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", "radar",
                                    "view_phrases_v0.yaml")
SCHEMA = "radar-view-phrases"
MAX_OPTIONAL_FAMILIES = 12          # 2 ** 12 prefixes for one (study, structure) at most; refuse beyond


class ViewPhrases:
    """The view phrase file, checked against a routing registry: view class -> phrase, and prefix texts."""

    def __init__(self, path, routing):
        with open(path, "rb") as f:
            data = f.read()
        raw = yaml.safe_load(data)
        if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
            raise ValueError(f"{path}: schema must be {SCHEMA!r}.")
        families = raw.get("families")
        unphrased = list(raw.get("unphrased") or [])
        if not isinstance(families, dict) or not families:
            raise ValueError(f"{path}: families must be a nonempty mapping of view family to phrase.")
        bad = [k for k, v in families.items() if not isinstance(v, str) or not v.strip() or v != v.strip()]
        if bad:
            raise ValueError(f"{path}: phrases of {bad} must be nonempty text without surrounding spaces.")
        if len(set(families.values())) != len(families):
            raise ValueError(f"{path}: two families share a phrase.")
        registry = sorted({routing.view_families.get(v, v) for v in routing.view_classes})
        listed = set(families) | set(unphrased)
        problems = []
        if set(families) & set(unphrased):
            problems.append(f"both phrased and unphrased: {sorted(set(families) & set(unphrased))}")
        if sorted(listed) != registry:
            missing, extra = sorted(set(registry) - listed), sorted(listed - set(registry))
            problems.append(f"families missing {missing}, not families of the registry {extra}")
        if problems:
            raise ValueError(f"{path} does not match the routing registry's view families: {'; '.join(problems)}.")
        content = {"families": [[k, v] for k, v in families.items()], "unphrased": sorted(unphrased)}
        self.sha256 = hashlib.sha256(json.dumps(content).encode("utf-8")).hexdigest()   # comments do not count
        self.version = raw.get("version")
        self.family = {v: routing.view_families.get(v, v) for v in routing.view_classes}
        self.phrases = {v: families.get(self.family[v]) for v in routing.view_classes}   # None: unphrased
        self.rank = {phrase: i for i, phrase in enumerate(families.values())}    # the fixed order of a prefix

    def record(self):
        """What the run's fingerprint keeps: the digest of the phrases, in their order, and the unphrased families."""
        return {"phrases_sha256": self.sha256, "version": self.version}

    def phrase(self, view):
        """The phrase of a view class; None for an unphrased or unknown view."""
        return self.phrases.get(view)

    def join(self, phrases):
        """The prefix text of a set of phrases, in the fixed order; "" for none."""
        ordered = sorted(set(phrases), key=self.rank.__getitem__)
        if not ordered:
            return ""
        text = f"{ordered[0]} view." if len(ordered) == 1 else f"{', '.join(ordered[:-1])} and {ordered[-1]} views."
        return text[0].upper() + text[1:]

    def prefix(self, views):
        """The prefix naming `views` (view classes of the clips feeding a structure, any order, repeats allowed);
        "" when none has a phrase."""
        return self.join({self.phrase(v) for v in views} - {None})

    def possible_prefixes(self, clip_views, structure_views, num_clips, selection):
        """Every prefix a bag of a study can give a structure (see the module docstring), sorted; "" (no feeding
        clip) left out. `clip_views`: the view class of each of the study's clips; `structure_views`: the views the
        registry lets the structure read."""
        readable = set(structure_views)
        members = {}
        for view in clip_views:
            members.setdefault(self.family.get(view, view), []).append(view)
        balanced = selection == "view_balanced" and len(members) <= num_clips
        every_family_in_bag = len(clip_views) <= num_clips or balanced
        sure, optional = set(), set()
        for views in members.values():
            phrase = self.phrase(views[0])
            fed = [v in readable for v in views]
            if phrase is None or not any(fed):
                continue
            (sure if every_family_in_bag and all(fed) else optional).add(phrase)
        optional = sorted(optional - sure, key=self.rank.__getitem__)
        if len(optional) > MAX_OPTIONAL_FAMILIES:
            raise ValueError(f"{len(optional)} optional view families: too many prefixes to enumerate.")
        out = {self.join(sure | set(chosen)) for k in range(len(optional) + 1)
               for chosen in itertools.combinations(optional, k)}
        return sorted(out - {""})


def prefixed(prefix, text):
    """The caption or prompt `text` after `prefix` ("" leaves it unchanged)."""
    return f"{prefix} {text}" if prefix else text


def prefixed_caption_id(caption_id, prefix):
    """A signed 64-bit id of (prefix, caption id): equal only when both are equal. "" keeps the caption id."""
    if not prefix:
        return int(caption_id)
    digest = hashlib.blake2b(f"{prefix}\0{int(caption_id)}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


def view_phrases(data, routing):
    """The ViewPhrases of a run's `data` config section (`view_prefix`, `view_phrases`: the phrase file, default
    `configs/radar/view_phrases_v0.yaml`), or None when `view_prefix` is off or absent."""
    on = data.get("view_prefix", False)
    if not isinstance(on, bool):
        raise ValueError("data.view_prefix must be true or false.")
    if not on:
        if data.get("view_phrases"):
            raise ValueError("data.view_phrases is set but data.view_prefix is off.")
        return None
    if data.get("stills"):
        raise ValueError("data.view_prefix names the views of a bag's clips; it is defined for clip-only runs.")
    return ViewPhrases(data.get("view_phrases") or DEFAULT_VIEW_PHRASES, routing)


def longest_prefixes(phrases, routing, names):
    """{structure: the prefix naming every view it may read}, the longest it can get (caption length checks)."""
    return {name: phrases.prefix(routing.structure(name).views) for name in names}
