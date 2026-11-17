# evals/view_classifier/classes.py

"""The view classes: the current classifier's 20 (`OLD_CLASSES`, in its output order) and the 23 the human labels use
(`NEW_CLASSES`, the options of the view review form `configs/radar/labelbox/view_review_v1.json` without `unsure`).

Every new class has one parent among the old classes (`PARENT`): itself when the name is kept, `PSAX` for the three
PSAX levels and `PLAX_ZOOM` for `PLAX_ZOOM_AV`. Collapsing a new label to its parent compares a new prediction with
the old classifier at the old granularity.

An old class can give pseudo-labels only when exactly one new class descends from it (`pseudo_label_classes`): the old
classifier's `PSAX` cannot say which level, and its `PLAX_ZOOM` cannot tell a zoom on the aortic valve from another one.

The label project's live form has six more options than `view_review_v1.json` (`FORM_EXTRAS`, with the parent each is
given here). A run's class list and parents come from its config (`from_config`): a `classes` mapping
{new class: old parent}, in output order; without one, `NEW_CLASSES` and `PARENT`.
"""

import json

OLD_CLASSES = (
    "PSAX", "OTHER", "A4C", "PLAX_ZOOM", "A5C", "PLAX", "A2C", "A4C_ZOOM", "A3C_LV", "PLAX_DEEP",
    "SUBCOSTAL", "A3C", "RVINF", "A2C_LV", "A4C_LV", "A5C_ZOOM", "SUPRASTERNAL", "A3C_ZOOM", "A2C_ZOOM", "PSAX_AV",
)

NEW_CLASSES = (
    "A2C", "A2C_LV", "A2C_ZOOM", "A3C", "A3C_LV", "A3C_ZOOM", "A4C", "A4C_LV", "A4C_ZOOM", "A5C", "A5C_ZOOM",
    "PLAX", "PLAX_DEEP", "PLAX_ZOOM", "PLAX_ZOOM_AV", "PSAX_BASAL", "PSAX_MID", "PSAX_APICAL", "PSAX_AV",
    "RVINF", "SUBCOSTAL", "SUPRASTERNAL", "OTHER",
)

SPLIT_CLASSES = {"PSAX_BASAL": "PSAX", "PSAX_MID": "PSAX", "PSAX_APICAL": "PSAX", "PLAX_ZOOM_AV": "PLAX_ZOOM"}
PARENT = {c: SPLIT_CLASSES.get(c, c) for c in NEW_CLASSES}

# Options added to the label project's form after view_review_v1.json (seen in the export of 2026-11-17), with the
# parent proposed for each: the old class that held such clips. To confirm with the reviewers (DESIGN.md).
FORM_EXTRAS = {"A4C_RV": "A4C", "PLAX_ZOOM_MV": "PLAX_ZOOM", "RVOT": "PSAX", "AORTA_ASC": "OTHER",
               "SUBCOSTAL_4C": "SUBCOSTAL", "SUBCOSTAL_IVC": "SUBCOSTAL"}

# Other spellings of a class that a label export or a document may use.
ALIASES = {"PLAX_AV_ZOOM": "PLAX_ZOOM_AV", "PLAX_ZOOM_AOV": "PLAX_ZOOM_AV"}
UNSURE = "unsure"


def check_classes(old=OLD_CLASSES, new=NEW_CLASSES, parent=None):
    """Raise ValueError unless every new class has a parent among the old ones and every old class has a child."""
    parent = PARENT if parent is None else parent
    if len(set(old)) != len(old) or len(set(new)) != len(new):
        raise ValueError("A class is listed twice.")
    missing = [c for c in new if parent.get(c) not in old]
    if missing:
        raise ValueError(f"New classes without an old parent: {missing}.")
    orphans = sorted(set(old) - {parent[c] for c in new})
    if orphans:
        raise ValueError(f"Old classes no new class descends from: {orphans}.")


def read_old_mapping(path):
    """The classifier's classes in output order, from `view_classifier_class_mapping.json` ({"index": "name"}).
    Refuses a mapping that differs from `OLD_CLASSES`, which the rest of the code assumes."""
    with open(path) as f:
        mapping = json.load(f)
    order = tuple(mapping[str(i)] for i in range(len(mapping)))
    if order != OLD_CLASSES:
        raise ValueError("The class mapping differs from OLD_CLASSES: update classes.py before using these weights.")
    return order


def children(old_class, new=NEW_CLASSES, parent=None):
    parent = PARENT if parent is None else parent
    return [c for c in new if parent[c] == old_class]


def pseudo_label_classes(old=OLD_CLASSES, new=NEW_CLASSES, parent=None, exclude=()):
    """{old class: new class} for the old classes with exactly one descendant, which is then their pseudo-label;
    `exclude` removes more old classes (for example `OTHER`)."""
    out = {}
    for o in old:
        kids = children(o, new, parent)
        if len(kids) == 1 and o not in set(exclude):
            out[o] = kids[0]
    return out


def canonical(name, aliases=None):
    """A label as the class list spells it: stripped, `unsure` in lower case, aliases resolved; '' when empty."""
    if name is None:
        return ""
    text = str(name).strip()
    if text.lower() in ("", "nan", "none"):
        return ""
    if text.lower() == UNSURE:
        return UNSURE
    aliases = ALIASES if aliases is None else aliases
    return aliases.get(text, aliases.get(text.upper(), text))


def collapse(names, parent=None):
    """The old-class parent of each new class name."""
    parent = PARENT if parent is None else parent
    return [parent[n] for n in names]


def head_rows(old=OLD_CLASSES, new=NEW_CLASSES, parent=None):
    """For each new class, (row of the old head it starts from, bias offset). An old class whose probability is shared
    by k new classes gives each of them its row and its bias minus log(k), so the summed probability of the k classes
    equals the old class's probability before any training."""
    import math
    parent = PARENT if parent is None else parent
    share = {o: len(children(o, new, parent)) for o in old}
    return [(old.index(parent[c]), -math.log(share[parent[c]])) for c in new]


def form_classes(path):
    """The class names of a Labelbox form export (`view_labels.form.json`): the `value`s of the `view` question's
    options, `unsure` aside, in form order."""
    with open(path) as f:
        form = json.load(f)
    views = [c for c in form.get("classifications", []) if c.get("name") == "view"]
    if len(views) != 1:
        raise ValueError("The form has no single `view` question.")
    return tuple(str(o["value"]) for o in views[0].get("options", []) if str(o["value"]).lower() != UNSURE)


def check_against_form(names, form_path):
    """Raise when the live form offers a view the run's class list lacks (its labels could not be used); return the
    run's classes the form does not offer (kept as outputs, but no reviewer can give them)."""
    offered = form_classes(form_path)
    missing = [c for c in offered if c not in names]
    if missing:
        raise ValueError(f"The label form offers views the config's `classes` lacks: {missing}. Add them with a parent.")
    return [c for c in names if c not in offered]


def from_config(cfg):
    """(class names in output order, {class: old parent}) of a run: its config's `classes` mapping, checked, or the
    defaults."""
    mapping = cfg.get("classes") if isinstance(cfg, dict) else None
    if not mapping:
        return NEW_CLASSES, dict(PARENT)
    if not isinstance(mapping, dict):
        raise ValueError("`classes` must map each class to its parent among the current classifier's classes.")
    names, parent = tuple(mapping), {str(k): str(v) for k, v in mapping.items()}
    if UNSURE in {n.lower() for n in names}:
        raise ValueError("`unsure` is not a class.")
    check_classes(OLD_CLASSES, names, parent)
    return names, parent
