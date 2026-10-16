"""DeepECHO, EchoPrime and PanEcho labels mapped to the caption ontology: completeness against the vendored assets,
mapping semantics and the validator's refusals."""

import ast
import copy
import json
import unittest
import warnings

import yaml

from app.radar.label_mappings import (
    DEFAULT_MAPPINGS_PATH,
    ECHOPRIME_FINDINGS_PATH,
    ECHOPRIME_VIEWS_PATH,
    UNSPECIFIED,
    MappingError,
    _echoprime_coarse_views,
    completeness_problems,
    load_mappings,
    source_vocabularies,
    validate_mappings,
)
from app.radar.ontology import load_ontology

ONTOLOGY = load_ontology()


def raw():
    with open(DEFAULT_MAPPINGS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def echoprime_isin():
    """EchoPrime's own phrase matcher (`isin` in utils.py), taken from the vendored source without importing the
    module, which needs cv2 and torch."""
    with open(ECHOPRIME_VIEWS_PATH, encoding="utf-8") as f, warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(f.read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "isin")
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), ECHOPRIME_VIEWS_PATH, "exec"), namespace)
    return namespace["isin"]


class TestShippedMappings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_mappings(ONTOLOGY)
        cls.vocab = source_vocabularies()

    def test_every_source_label_and_class_is_mapped(self):
        self.assertEqual(completeness_problems(self.m, self.vocab), [])
        self.assertEqual(len(self.vocab["panecho_tasks"]), 40)
        self.assertEqual(len(self.vocab["echoprime_findings"]), 25)
        self.assertEqual(len(self.vocab["echoprime_views"]), 11)

    def test_every_echoprime_finding_relates_to_a_concept(self):
        for label, entry in self.m.sources["echoprime_findings"].entries.items():
            self.assertNotEqual(entry.relation, "unmapped", label)
            self.assertTrue(entry.concepts, label)

    def test_every_label_names_its_structures(self):
        for source in self.m.sources.values():
            if source.kind == "task":
                for label, entry in source.entries.items():
                    self.assertTrue(entry.structures, f"{source.name}.{label}")

    def test_units_and_transformations(self):
        tv = self.m.sources["panecho_tasks"].entries["TVPkGrad"]
        self.assertEqual((tv.relation, tv.unit), ("transformed", "mmHg"))
        self.assertEqual(ONTOLOGY.concept(tv.concepts[0]).unit, "m/s")  # a gradient is not a velocity
        self.assertEqual(self.m.sources["panecho_tasks"].entries["TAPSE"].relation, "transformed")
        self.assertEqual(self.m.sources["panecho_tasks"].entries["RVSP"].relation, "exact")

    def test_grouped_classes_stay_grouped(self):
        mr = self.m.sources["panecho_tasks"].entries["MVRegurgitation"]
        self.assertEqual(mr.relation, "grouped")
        self.assertEqual(mr.classes["Moderate|Severe"], ("moderate", "moderate_to_severe", "severe"))
        self.assertEqual(mr.unassigned, ("mild_to_moderate",))
        rap = self.m.sources["panecho_tasks"].entries["RAP-8-or-higher"]
        self.assertEqual(rap.classes, {"1.0": {"at_least": 8}, "0.0": {"below": 8}})

    def test_echoprime_phrase_matches_imply_no_value(self):
        """Every EchoPrime binary task is positive on a report sentence that negates its own phrase, so neither of
        its classes may imply a value (the mapping is only related)."""
        isin = echoprime_isin()
        self.assertTrue(isin("SEVERE MITRAL REGURGITATION", "No evidence of severe mitral regurgitation."))
        with open(ECHOPRIME_FINDINGS_PATH, encoding="utf-8") as f:
            tasks = json.load(f)
        binary = [task for task, spec in tasks.items() if spec["mode"] == "binary"]
        self.assertEqual(len(binary), 23)
        for task in binary:
            negated = [f"Compared with the prior study{phrase}is no longer seen." for phrase in
                       tasks[task]["label_sources"]]
            self.assertTrue(any(isin(phrase, text) for phrase, text in zip(tasks[task]["label_sources"], negated)),
                            task)
            entry = self.m.sources["echoprime_findings"].entries[task]
            self.assertEqual(entry.relation, "related", task)
            self.assertEqual(entry.classes, {"positive": None, "negative": None}, task)
            self.assertTrue(entry.notes, task)  # what the phrases state

    def test_ambiguous_views_keep_their_ambiguity(self):
        doppler = self.m.sources["echoprime_views"].entries["Apical_Doppler"]
        self.assertEqual(doppler.relation, "grouped")
        self.assertIn("A4C", doppler.expanded_views)
        self.assertIn("A2C_ZOOM", doppler.expanded_views)
        self.assertEqual(doppler.acquisitions, ("color_flow", "spectral_doppler"))
        self.assertEqual(self.m.sources["echoprime_views"].entries["A4C"].acquisitions, UNSPECIFIED)
        deepecho = self.m.sources["deepecho_views"].entries
        self.assertEqual({e.relation for e in deepecho.values()}, {"exact"})

    def test_coarse_views_are_read_without_importing_echoprime(self):
        views = _echoprime_coarse_views(ECHOPRIME_VIEWS_PATH)
        self.assertEqual(views[0], "A2C")
        self.assertIn("Apical_Doppler", views)


class TestCompleteness(unittest.TestCase):
    def test_missing_and_invented_labels_and_classes(self):
        m = load_mappings(ONTOLOGY)
        vocab = copy.deepcopy(source_vocabularies())
        vocab["panecho_tasks"]["NewTask"] = None
        vocab["panecho_tasks"]["MVRegurgitation"] = ("Mild", "Moderate|Severe")
        del vocab["echoprime_views"]["SSN"]
        problems = "\n".join(completeness_problems(m, vocab))
        self.assertIn("missing labels ['NewTask']", problems)
        self.assertIn("MVRegurgitation: classes", problems)
        self.assertIn("unknown labels ['SSN']", problems)

    def test_related_task_classes_are_checked_too(self):
        m = load_mappings(ONTOLOGY)
        vocab = copy.deepcopy(source_vocabularies())
        self.assertEqual(m.sources["panecho_tasks"].entries["LVOT20mmHg"].relation, "related")
        vocab["panecho_tasks"]["LVOT20mmHg"] = ("low", "mid", "high")
        vocab["echoprime_findings"]["pacemaker"] = ("present",)
        problems = "\n".join(completeness_problems(m, vocab))
        self.assertIn("LVOT20mmHg: classes", problems)
        self.assertIn("pacemaker: classes", problems)
        vocab = copy.deepcopy(source_vocabularies())
        vocab["panecho_tasks"]["LVEDV"] = ("small", "large")  # a regression task turned classification
        self.assertIn("LVEDV: classes", "\n".join(completeness_problems(m, vocab)))


class TestValidator(unittest.TestCase):
    def refused(self, mutate, message):
        data = copy.deepcopy(raw())
        mutate(data)
        with self.assertRaises(MappingError) as ctx:
            validate_mappings(data, ONTOLOGY)
        self.assertIn(message, str(ctx.exception))

    def test_shipped_file_passes(self):
        validate_mappings(raw(), ONTOLOGY)

    def test_refusals(self):
        def label(d, source, name):
            return d["sources"][source]["labels"][name]

        self.refused(lambda d: d.update(ontology="radar-caption-ontology-v2"), "is not the loaded ontology")
        self.refused(lambda d: label(d, "panecho_tasks", "MVRegurgitation").update(relation="exact"),
                     "an exact mapping gives each class one value")
        self.refused(lambda d: label(d, "panecho_tasks", "MVRegurgitation").pop("unassigned"),
                     "`unassigned` must list exactly")
        self.refused(lambda d: label(d, "panecho_tasks", "LVSize")["classes"].update(Normal=["normal", "huge"]),
                     "['huge'] are not values of lv_size")
        self.refused(lambda d: label(d, "panecho_tasks", "AVStructure")["classes"].update(Normal=["false"]),
                     "are not values of bicuspid_aortic_valve")  # the string "false" is not the boolean false
        self.refused(lambda d: label(d, "panecho_tasks", "TVPkGrad").update(relation="exact"),
                     "has the concept's unit 'm/s', not 'mmHg'")
        self.refused(lambda d: label(d, "panecho_tasks", "TVPkGrad").pop("transform"), "a unit and a transform")
        self.refused(lambda d: label(d, "panecho_tasks", "EF").update(transform="x"), "only a transformed mapping")
        self.refused(lambda d: label(d, "panecho_tasks", "LVSV").pop("reason"), "needs a reason")
        self.refused(lambda d: label(d, "panecho_tasks", "LVSV").update(structures=["lung"]), "unknown ['lung']")
        self.refused(lambda d: label(d, "panecho_tasks", "LVEDV").update(concepts=["lv_volume"]),
                     "unknown ['lv_volume']")
        self.refused(lambda d: label(d, "panecho_tasks", "EF").update(concept="lvef"), "unknown concept 'lvef'")
        self.refused(lambda d: label(d, "panecho_tasks", "RAP-8-or-higher")["classes"].update(
            {"1.0": {"at_least": 80}}), "is empty or outside")
        self.refused(lambda d: label(d, "panecho_tasks", "RAP-8-or-higher")["classes"].update(
            {"1.0": {"at_least": "8"}}), "must be numbers")
        self.refused(lambda d: label(d, "panecho_tasks", "RAP-8-or-higher")["classes"].update(
            {"0.0": {"below": 9}}), "classes [['1.0', '0.0']] overlap")
        self.refused(lambda d: label(d, "panecho_tasks", "TVRegurgitation")["classes"].update(
            Mild=["mild", "moderate"]), "classes [['Mild', 'Moderate|Severe']] overlap")
        self.refused(lambda d: label(d, "panecho_tasks", "AVStructure")["classes"].update(Normal=[True]),
                     "classes [['Bicuspid', 'Normal']] overlap")
        self.refused(lambda d: label(d, "panecho_tasks", "EF").update(units="%"), "unknown fields ['units']")
        self.refused(lambda d: label(d, "echoprime_views", "A4C").update(relation="exact"), "maps to one view")
        self.refused(lambda d: label(d, "echoprime_views", "SSN").update(relation="grouped"), "covers several views")
        self.refused(lambda d: label(d, "echoprime_views", "A4C").update(acquisitions=["doppler"]),
                     "acquisitions must")
        self.refused(lambda d: label(d, "echoprime_views", "A4C").update(views=["A6C"]), "neither a view nor")
        self.refused(lambda d: label(d, "echoprime_views", "A4C").update(relation="related"), "exact or grouped")
        self.refused(lambda d: label(d, "panecho_tasks", "AVStructure")["classes"].update(Normal="none"),
                     "a class maps to a non-empty list of values")
        self.refused(lambda d: label(d, "echoprime_findings", "mitral_regurgitation").update(
            relation="grouped", concept="mitral_regurgitation", classes={"positive": ["severe"]}),
            "unknown fields ['concepts']")
        self.refused(lambda d: label(d, "panecho_tasks", "LVOT20mmHg").update(classes={"0.0": [False]}),
                     "lists its source classes as names")


if __name__ == "__main__":
    unittest.main()
