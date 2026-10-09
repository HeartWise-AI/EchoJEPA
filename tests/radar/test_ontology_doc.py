"""The generated parts of the RADAR documents match the ontology, mapping and example YAML files."""

import unittest

from app.radar.label_mappings import load_mappings
from app.radar.ontology import load_ontology
from app.radar.ontology_doc import (
    DEFAULT_DOC_PATH,
    INSTRUCTIONS_PATHS,
    document_sections,
    mapping_text,
    prior_text,
    sync,
)
from app.radar.review_rules import load_examples


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class TestOntologyDoc(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.o = load_ontology()
        cls.m = load_mappings(cls.o)
        cls.sections = document_sections(cls.o, cls.m, load_examples(cls.o))
        cls.text = read(DEFAULT_DOC_PATH)

    def test_documents_are_up_to_date(self):
        self.assertEqual(set(self.sections), {DEFAULT_DOC_PATH, *INSTRUCTIONS_PATHS.values()})
        for path, sections in self.sections.items():
            text = read(path)
            self.assertEqual(sync(text, sections), text,
                             f"{path} is out of date: run python -m app.radar.ontology_doc")

    def test_every_section_is_required_once(self):
        sections = self.sections[DEFAULT_DOC_PATH]
        name = next(iter(sections))
        begin, end = f"<!-- BEGIN GENERATED: {name} -->\n", f"<!-- END GENERATED: {name} -->"
        start = self.text.index(begin)
        stop = self.text.index(end) + len(end)
        with self.assertRaises(ValueError):
            sync(self.text[:start] + self.text[stop:], sections)
        with self.assertRaises(ValueError):
            sync(self.text + "\n" + self.text[start:stop], sections)

    def test_instructions_share_ids_across_languages(self):
        en, fr = (read(INSTRUCTIONS_PATHS[lang]) for lang in ("en", "fr"))
        for q in self.o.review.questions:
            self.assertIn(f"`{q.id}`", en)
            self.assertIn(f"`{q.id}`", fr)
        self.assertEqual(en.count("\n### "), fr.count("\n### "))  # same rules and examples

    def test_prior_text_names_whole_groups(self):
        self.assertEqual(prior_text(self.o, self.o.concept("ivc_visible")), "`subcostal` (bmode, color_flow, mmode)")
        self.assertIn("`apical_4c`", prior_text(self.o, self.o.concept("mitral_regurgitation")))
        self.assertNotIn("A4C_LV", prior_text(self.o, self.o.concept("mitral_regurgitation")))

    def test_mapping_text(self):
        panecho = self.m.sources["panecho_tasks"].entries
        self.assertEqual(mapping_text(panecho["RAP-8-or-higher"]),
                         "`ra_pressure`: `1.0` → at least 8; `0.0` → below 8; "
                         "also related to `ivc_dilation`, `ivc_collapse`")
        self.assertIn("no class: `mild_to_moderate`", mapping_text(panecho["AVRegurg"]))
        self.assertEqual(mapping_text(panecho["AVStructure"]), "`bicuspid_aortic_valve`: `Bicuspid` → `true`; "
                                                               "`Normal` → `false`")
        self.assertTrue(mapping_text(panecho["LVSV"]).startswith("structures `left_ventricle`: "))


if __name__ == "__main__":
    unittest.main()
