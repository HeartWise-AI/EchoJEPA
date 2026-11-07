# tests/models/test_radar_routing.py

"""The RADAR routing registry: the shipped v0 file follows the reviewed rules (level-free views, IVC only through a
named weak override, no PSAX for the mitral valve, RA and Hemo untrained), the loader refuses malformed registries,
and clip eligibility masks padded and unknown views."""

import os
import tempfile
import unittest

import torch
import yaml

from app.radar.ontology import VIEWS
from src.models.radar.routing import load_routing


class TestShippedRegistry(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.routing = load_routing()

    def test_view_classes_are_the_ontology_views_in_order(self):
        self.assertEqual(self.routing.view_classes, VIEWS)

    def test_trained_structures_and_order(self):
        trained = [s.name for s in self.routing.trained]
        self.assertEqual(trained, ["lv", "rv", "la", "mitral", "aortic", "tricuspid", "pulmonic", "aorta", "ivc",
                                   "pericardium"])
        self.assertFalse(self.routing.structure("ra").train)
        self.assertFalse(self.routing.structure("hemo").train)

    def test_reviewed_routing_rules(self):
        self.assertNotIn("PSAX", self.routing.structure("mitral").views)
        self.assertEqual(self.routing.structure("ivc").views, ("SUBCOSTAL",))
        self.assertEqual(self.routing.structure("ivc").routing, "weak")
        weak = [s.name for s in self.routing.structures if s.routing == "weak"]
        self.assertEqual(weak, ["ivc"])
        self.assertTrue(all("SUBCOSTAL" not in s.views for s in self.routing.structures if s.name != "ivc"))
        self.assertEqual(self.routing.structure("pulmonic").views, ("PSAX_AV",))

    def test_other_is_never_eligible(self):
        table = self.routing.eligibility()
        self.assertFalse(table[self.routing.view_index("OTHER")].any())
        self.assertFalse(table[-1].any())  # unknown or padded

    def test_clip_eligibility(self):
        names = ["lv", "pulmonic", "ivc"]
        ids = torch.tensor([[self.routing.view_index("A4C"), self.routing.view_index("PSAX_AV"),
                             self.routing.view_index("SUBCOSTAL"), -1]])
        valid = torch.tensor([[True, True, False, False]])
        got = self.routing.clip_eligibility(ids, valid, names)
        self.assertEqual(got.shape, (1, 3, 4))
        expected = torch.tensor([[[True, False, False, False],    # lv: A4C only (PSAX_AV is not an LV view)
                                  [False, True, False, False],    # pulmonic: PSAX_AV
                                  [False, False, False, False]]])  # ivc: its subcostal clip is padded
        self.assertTrue(torch.equal(got, expected))

    def test_view_families(self):
        families = self.routing.view_families
        self.assertEqual(families["A4C_ZOOM"], "A4C")
        self.assertEqual(families["PLAX_DEEP"], "PLAX")
        self.assertNotIn("PSAX_AV", families)                  # the pulmonic valve's only view stays its own family
        self.assertEqual(self.routing.structure("pulmonic").views, ("PSAX_AV",))
        self.assertTrue(set(families) | set(families.values()) <= set(self.routing.view_classes))

    def test_unknown_view_name_has_no_index(self):
        self.assertEqual(self.routing.view_index("PSAX_MV"), -1)


class TestLoaderRefusals(unittest.TestCase):

    def _load(self, edit):
        from src.models.radar.routing import DEFAULT_ROUTING
        with open(DEFAULT_ROUTING) as f:
            raw = yaml.safe_load(f)
        edit(raw)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "routing.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(raw, f)
            return load_routing(path)

    def test_refuses(self):
        cases = {
            "unknown views": lambda r: r["structures"][0]["views"].append("PSAX_MV"),
            "cannot be routed": lambda r: r["structures"][0]["views"].append("OTHER"),
            "routing must be": lambda r: r["structures"][0].update(routing="levels"),
            "needs views and a report_column": lambda r: r["structures"][-1].update(train=True),
            "distinct nonempty": lambda r: r["structures"].append(dict(r["structures"][0])),
            "schema must be": lambda r: r.update(schema="other"),
            "view_families names unknown": lambda r: r["view_families"].update(A4C_RV="A4C"),
            "families to other families": lambda r: r["view_families"].update(A4C="PLAX"),
        }
        for message, edit in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                self._load(edit)

    def test_sha256_changes_with_content(self):
        a = self._load(lambda r: None)
        b = self._load(lambda r: r["structures"][0].update(note="edited"))
        self.assertNotEqual(a.sha256, b.sha256)


if __name__ == "__main__":
    unittest.main()
