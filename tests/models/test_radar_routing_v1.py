# tests/models/test_radar_routing_v1.py

"""The v1 routing registry (Doppler stills): v0 is still the default and is unchanged in what it trains; v1 equals
v0 except Hemo, which trains on stills only with a structured caption; still tags route stills to structures, an
unknown tag or a padded slot to none; the loader refuses malformed still routing."""

import os
import tempfile
import unittest

import torch
import yaml

from src.models.radar.routing import DEFAULT_ROUTING, load_routing

V1 = os.path.join(os.path.dirname(DEFAULT_ROUTING), "radar_routing_v1_hemo.yaml")


class TestV1Registry(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.v0, cls.v1 = load_routing(), load_routing(V1)

    def test_v0_is_default_and_has_no_still_routing(self):
        self.assertEqual(self.v0.version, 0)
        self.assertEqual(self.v0.still_tags, ())
        self.assertFalse(self.v0.structure("hemo").train)

    def test_v1_equals_v0_but_hemo(self):
        self.assertEqual(self.v1.version, 1)
        self.assertEqual(self.v1.view_classes, self.v0.view_classes)
        self.assertEqual(self.v1.view_families, self.v0.view_families)
        for s0 in self.v0.structures:
            s1 = self.v1.structure(s0.name)
            if s0.name == "hemo":
                continue
            self.assertEqual((s1.views, s1.report_column, s1.train, s1.routing),
                             (s0.views, s0.report_column, s0.train, s0.routing))
        hemo = self.v1.structure("hemo")
        self.assertTrue(hemo.train)
        self.assertEqual((hemo.views, hemo.report_column, hemo.caption_source), ((), None, "structured"))
        self.assertEqual([s.name for s in self.v1.trained][-1], "hemo")   # appended: v0 query order kept

    def test_still_eligibility(self):
        names = [s.name for s in self.v1.trained]
        tags = torch.tensor([[self.v1.still_tag_index("TR"), self.v1.still_tag_index("LVOT"),
                              self.v1.still_tag_index("AO"), self.v1.still_tag_index("MV_INFLOW")]])
        valid = torch.tensor([[True, True, True, False]])
        got = self.v1.still_eligibility(tags, valid, names)
        self.assertEqual(got.shape, (1, len(names), 4))
        fed = {names[s] for s in got[0, :, 0].nonzero().flatten().tolist()}
        self.assertEqual(fed, {"tricuspid", "hemo"})
        self.assertEqual({names[s] for s in got[0, :, 1].nonzero().flatten().tolist()}, {"aortic"})
        self.assertFalse(got[0, :, 2].any())          # AO: no tag, no structure
        self.assertFalse(got[0, :, 3].any())          # padded
        self.assertEqual(self.v1.still_type_index("cw_AV"), 0)
        self.assertEqual(self.v1.still_type_index("cw_unknown"), -1)

    def test_no_still_feeds_a_bmode_only_structure(self):
        fed = {name for targets in self.v1.still_structs.values() for name in targets}
        self.assertEqual(fed & {"lv", "rv", "la", "ivc", "pericardium"}, set())


class TestV1Refusals(unittest.TestCase):

    def _load(self, edit):
        with open(V1) as f:
            raw = yaml.safe_load(f)
        edit(raw)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "routing.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(raw, f)
            return load_routing(path)

    def test_refuses(self):
        def no_still_for_hemo(r):
            for targets in r["still_structs"].values():
                if "hemo" in targets:
                    targets.remove("hemo")

        def hemo_from_report(r):
            r["structures"][-1]["caption_source"] = "report"

        cases = {
            "names unknown structures": lambda r: r["still_structs"].update(XX=["valve"]),
            "distinct structure names": lambda r: r["still_structs"].update(AV=["aortic", "aortic"]),
            "caption_source must be": lambda r: r["structures"][-1].update(caption_source="template"),
            "needs views and a report_column": no_still_for_hemo,
            "needs views and a report_column ": hemo_from_report,
            "still_types must be": lambda r: r.update(still_types=["cw_AV", "cw_AV"]),
        }
        for message, edit in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message.strip()):
                self._load(edit)


if __name__ == "__main__":
    unittest.main()
