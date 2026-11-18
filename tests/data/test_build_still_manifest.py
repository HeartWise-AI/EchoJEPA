# tests/data/test_build_still_manifest.py

"""`data/build_still_manifest.py` on synthetic measurement and metadata tables (made-up identifiers only): still tags
and types, unit normalisation and plausible ranges, one row per still with its beats averaged, the study found
through the DICOM study folder (an ambiguous folder is left out), age from the dates without the birth date, the
patient split inherited from RADAR's split or drawn from a salted hash, and the end-to-end build (no patient id or
birth date in the outputs, outputs refused inside the repository)."""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "data") not in sys.path:
    sys.path.append(os.path.join(REPO, "data"))
import build_still_manifest as bsm  # noqa: E402


def meas_row(png, study_dir, struct, quantity, value, unit, mode="CW", sop=None, box=(10, 10, 60, 20)):
    return {"png_path": f"/stills/{study_dir}/{png}.png", "dcm_path": f"/dicom/{study_dir}/{png}.dcm",
            "SOPInstanceUID": sop or f"sop-{study_dir}-{png}", "StudyInstanceUID": f"extractor-{study_dir}",
            "struct": struct, "quantity": quantity, "value": value, "unit": unit, "mode": mode, "year": 2024,
            "Manufacturer": "Vendor", "rows": 708, "cols": 1016, "spec_x0": 0, "spec_y0": 250, "spec_x1": 1016,
            "spec_y1": 700, "spec_dy": -1.0, "spec_ref_y0": 100, "x0": box[0], "y0": box[1], "x1": box[2],
            "y1": box[3]}


def meas_table():
    return pd.DataFrame([
        meas_row("a", "d1", "AV", "VMAX", 3.0, "m/s"),
        meas_row("a", "d1", "AV", "VMAX", 3.4, "m/s", box=(10, 30, 60, 40)),
        meas_row("a", "d1", "AV", "MAXPG", 36.0, "mmHg"), meas_row("a", "d1", "AV", "VTI", 900.0, "cm"),
        meas_row("b", "d1", "LVOT", "VMAX", 95.0, "cm/s", mode="PW_TDI"),
        meas_row("c", "d2", "MV", "E", 0.8, "m/s", mode="PW_TDI"),
        meas_row("c", "d2", "MV", "DT", 0.2, "s", mode="PW_TDI"),
        meas_row("d", "d3", "TR", "VMAX", 2.9, "m/s"),
        meas_row("e", "d4", "", "HR", 70.0, "bpm"),
        meas_row("f", "d5", "AV", "VMAX", 2.0, "m/s"),
    ])


def metadata_table():
    rows = []
    for folder, uid, mrn, sex, birth in (("d1", "1.2.3.1", "P001", "F", "19500615"),
                                         ("d2", "1.2.3.2", "P002", "M", "19800101"),
                                         ("d3", "1.2.3.3", "P003", "", "bad"),
                                         ("d4", "1.2.3.4", "P004", "M", "19900101")):
        for video in range(2):
            rows.append({"study_uid": f"'{uid}", "dcm_path": f"/dicom/{folder}/v{video}.dcm", "mrn": mrn,
                         "sex": sex, "birthdate": birth, "date": "20240615", "patient_weight_kg": 70.0,
                         "patient_height_m": 1.75})
    rows.append({"study_uid": "1.2.3.5", "dcm_path": "/dicom/d5/v0.dcm", "mrn": "P005", "sex": "M",
                 "birthdate": "19600101", "date": "20240101", "patient_weight_kg": None, "patient_height_m": None})
    rows.append({"study_uid": "1.2.3.6", "dcm_path": "/dicom/d5/v1.dcm", "mrn": "P005", "sex": "M",
                 "birthdate": "19600101", "date": "20240101", "patient_weight_kg": None, "patient_height_m": None})
    return pd.DataFrame(rows)


class TestRules(unittest.TestCase):

    def test_tags_and_types(self):
        self.assertEqual(bsm.still_tag(["AV", "AV", "LVOT"], ["VMAX"], "CW"), "AV")
        self.assertEqual(bsm.still_tag(["MV"], ["E", "A"], "PW_TDI"), "MV_INFLOW")
        self.assertEqual(bsm.still_tag(["MV"], ["VTI", "MEANPG"], "CW"), "MV")
        self.assertEqual(bsm.still_tag([""], ["E'"], "PW_TDI"), "TDI")
        self.assertIsNone(bsm.still_tag(["AO", ""], ["VMAX"], "CW"))
        self.assertEqual(bsm.still_type("CW", "AV"), "cw_AV")
        self.assertEqual(bsm.still_type("PW_TDI", "TDI"), "tdi_TDI")
        self.assertEqual(bsm.still_type("CW", None), "")

    def test_units(self):
        self.assertAlmostEqual(bsm.to_unit(95.0, "cm/s", "velocity"), 0.95)
        self.assertAlmostEqual(bsm.to_unit(0.2, "s", "time"), 200.0)
        self.assertTrue(np.isnan(bsm.to_unit(3.0, "mmHg", "velocity")))

    def test_a_missing_unit_is_no_unit(self):
        for missing in (None, float("nan"), pd.NA):
            with self.subTest(unit=missing):
                self.assertEqual(bsm.to_unit(30.0, missing, "pressure"), 30.0)
                self.assertTrue(np.isnan(bsm.to_unit(2.0, missing, "velocity")))

    def test_aggregate(self):
        stills, dropped = bsm.aggregate_stills(meas_table())
        self.assertEqual(dropped["out_of_range"], 1)         # the 900 cm VTI
        self.assertEqual(json.loads(stills.set_index("png_path").loc["/stills/d1/a.png"].signed_values)["AV_VMAX"],
                         [3.0, 3.4])
        a = stills.set_index("png_path").loc["/stills/d1/a.png"]
        self.assertAlmostEqual(a.y_AV_VMAX, 3.2)              # mean over beats
        self.assertAlmostEqual(a.y_AV_MAXPG, 36.0)
        self.assertNotIn("y_AV_VTI", a.dropna().index)
        self.assertEqual((a.still_tag, a.still_type, a.n_rows), ("AV", "cw_AV", 4))
        self.assertEqual(json.loads(a.text_boxes), [[10, 10, 60, 20], [10, 30, 60, 40]])
        self.assertAlmostEqual(a.v_top_ms, 1.0)
        b = stills.set_index("png_path").loc["/stills/d1/b.png"]
        self.assertAlmostEqual(b.y_LVOT_VMAX, 0.95)
        c = stills.set_index("png_path").loc["/stills/d2/c.png"]
        self.assertEqual(c.still_tag, "MV_INFLOW")
        self.assertAlmostEqual(c.y_MV_DT, 200.0)
        self.assertEqual(stills.set_index("png_path").loc["/stills/d4/e.png"].still_tag, "")
        self.assertTrue(all(len(s) == 16 and "sop" not in s for s in stills.still_id))

    def test_negative_values_are_kept_as_read_and_a_magnitude_only_where_asked(self):
        rows = pd.DataFrame([meas_row("n", "d1", "TR", "VMAX", -2.8, "m/s"),
                             meas_row("n", "d1", "TR", "VMAX", -3.0, "m/s"),
                             meas_row("n", "d1", "TR", "MAXPG", -31.0, "mmHg"),
                             meas_row("m", "d1", "AV", "VMAX", -4.0, "m/s"),
                             meas_row("m", "d1", "AV", "VMAX", 4.0, "m/s")])
        plain, dropped = bsm.aggregate_stills(rows)
        n = plain.set_index("png_path").loc["/stills/d1/n.png"]
        self.assertFalse(any(c.startswith("y_") and pd.notna(n[c]) for c in plain.columns))    # no blind absolute
        self.assertEqual(json.loads(n.signed_values), {"TR_MAXPG": [-31.0], "TR_VMAX": [-2.8, -3.0]})
        self.assertEqual((dropped["negative_dropped"], dropped["mixed_sign"], dropped["sign_flipped"]), (3, 1, 0))
        flipped, dropped = bsm.aggregate_stills(rows, magnitude={"VMAX"})
        n = flipped.set_index("png_path").loc["/stills/d1/n.png"]
        self.assertAlmostEqual(n.y_TR_VMAX, 2.9)
        self.assertNotIn("y_TR_MAXPG", n.dropna().index)             # only the quantities asked for
        self.assertEqual(json.loads(n.signed_values)["TR_VMAX"], [-2.8, -3.0])              # the sign stays on record
        self.assertEqual((dropped["sign_flipped"], dropped["negative_dropped"], dropped["mixed_sign"]), (2, 1, 1))
        m = flipped.set_index("png_path").loc["/stills/d1/m.png"]
        self.assertNotIn("y_AV_VMAX", m.dropna().index)              # beats of opposite sign: no target
        counts = bsm.negative_values(bsm.normalize_meas(rows))
        self.assertEqual(counts["TR_VMAX|Vendor"], ["<5", 2])

    def test_age(self):
        age = bsm.age_years(["19500615", "bad", "20300101"], ["20240615", "20240615", "20240101"])
        self.assertAlmostEqual(age[0], 74.0, places=1)
        self.assertTrue(np.isnan(age[1]) and np.isnan(age[2]))


class TestLinkAndSplit(unittest.TestCase):

    def test_link_through_study_folder(self):
        stills, _ = bsm.aggregate_stills(meas_table())
        linked, demo, unlinked = bsm.link_studies(stills, metadata_table(), "salt")
        self.assertEqual(unlinked, 1)                          # d5 has two study_uids: ambiguous
        self.assertEqual(set(linked.study_id), {"1.2.3.1", "1.2.3.2", "1.2.3.3", "1.2.3.4"})
        self.assertNotIn("extractor-d1", set(linked.study_id))
        self.assertEqual(list(demo.columns), list(bsm.DEMOGRAPHIC_COLUMNS))
        d = demo.set_index("study_id")
        self.assertAlmostEqual(d.loc["1.2.3.1", "age_years"], 74.0, places=1)
        self.assertTrue(np.isnan(d.loc["1.2.3.3", "age_years"]))
        self.assertEqual(d.loc["1.2.3.3", "sex"], "")
        self.assertAlmostEqual(d.loc["1.2.3.1", "bsa_m2"], 1.84, places=2)
        self.assertNotIn("P001", set(demo.patient_hash))

    def test_splits_inherited_drawn_and_checked(self):
        stills, _ = bsm.aggregate_stills(meas_table())
        linked, _, _ = bsm.link_studies(stills, metadata_table(), "salt")
        splits = pd.DataFrame({"study_id": ["1.2.3.1", "9.9.9"], "split": ["val", "test"],
                               "patient_id": ["P001", "P002"]})
        out, counts = bsm.assign_splits(linked, splits, metadata_table(), "salt", (0.8, 0.1, 0.1))
        by_study = out.drop_duplicates("study_id").set_index("study_id").split
        self.assertEqual(by_study["1.2.3.1"], "val")
        self.assertEqual(by_study["1.2.3.2"], "test")          # another study of an inherited patient
        self.assertEqual(counts, {"patients_inherited": 2, "patients_drawn": 2})
        again, _ = bsm.assign_splits(linked, splits, metadata_table(), "salt", (0.8, 0.1, 0.1))
        self.assertTrue(out.split.equals(again.split))
        two = pd.concat([splits, pd.DataFrame({"study_id": ["8.8.8"], "split": ["train"], "patient_id": ["P001"]})])
        with self.assertRaisesRegex(SystemExit, "two splits"):
            bsm.assign_splits(linked, two, metadata_table(), "salt", (0.8, 0.1, 0.1))

    def test_archive_wide_patient_split_comes_before_the_hash(self):
        stills, _ = bsm.aggregate_stills(meas_table())
        linked, _, _ = bsm.link_studies(stills, metadata_table(), "salt")
        splits = pd.DataFrame({"study_id": ["1.2.3.1", "9.9.9"], "split": ["val", "test"],
                               "patient_id": ["P001", "P002"]})
        drawn, _ = bsm.assign_splits(linked, splits, metadata_table(), "salt", (0.8, 0.1, 0.1))
        loose = drawn[~drawn.patient.isin({"P001", "P002"})].drop_duplicates("patient")
        patient, was = loose.patient.iloc[0], loose.split.iloc[0]
        other = next(s for s in bsm.SPLITS if s != was)
        table = pd.DataFrame({"patient_id": ["P001", "00" + patient, "ZZZ"], "split": ["val", other, "train"]})
        out, counts = bsm.assign_splits(linked, splits, metadata_table(), "salt", (0.8, 0.1, 0.1),
                                        patient_splits=table)
        self.assertEqual(set(out[out.patient == patient].split), {other})       # the table, not the hash
        self.assertEqual(counts["patients_inherited_from_patient_splits"], 1)
        self.assertEqual(counts["patients_drawn"], loose.patient.nunique() - 1)
        self.assertEqual(set(out[out.patient == "P001"].split), {"val"})
        clash = table.assign(split=["train", other, "train"])                   # the cohort split says val
        with self.assertRaisesRegex(SystemExit, "another split in --patient-splits"):
            bsm.assign_splits(linked, splits, metadata_table(), "salt", (0.8, 0.1, 0.1), patient_splits=clash)
        twice = pd.concat([table, pd.DataFrame({"patient_id": ["ZZZ"], "split": ["val"]})])
        with self.assertRaisesRegex(SystemExit, "two splits"):
            bsm.assign_splits(linked, splits, metadata_table(), "salt", (0.8, 0.1, 0.1), patient_splits=twice)

    def test_hashed_split_fractions(self):
        draws = pd.Series([bsm.hashed_split(f"P{i}", "s", (0.8, 0.1, 0.1)) for i in range(4000)]).value_counts(
            normalize=True)
        self.assertAlmostEqual(draws["train"], 0.8, delta=0.03)

    def test_temporal_years_are_held_out_and_never_train(self):
        rows = meas_table()
        late = pd.DataFrame([{**meas_row("x", "d1", "AV", "VMAX", 3.0, "m/s"), "year": 2025},      # P001: train
                             {**meas_row("y", "d2", "AV", "VMAX", 3.0, "m/s"), "year": 2025},      # P002: test
                             {**meas_row("z", "d4", "AV", "VMAX", 3.0, "m/s"), "year": 2025}])     # P004: only 2025
        rows = pd.concat([rows[~rows.png_path.str.contains("/d4/")], late], ignore_index=True)
        stills, _ = bsm.aggregate_stills(rows)
        linked, _, _ = bsm.link_studies(stills, metadata_table(), "salt")
        splits = pd.DataFrame({"study_id": ["1.2.3.1", "1.2.3.2"], "split": ["train", "test"]})
        out, counts = bsm.assign_splits(linked, splits, metadata_table(), "salt", (0.8, 0.1, 0.1), [2025])
        by_png = out.set_index("png_path").split
        self.assertNotIn("/stills/d1/x.png", by_png.index)                  # a trained patient's 2025 still leaves
        self.assertEqual(by_png["/stills/d2/y.png"], bsm.TEMPORAL_SPLIT)
        self.assertEqual(by_png["/stills/d4/z.png"], bsm.TEMPORAL_SPLIT)    # whatever split the hash would draw
        self.assertEqual(by_png["/stills/d1/a.png"], "train")
        self.assertEqual(by_png["/stills/d2/c.png"], "test")                # the random test keeps the earlier years
        self.assertFalse(((pd.to_numeric(out.year) == 2025) & out.split.isin(["train", "val"])).any())
        self.assertEqual((counts["temporal_stills"], counts["temporal_stills_left_out_trained_patients"]), (2, 1))


class TestV11Quantities(unittest.TestCase):

    def test_mitral_inflow_and_mva_by_pht_are_targets(self):
        rows = pd.DataFrame([meas_row("i", "d2", "MV", "E", 0.8, "m/s", mode="PW_TDI"),
                             meas_row("i", "d2", "MV", "A", 60.0, "cm/s", mode="PW_TDI"),
                             meas_row("i", "d2", "MV", "E/A", 1.33, None, mode="PW_TDI"),
                             meas_row("i", "d2", "MV", "E/A", 25.0, None, mode="PW_TDI"),          # out of range
                             meas_row("s", "d2", "MV", "PHT", 180.0, "ms"),
                             meas_row("s", "d2", "MVA", "(PHT)", 1.2, "cm²"),
                             meas_row("s", "d2", "MVA", "(VTI)", 1.1, "cm2")])                     # not a target
        stills, dropped = bsm.aggregate_stills(rows)
        i = stills.set_index("png_path").loc["/stills/d2/i.png"]
        self.assertEqual(i.still_tag, "MV_INFLOW")
        self.assertAlmostEqual(i.y_MV_E, 0.8)
        self.assertAlmostEqual(i.y_MV_A, 0.6)
        self.assertAlmostEqual(i.y_MV_EA, 1.33)
        self.assertEqual(dropped["out_of_range"], 1)
        s = stills.set_index("png_path").loc["/stills/d2/s.png"]
        self.assertAlmostEqual(s.y_MVA_PHT, 1.2)
        self.assertAlmostEqual(s.y_MV_PHT, 180.0)
        for column in ("y_MV_EA", "y_MVA_PHT"):
            self.assertIn(column, bsm.TARGET_COLUMNS)
            self.assertNotIn("/", column)
        self.assertEqual(bsm.target_of("MV", "E/A"), ("MV_EA", "ratio"))
        self.assertIsNone(bsm.target_of("TV", "E/A"))
        self.assertIn("EA", next(g for g in bsm.LEAK_GROUPS if "E" in g))

    def test_tissue_doppler_is_never_a_target(self):
        rows = pd.DataFrame([meas_row("t", "d2", "MV", "E'", 0.09, "m/s", mode="PW_TDI"),
                             meas_row("t", "d2", "MV", "E", 0.8, "m/s", mode="PW_TDI"),
                             meas_row("t", "d2", "TV", "S'", 0.12, "m/s", mode="PW_TDI"),
                             meas_row("u", "d2", "MV", "E/E'", 9.0, None, mode="PW_TDI"),
                             meas_row("u", "d2", "MV", "E", 0.7, "m/s", mode="PW_TDI")])
        stills, dropped = bsm.aggregate_stills(rows)
        t = stills.set_index("png_path").loc["/stills/d2/t.png"]
        self.assertEqual((t.still_tag, t.still_type), ("TDI", "tdi_TDI"))           # still routed
        self.assertFalse(any(c.startswith("y_") and pd.notna(t[c]) for c in stills.columns))
        self.assertEqual(json.loads(t.signed_values), {})
        self.assertEqual(dropped["tdi_excluded"], 1)                                # the MV E read on the TDI still
        u = stills.set_index("png_path").loc["/stills/d2/u.png"]
        self.assertAlmostEqual(u.y_MV_E, 0.7)
        self.assertNotIn("MV_E/E'", json.loads(u.signed_values))
        self.assertIsNone(bsm.target_of("MV", "E'"))

    def test_workers_give_the_same_table(self):
        rows = pd.concat([meas_table()] * 1, ignore_index=True)
        many = pd.DataFrame([meas_row(f"p{i}", f"d{i % 7}", "AV", "VMAX", 2.0 + (i % 5) / 10, "m/s")
                             for i in range(2500)])
        rows = pd.concat([rows, many], ignore_index=True)
        one, d1 = bsm.aggregate_stills(rows)
        three, d3 = bsm.aggregate_stills(rows, workers=3)
        pd.testing.assert_frame_equal(one.reset_index(drop=True), three.reset_index(drop=True))
        self.assertEqual(d1, d3)


def fake_headers(table):
    """A `read_header` over made-up headers: {dcm_path: header}."""
    def read(path):
        if path not in table:
            raise OSError("unreadable")
        return table[path]
    return read


def header(uid, patient, sex="F", birth="19700101", date="20250301", weight="70", size="1.70"):
    return {"StudyInstanceUID": uid, "PatientID": patient, "PatientSex": sex, "PatientBirthDate": birth,
            "StudyDate": date, "PatientWeight": weight, "PatientSize": size}


class TestYearsOutsideTheMetadata(unittest.TestCase):
    """2025 is not in the video metadata: its stills link through their own DICOM headers, and a temporal year that
    does not link stops the build instead of leaving an empty temporal test."""

    def late_rows(self):
        return pd.DataFrame([{**meas_row("x", "Q001/2.25.1", "AV", "VMAX", 3.0, "m/s"), "year": 2025},
                             {**meas_row("y", "Q001/2.25.1", "TR", "VMAX", 2.8, "m/s"), "year": 2025},
                             {**meas_row("z", "P002/2.25.2", "AV", "VMAX", 2.0, "m/s"), "year": 2025},
                             {**meas_row("w", "Q003/2.25.3", "AV", "VMAX", 2.0, "m/s"), "year": 2025},
                             {**meas_row("v", "Q003/2.25.3", "AV", "VMAX", 2.1, "m/s"), "year": 2025}])

    def headers(self):
        return {"/dicom/Q001/2.25.1/x.dcm": header("2.25.1", "Q001"),
                "/dicom/P002/2.25.2/z.dcm": header("2.25.2", "0P002", sex="M", birth="19800101"),
                "/dicom/Q003/2.25.3/v.dcm": header("2.25.3", "Q004")}         # header disagrees with the folder

    def test_header_metadata_links_and_keeps_the_ambiguity_rules(self):
        rows = pd.concat([meas_table(), self.late_rows()], ignore_index=True)
        stills, _ = bsm.aggregate_stills(rows)
        meta = metadata_table()
        missing = stills.dcm_path[~bsm.study_folder(stills.dcm_path).isin(set(bsm.study_folder(meta.dcm_path)))]
        extra, counts = bsm.header_metadata(missing, read=fake_headers(self.headers()))
        self.assertEqual(counts, {"folders": 3, "unreadable": 0, "layout_mismatch": 1})     # one file per folder
        self.assertEqual(list(extra.columns), [*bsm.METADATA_COLUMNS, "link_source"])
        both = pd.concat([meta.assign(link_source="metadata"), extra], ignore_index=True)
        linked, demo, unlinked = bsm.link_studies(stills, both, "salt")
        by_png = linked.set_index("png_path")
        self.assertEqual(by_png.loc["/stills/Q001/2.25.1/x.png", "study_id"], "2.25.1")
        self.assertEqual(by_png.loc["/stills/Q001/2.25.1/x.png", "link_source"], "dicom_header")
        self.assertEqual(by_png.loc["/stills/d1/a.png", "link_source"], "metadata")
        self.assertNotIn("/stills/Q003/2.25.3/w.png", by_png.index)          # header disagrees: left out
        self.assertEqual(unlinked, 3)                                # d5 (two studies) and n3's two stills
        d = demo.set_index("study_id")
        self.assertAlmostEqual(d.loc["2.25.2", "age_years"], 45.2, places=1)
        self.assertAlmostEqual(d.loc["2.25.1", "bsa_m2"], 1.82, places=2)
        rates = bsm.link_rates(stills, linked, set(bsm.study_folder(meta.dcm_path)))
        self.assertEqual(rates["2025"]["rate_metadata_only"], 0.0)
        self.assertEqual((rates["2025"]["linked"], rates["2025"]["stills"]), (3, 5))
        self.assertEqual(rates["2025"]["via_dicom_header"], 3)
        # The 2025 study of a patient seen earlier (P002) shares that patient's split.
        splits = pd.DataFrame({"study_id": ["1.2.3.2"], "split": ["train"]})
        out, _ = bsm.assign_splits(linked, splits, both, "salt", (0.8, 0.1, 0.1), [2025])
        self.assertNotIn("/stills/P002/2.25.2/z.png", set(out.png_path))      # trained patient: not in the temporal test

    def test_unreadable_header_is_counted_not_named(self):
        extra, counts = bsm.header_metadata(["/dicom/P9/n9/a.dcm"], read=fake_headers({}))
        self.assertEqual((len(extra), counts["unreadable"]), (0, 1))

    def test_a_temporal_year_that_does_not_link_stops_the_build(self):
        rates = {"2025": {"stills": 10, "linked": 0, "rate": 0.0}, "2024": {"stills": 5, "linked": 5, "rate": 1.0}}
        with self.assertRaisesRegex(SystemExit, "2025.*0 of 10"):
            bsm.check_temporal_links(rates, [2025], 0.5)
        with self.assertRaisesRegex(SystemExit, "no stills"):
            bsm.check_temporal_links(rates, [2026], 0.5)
        bsm.check_temporal_links(rates, [2024], 0.5)
        bsm.check_temporal_links({"2025": {"stills": 10, "linked": 6, "rate": 0.6}}, [2025], 0.5)

    def test_end_to_end_with_and_without_the_fallback(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        meas, meta = os.path.join(root, "meas.parquet"), os.path.join(root, "meta.parquet")
        splits, salt_file = os.path.join(root, "studies.csv"), os.path.join(root, "salt.txt")
        pd.concat([meas_table(), self.late_rows()], ignore_index=True).to_parquet(meas)
        metadata_table().to_parquet(meta)
        pd.DataFrame({"study_id": ["1.2.3.1"], "split": ["train"]}).to_csv(splits, index=False)
        with open(salt_file, "w") as f:
            f.write("k" * 40 + "\n")
        base = ["--meas", meas, "--metadata", meta, "--splits", splits, "--salt-file", salt_file,
                "--temporal-test-years", "2025"]
        with self.assertRaisesRegex(SystemExit, "Temporal test year 2025"):
            bsm.main(base + ["--no-header-fallback", "--out-dir", os.path.join(root, "a")])
        self.assertFalse(os.path.exists(os.path.join(root, "a", "stills.csv")))
        original = bsm.read_header
        bsm.read_header = fake_headers(self.headers())
        try:
            bsm.main(base + ["--out-dir", os.path.join(root, "b")])
        finally:
            bsm.read_header = original
        info = json.load(open(os.path.join(root, "b", "build_info.json")))
        self.assertEqual(info["link_rate_by_year"]["2025"]["linked"], 3)
        self.assertEqual(info["header_fallback"], {"folders": 3, "unreadable": 0, "layout_mismatch": 1})
        stills = pd.read_csv(os.path.join(root, "b", "stills.csv"), dtype=str)
        self.assertGreater((stills.split == bsm.TEMPORAL_SPLIT).sum(), 0)
        # Paths hold the patient folder by design (the table stays on the NAS); no other column holds an id.
        text = stills.drop(columns=["png_path", "dcm_path"]).to_csv(index=False)
        text += open(os.path.join(root, "b", "demographics.csv")).read()
        for secret in ("Q001", "Q003", "P002", "19700101", "19800101"):
            self.assertNotIn(secret, text)


class TestSalt(unittest.TestCase):

    def test_no_default_and_never_in_the_repository(self):
        with self.assertRaises(SystemExit):
            bsm.read_salt(None, environ={})
        with self.assertRaises(SystemExit):
            bsm.read_salt(None, environ={bsm.SALT_ENV: "short"})
        self.assertEqual(bsm.read_salt(None, environ={bsm.SALT_ENV: "s" * 40}), "s" * 40)
        with self.assertRaises(SystemExit):
            bsm.read_salt(os.path.join(REPO, "salt.txt"))
        self.assertNotEqual(bsm.salt_check("a" * 40), bsm.salt_check("b" * 40))
        self.assertNotIn("a" * 8, bsm.salt_check("a" * 40))


class TestMain(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_end_to_end(self):
        meas, meta = os.path.join(self.root, "meas_2024.parquet"), os.path.join(self.root, "meta.parquet")
        splits = os.path.join(self.root, "studies.csv")
        meas_table().to_parquet(meas)
        metadata_table().to_parquet(meta)
        pd.DataFrame({"study_id": ["1.2.3.1"], "split": ["train"]}).to_csv(splits, index=False)
        echonet = os.path.join(self.root, "echonet.parquet")
        pd.DataFrame({"png_path": ["/stills/d1/a.png"], "pred_cms": [310.0], "heat_max": [0.9]}).to_parquet(echonet)
        out = os.path.join(self.root, "out")
        salt_file = os.path.join(self.root, "salt.txt")
        with open(salt_file, "w") as f:
            f.write("k" * 40 + "\n")
        with self.assertRaises(SystemExit):                  # no salt, no build
            bsm.main(["--meas", meas, "--metadata", meta, "--splits", splits, "--out-dir", out])
        bsm.main(["--meas", meas, "--metadata", meta, "--splits", splits, "--echonet", echonet, "--out-dir", out,
                  "--salt-file", salt_file])
        stills = pd.read_csv(os.path.join(out, "stills.csv"), dtype={"study_id": str})
        demo = pd.read_csv(os.path.join(out, "demographics.csv"), dtype={"study_id": str})
        self.assertEqual(list(stills.columns), list(bsm.STILL_COLUMNS))
        self.assertEqual(len(stills), 5)
        a = stills.set_index("png_path").loc["/stills/d1/a.png"]
        self.assertEqual((a.echonet_present, a.echonet_vmax_ms), (1, 3.1))
        text = open(os.path.join(out, "stills.csv")).read() + open(os.path.join(out, "demographics.csv")).read()
        for secret in ("P001", "P002", "19500615", "19800101", "extractor-d1"):
            self.assertNotIn(secret, text)
        self.assertNotIn("birthdate", demo.columns)
        info = json.load(open(os.path.join(out, "build_info.json")))
        self.assertEqual((info["stills"], info["unlinked_stills"]), (5, 1))
        self.assertNotIn("salt", info)
        self.assertNotIn("k" * 40, json.dumps(info))
        self.assertEqual(info["salt_check"], bsm.salt_check("k" * 40))
        self.assertEqual(info["magnitude"], [])
        with self.assertRaises(SystemExit):
            bsm.main(["--meas", meas, "--metadata", meta, "--splits", splits, "--salt-file", salt_file, "--out-dir",
                      os.path.join(REPO, "tmp_stills_out")])


if __name__ == "__main__":
    unittest.main()
