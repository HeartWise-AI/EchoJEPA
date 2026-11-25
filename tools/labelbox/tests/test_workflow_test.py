import importlib.util
import json
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools.labelbox.pilot_upload import (
    CLIPS,
    ITEMS,
    STUDY_KEY_PREFIX,
    TEST_KEY_PREFIX,
    WORKSPACES,
    preview,
    read_plan,
    sha256,
)
from tools.labelbox.workflow_test import (
    COPIES,
    PAGE,
    STUDY,
    _metadata,
    make_copies,
    make_items,
    probe,
    read_pilot,
    report_records,
    study_key,
    study_table,
    to_test_key,
)

HAS_MEDIA = all(importlib.util.find_spec(name) for name in ("imageio_ffmpeg", "pyarrow"))
STUDY_ID = "1.2.840.113619.2.55.3.1234567"
OTHER_STUDY_ID = "1.2.840.113619.2.55.3.7654321"
REPORTS = pd.DataFrame([
    {"AccessionNumber": "'ACC100'",
     "Left ventricule findings": "Dysfonction légère du ventricule gauche. // Ventricule gauche non-dilaté",
     "Mitral valve findings": "Insuffisance mitrale légère", "Visually Estimated EF": 48.0},
    {"AccessionNumber": "ACC200", "Left ventricule findings": None, "Mitral valve findings": None,
     "Visually Estimated EF": float("nan")},
])


def pilot_key(char):
    return "radar-pilot-" + char * 24


def make_video(path, frames=12):
    """A 4:4:4 H.264 clip of odd size (65 x 49), like the archive's format, carrying identifier-like tags the copy
    must drop. The crop follows the conversion to 4:4:4, which alone can hold an odd side."""
    import imageio_ffmpeg

    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc2=size=66x50:rate=25", "-vf", "format=yuv444p,crop=65:49:0:0",
                    "-frames:v", str(frames), "-c:v", "libx264",
                    "-pix_fmt", "yuv444p", "-metadata", "title=DOE^JANE", "-metadata", "comment=MRN 0000000",
                    "-metadata:s:v:0", "handler_name=PATIENT HANDLER", str(path)], check=True, capture_output=True)


def make_pilot(root: Path, videos=None):
    """A pilot folder as `radar_pilot.py shortlist` writes it: two studies, the first with two caption review clips
    and one OTHER clip."""
    videos = videos or {}
    pilot = root / "pilot"
    pilot.mkdir()
    pd.DataFrame([
        {"category": "valve_disease", "rank": "7", "study_id": STUDY_ID, "patient_id": "P1", "accession": "ACC100"},
        {"category": "normal", "rank": "0", "study_id": OTHER_STUDY_ID, "patient_id": "P2", "accession": "ACC200"},
    ]).to_csv(pilot / "shortlist.csv", index=False)
    clip = dict(patient_id="P1", study_id=STUDY_ID, dicom_path="", vendor="GE", model="Vivid E95", n_frames="12")
    pd.DataFrame([
        {**clip, "view": "A4C", "acq_category": "bmode", "form": "column_review", "global_key": pilot_key("a"),
         "video_path": str(videos.get("a", "/archive/a.mp4"))},
        {**clip, "view": "A4C", "acq_category": "color_flow", "form": "column_review", "global_key": pilot_key("b"),
         "video_path": str(videos.get("b", "/archive/b.mp4"))},
        {**clip, "view": "OTHER", "acq_category": "bmode", "form": "view_type_review", "global_key": pilot_key("c"),
         "video_path": str(videos.get("c", "/archive/c.mp4"))},
        {**clip, "patient_id": "P2", "study_id": OTHER_STUDY_ID, "view": "PSAX", "acq_category": "bmode",
         "form": "column_review", "global_key": pilot_key("d"), "video_path": "/archive/d.mp4"},
    ]).to_csv(pilot / "shortlist_clips.csv", index=False)
    (pilot / ".global_key_secret").write_text("ab" * 32)
    reports = root / "reports.parquet"
    if HAS_MEDIA:
        REPORTS.to_parquet(reports)
        (pilot / "shortlist.json").write_text(json.dumps({"inputs": {"reports": sha256(reports)}}))
    return pilot, reports


class KeyTests(unittest.TestCase):
    def test_test_keys_keep_the_digest_under_the_test_prefix(self):
        self.assertEqual(to_test_key(pilot_key("a")), TEST_KEY_PREFIX + "a" * 24)
        for wrong in ("radar-pilot-short", TEST_KEY_PREFIX + "a" * 24, "1.2.840.1"):
            with self.assertRaises(ValueError):
                to_test_key(wrong)

    def test_study_key_is_an_hmac_of_the_study_id(self):
        key = study_key(b"secret", STUDY_ID)
        self.assertTrue(key.startswith(STUDY_KEY_PREFIX))
        self.assertNotIn("1234567", key)
        self.assertEqual(key, study_key(b"secret", STUDY_ID))
        self.assertNotEqual(key, study_key(b"another secret", STUDY_ID))


class TableTests(unittest.TestCase):
    def test_metadata_entries_are_read_per_block(self):
        text = ("Input #0, mov,mp4, from 'x.mp4':\n  Metadata:\n    major_brand     : isom\n"
                "    title           : DOE^JANE\n  Duration: 00:00:01.00, start: 0.000000\n"
                "  Stream #0:0[0x1](und): Video: h264 (High), yuv420p(progressive), 66x50\n"
                "      Metadata:\n        handler_name    : VideoHandler\n")
        self.assertEqual(_metadata(text), {"major_brand": "isom", "title": "DOE^JANE", "handler_name": "VideoHandler"})

    def test_report_rows_are_matched_by_normalized_accession(self):
        records = report_records(REPORTS, ["ACC100", "ACC200"])
        self.assertEqual(records["ACC100"]["Visually Estimated EF"], 48.0)
        self.assertIsNone(records["ACC200"]["Visually Estimated EF"])
        self.assertIsNone(records["ACC200"]["Mitral valve findings"])
        with self.assertRaises(SystemExit):
            report_records(REPORTS, ["ACC300"])
        with self.assertRaises(SystemExit):
            report_records(pd.concat([REPORTS, REPORTS]), ["ACC100"])

    def test_study_table_counts_what_a_test_covers(self):
        with tempfile.TemporaryDirectory() as root:
            studies, clips = read_pilot(make_pilot(Path(root))[0])
        records = report_records(REPORTS, ["ACC100", "ACC200"])
        rows = {row["study"]: row for row in study_table(studies, clips, records)}

        self.assertEqual(rows["valve_disease-7"], {
            "study": "valve_disease-7", "clips": 3, "caption": 2, "bmode": 1, "colour": 1, "view_type": 1, "views": 1,
            "core_views": 1, "prefilled_clips": 2, "findings_median": rows["valve_disease-7"]["findings_median"],
            "findings_max": rows["valve_disease-7"]["findings_max"]})
        self.assertGreater(rows["valve_disease-7"]["findings_max"], 0)
        self.assertEqual(rows["normal-0"]["prefilled_clips"], 0)  # PSAX: no prefill without its slice level.


@unittest.skipUnless(HAS_MEDIA, "imageio-ffmpeg or pyarrow not installed")
class CopiesAndItemsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import imageio_ffmpeg

        cls.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cls.root = Path(tempfile.mkdtemp())
        archive = cls.root / "archive"
        archive.mkdir()
        videos = {char: archive / f"{char}.mp4" for char in "abc"}
        for path in videos.values():
            make_video(path)
        cls.pilot, cls.reports = make_pilot(cls.root, videos)
        cls.out = cls.root / "test"
        cls.records = make_copies(str(cls.pilot), "valve_disease-7", str(cls.out))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    def test_copies_are_playable_complete_and_without_metadata(self):
        self.assertEqual([r["global_key"] for r in self.records],
                         [to_test_key(pilot_key(char)) for char in "abc"])
        for record in self.records:
            path = self.out / CLIPS / f"{record['global_key']}.mp4"
            info = probe(self.ffmpeg, str(path))
            self.assertEqual((info["codec"], info["pix_fmt"], info["streams"]), ("h264", "yuv420p", 1))
            self.assertEqual(info["size"], (66, 50))  # 65 x 49 padded to even sides.
            self.assertNotIn("title", info["metadata"])
            self.assertNotIn("comment", info["metadata"])
            self.assertEqual(info["metadata"].get("handler_name"), "VideoHandler")
            self.assertEqual(record["frames"], 12)
            self.assertEqual(record["copy_sha256"], sha256(path))
            self.assertNotIn(b"DOE^JANE", path.read_bytes())
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.out.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((self.out / COPIES).stat().st_mode), 0o600)

    def test_page_and_record_name_no_study_or_path(self):
        page = (self.out / PAGE).read_text()
        record = (self.out / STUDY).read_text()
        for text in (page, record, (self.out / COPIES).read_text()):
            self.assertNotIn(STUDY_ID, text)
            self.assertNotIn(str(self.root / "archive"), text)
            self.assertNotIn("radar-pilot-", text)
        self.assertEqual(page.count('type="checkbox"'), 3)
        self.assertTrue(json.loads(record)["study_key"].startswith(STUDY_KEY_PREFIX))

    def test_copies_refuse_a_used_folder_or_an_unknown_study(self):
        with self.assertRaisesRegex(SystemExit, "new or empty folder"):
            make_copies(str(self.pilot), "valve_disease-7", str(self.out))
        with self.assertRaisesRegex(SystemExit, "No shortlisted study"):
            make_copies(str(self.pilot), "valve_disease-9", str(self.root / "unknown"))

    def picked(self, *chars):
        path = self.root / f"picked-{''.join(chars)}.csv"
        pd.DataFrame({"global_key": [to_test_key(pilot_key(char)) for char in chars]}).to_csv(path, index=False)
        return str(path)

    def test_items_feed_the_uploader(self):
        summary = make_items(str(self.pilot), str(self.out), self.picked("a", "c"), str(self.reports))

        self.assertEqual((summary["candidates"], summary["view_type_items"]), (1, 1))
        self.assertGreater(summary["prefilled_columns"], 0)
        plan = read_plan(self.out)
        self.assertEqual([row.form for row in plan.rows], ["column_review", "view_type_review"])
        self.assertEqual(len({row.study_key for row in plan.rows}), 1)
        sent = json.dumps(preview(plan, WORKSPACES["test"]), ensure_ascii=False)
        self.assertNotIn("Dysfonction", sent)
        self.assertNotIn(STUDY_ID, sent)
        self.assertEqual(stat.S_IMODE((self.out / ITEMS / "candidates.json").stat().st_mode), 0o600)

    def test_items_refuse_unknown_repeated_or_changed_picks(self):
        unknown = self.root / "picked-unknown.csv"
        pd.DataFrame({"global_key": [to_test_key(pilot_key("d"))]}).to_csv(unknown, index=False)
        with self.assertRaisesRegex(SystemExit, "not copies of this study"):
            make_items(str(self.pilot), str(self.out), str(unknown), str(self.reports))
        with self.assertRaisesRegex(SystemExit, "each once"):
            make_items(str(self.pilot), str(self.out), self.picked("a", "a"), str(self.reports))
        other_reports = self.root / "other_reports.parquet"
        REPORTS.iloc[:1].to_parquet(other_reports)
        with self.assertRaisesRegex(SystemExit, "not the report file"):
            make_items(str(self.pilot), str(self.out), self.picked("a"), str(other_reports))

    def test_items_refuse_a_copy_changed_since_copies(self):
        out = self.root / "changed"
        shutil.copytree(self.out, out)
        (out / CLIPS / f"{to_test_key(pilot_key('b'))}.mp4").write_bytes(b"changed")
        with self.assertRaisesRegex(SystemExit, "missing or changed"):
            make_items(str(self.pilot), str(out), self.picked("b"), str(self.reports))


if __name__ == "__main__":
    unittest.main()
