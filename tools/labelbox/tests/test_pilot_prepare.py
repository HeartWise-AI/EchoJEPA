import csv
import json
import shutil
import stat
import tempfile
import unittest
from pathlib import Path

from tools.labelbox.column_review import ROOT, SPEC, build_candidate
from tools.labelbox.pilot_prepare import RECORD, prepare
from tools.labelbox.pilot_upload import (
    CLIPS,
    COLUMN_REVIEW,
    ITEMS,
    PILOT,
    PILOT_STUDY_KEY_PREFIX,
    TEST_KEY_PREFIX,
    read_plan,
    sha256,
)
from tools.labelbox.tests.test_workflow_test import HAS_MEDIA, OTHER_STUDY_ID, STUDY_ID, make_video, pilot_key
from tools.labelbox.workflow_test import study_key

REPORT = {
    "Left ventricule findings": "Dysfonction légère du ventricule gauche. // Ventricule gauche non-dilaté",
    "Mitral valve findings": "Insuffisance mitrale légère",
    "Visually Estimated EF": 48.0,
    "LA Volume Index (BP MOD) DuBois": 0.02854,
}
SECRET = "ab" * 32
KEY_FIELDS = ["global_key", "form", "study_id", "video_path", "video_sha256"]


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_pilot(root: Path, *, media: bool) -> Path:
    """A pilot folder as `radar_pilot.py candidates` leaves it: three caption review clips of two studies."""
    pilot, archive = root / "pilot", root / "archive"
    pilot.mkdir()
    archive.mkdir()
    listed, candidates = [], []
    for char, study, view in (("a", STUDY_ID, "A4C"), ("b", STUDY_ID, "PLAX"), ("c", OTHER_STUDY_ID, "A2C")):
        video = archive / f"{char}.mp4"
        if media:
            make_video(video)
        else:
            video.write_bytes(f"video {char}".encode())
        listed.append({"global_key": pilot_key(char), "form": COLUMN_REVIEW, "study_id": study,
                       "video_path": str(video), "video_sha256": sha256(video)})
        candidates.append(build_candidate(REPORT, view, "bmode", global_key=pilot_key(char), input_kind="video"))
    (pilot / "candidates.json").write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(pilot / "global_keys.csv", listed, KEY_FIELDS)
    write_csv(pilot / "view_type_items.csv", [], ["global_key"])
    summary = {"provisional_screen": False, "spec": {str(SPEC.relative_to(ROOT)): sha256(SPEC)}}
    (pilot / "candidates_summary.json").write_text(json.dumps(summary))
    (pilot / ".global_key_secret").write_text(SECRET)
    return pilot


class CheckTests(unittest.TestCase):
    """What stops the preparation before any copy is written."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.pilot = make_pilot(self.root, media=False)
        self.out = self.root / "upload"

    def tearDown(self):
        shutil.rmtree(self.root)

    def refused(self, message):
        with self.assertRaisesRegex(SystemExit, message):
            prepare(self.pilot, self.out, ffmpeg="no-ffmpeg")
        self.assertFalse(self.out.exists())

    def rewrite(self, name, change):
        path = self.pilot / name
        if name.endswith(".json"):
            path.write_text(json.dumps(change(json.loads(path.read_text()))))
        else:
            with open(path, newline="") as f:
                rows = change(list(csv.DictReader(f)))
            write_csv(path, rows, KEY_FIELDS)

    def test_refuses_a_provisional_screen(self):
        self.rewrite("candidates_summary.json", lambda summary: {**summary, "provisional_screen": True})
        self.refused("provisional thresholds")

    def test_refuses_candidates_built_from_another_specification(self):
        self.rewrite("candidates_summary.json", lambda summary: {**summary, "spec": {"configs/spec.json": "0" * 64}})
        self.refused("another column review specification")

    def test_refuses_candidates_of_another_form_version(self):
        self.rewrite("candidates.json", lambda candidates: [{**c, "version": 2} for c in candidates])
        self.refused("another column review specification")

    def test_refuses_lists_that_disagree(self):
        self.rewrite("candidates.json", lambda candidates: candidates[1:])
        self.refused("column_review items and the clips global_keys.csv lists")

    def test_refuses_keys_that_are_not_the_pilots(self):
        self.rewrite("global_keys.csv", lambda rows: [{**row, "global_key": TEST_KEY_PREFIX + row["global_key"][-24:]}
                                                      for row in rows])
        self.refused("its own pilot global key")

    def test_refuses_a_clip_replaced_since_the_screen_without_naming_it(self):
        (self.root / "archive" / "b.mp4").write_bytes(b"re-masked since")
        with self.assertRaises(SystemExit) as raised:
            prepare(self.pilot, self.out, ffmpeg="no-ffmpeg")
        self.assertIn("1 clips no longer hold the video the screen read", str(raised.exception))
        self.assertNotIn(str(self.root), str(raised.exception))
        (self.root / "archive" / "b.mp4").unlink()
        self.refused("1 clips no longer hold")

    def test_refuses_a_used_folder_or_one_in_the_repository(self):
        with self.assertRaisesRegex(SystemExit, "outside the repository"):
            prepare(self.pilot, ROOT / "pilot-upload-never-made", ffmpeg="no-ffmpeg")
        self.assertFalse((ROOT / "pilot-upload-never-made").exists())
        self.out.mkdir()
        (self.out / "earlier.txt").write_text("")
        with self.assertRaisesRegex(SystemExit, "new or empty folder"):
            prepare(self.pilot, self.out, ffmpeg="no-ffmpeg")


@unittest.skipUnless(HAS_MEDIA, "needs imageio-ffmpeg and pyarrow")
class PrepareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        cls.pilot = make_pilot(cls.root, media=True)
        cls.out = cls.root / "upload"
        cls.record = prepare(cls.pilot, cls.out)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    def test_the_uploader_reads_the_folder_as_the_pilots(self):
        plan = read_plan(self.out)

        self.assertIs(plan.kind, PILOT)
        self.assertEqual([row.global_key for row in plan.rows], [pilot_key(char) for char in "abc"])
        keys = {study: study_key(SECRET.encode(), study, PILOT_STUDY_KEY_PREFIX) for study in (STUDY_ID, OTHER_STUDY_ID)}
        self.assertEqual([row.study_key for row in plan.rows], [keys[STUDY_ID], keys[STUDY_ID], keys[OTHER_STUDY_ID]])
        self.assertEqual(sorted(len(clips) for clips in plan.batches(COLUMN_REVIEW).values()), [1, 2])
        self.assertEqual((self.record["clips"], self.record["studies"], self.record["frames"]), (3, 2, 36))
        self.assertEqual(self.record["forms"], {COLUMN_REVIEW: 3})

    def test_the_attachment_shows_numbers_in_their_unit(self):
        attachment = read_plan(self.out).rows[0].attachments[1]["value"]
        self.assertIn("LA Volume Index (BP MOD) DuBois: 28.5 mL/m²", attachment)
        self.assertNotIn("Dysfonction", attachment)

    def test_the_folder_names_no_study_or_path(self):
        written = [self.out / RECORD, self.out / ITEMS / "global_keys.csv", self.out / ITEMS / "view_type_items.csv"]
        for path in written:
            text = path.read_text()
            for private in (STUDY_ID, OTHER_STUDY_ID, str(self.root), "video_path"):
                self.assertNotIn(private, text)
        candidates = self.out / ITEMS / "candidates.json"
        self.assertEqual(sha256(candidates), sha256(self.pilot / "candidates.json"))
        self.assertEqual(self.record["pilot"]["candidates.json"], sha256(candidates))

    def test_files_are_the_owners_only(self):
        for folder in (self.out, self.out / CLIPS, self.out / ITEMS):
            self.assertEqual(stat.S_IMODE(folder.stat().st_mode), 0o700)
        for path in [self.out / RECORD, *(self.out / ITEMS).iterdir(), *(self.out / CLIPS).iterdir()]:
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
