"""The RADAR ontology, mapping, review and documentation files carry no patient data or credentials.

Scanned: the files under configs/radar, docs/radar, app/radar and tests/radar, the review form builder and its test.
This file is left out: it holds the patterns and the synthetic strings that check them. The rules target patient
identifiers (DICOM UIDs, long digit runs such as record or accession numbers, e-mail addresses outside the reserved
example domains), clinical dates (allowed only in the approval records and the approval test, for the approval
itself), patient-file paths and names, report-style text and credentials. Concept ids, hashes and repository paths are
allowed.
"""

import fnmatch
import os
import re
import unittest

from app.radar.ontology import REPO_ROOT

SCANNED_DIRS = ("configs/radar", "docs/radar", "app/radar", "tests/radar")
SCANNED_FILES = ("tools/labelbox/review_ontology.py", "tools/labelbox/tests/test_review_ontology.py")
SUFFIXES = (".py", ".md", ".yaml", ".json")
DATES_ALLOWED = ("docs/radar/approval_v*.md", "tests/radar/test_approval.py")
THIS_FILE = os.path.relpath(os.path.abspath(__file__), REPO_ROOT)

RULES = {
    "DICOM UID": re.compile(r"(?<![\w.])\d+(?:\.\d+){4,}(?![\w.])"),
    "long digit run": re.compile(r"(?<![0-9A-Za-z])\d{7,}(?![0-9A-Za-z])"),
    "e-mail address": re.compile(r"[\w.+-]+@(?![\w.-]*\b(?:example|invalid)\b)[\w-]+\.[A-Za-z]{2,}"),
    "date": re.compile(r"(?<!\d)(?:(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])"
                       r"|(?:0[1-9]|[12]\d|3[01])[-/.](?:0[1-9]|1[0-2])[-/.](?:19|20)\d{2})(?!\d)"),
    "patient-file path": re.compile(r"(?<![\w.$~])/(?:media|mnt|data|volume|home|root|Users|nas)/[\w./-]+"),
    "video or DICOM file name": re.compile(r"[\w-]+\.(?:dcm|dicom|avi|mp4|mov|nii)\b", re.I),
    "report-style text": re.compile(r"\[SEP\]|(?:\b[A-Z]{2,}[ ,.;:']+){6,}"),
    "credential": re.compile(r"(?i)(?:api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*['\"]?[\w.\-]{12,}"
                             r"|\beyJ[\w-]{10,}\.[\w-]{10,}"),
}


def scanned_paths() -> list[str]:
    paths = [p for p in SCANNED_FILES if os.path.exists(os.path.join(REPO_ROOT, p))]
    for top in SCANNED_DIRS:
        for folder, dirs, files in os.walk(os.path.join(REPO_ROOT, top)):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            paths += [os.path.relpath(os.path.join(folder, f), REPO_ROOT) for f in files if f.endswith(SUFFIXES)]
    return sorted(p for p in set(paths) if p != THIS_FILE)


def findings(path: str, text: str) -> list[str]:
    """`path:line: rule` for every match; never the matched text itself."""
    out = []
    for rule, pattern in RULES.items():
        if rule == "date" and any(fnmatch.fnmatch(path, pattern) for pattern in DATES_ALLOWED):
            continue
        for m in pattern.finditer(text):
            out.append(f"{path}:{text.count(chr(10), 0, m.start()) + 1}: {rule}")
    return out


class TestPrivacy(unittest.TestCase):
    def test_scanned_files_are_clean(self):
        paths = scanned_paths()
        self.assertIn("configs/radar/labelbox/radar-caption-ontology-v1.json", paths)
        self.assertIn("docs/radar/labeling_instructions_v1.fr.md", paths)
        problems = []
        for path in paths:
            with open(os.path.join(REPO_ROOT, path), encoding="utf-8") as f:
                problems += findings(path, f.read())
        self.assertEqual(problems, [])

    def test_rules_catch_synthetic_identifiers(self):
        caught = {
            "DICOM UID": "study 1.2.840.10008.5.1.4",
            "long digit run": "dossier 0012345678",
            "e-mail address": "contact someone@hospital.org",
            "date": "examen du 14/03/2019",
            "patient-file path": "see /media/data1/datasets/echo/clip",
            "video or DICOM file name": "clip_0001.avi",
            "report-style text": "THE LEFT VENTRICULAR EJECTION FRACTION IS SEVERELY REDUCED.",
            "credential": "LABELBOX_API_KEY = abcdefghijklmnop1234",
        }
        for rule, text in caught.items():
            self.assertTrue(RULES[rule].search(text), rule)

    def test_rules_allow_ontology_content(self):
        allowed = ("`mitral_regurgitation`", "A4C_ZOOM, PLAX_DEEP, PSAX_AV", "radar-caption-ontology-v1",
                   "sha256: " + "0123456789abcdef" * 4, "labelbox 7.12.0", "configs/radar/labelbox/x.json",
                   "$HOME/.venvs/echojepa-labelbox", "https://github.com/HeartWise-AI/EchoJEPA/issues/21",
                   "user.email=test@example.invalid", "tokenizer: local snapshot only",
                   "Coupe apicale 4 cavités (A4C), Doppler couleur. Insuffisance mitrale : modérée.")
        for text in allowed:
            self.assertEqual(findings("docs/radar/example.md", text), [], text)
        self.assertEqual(findings("docs/radar/approval_v1.md", "date: 2026-10-20"), [])
        self.assertTrue(findings("docs/radar/ontology_v1.md", "date: 2026-10-20"))


if __name__ == "__main__":
    unittest.main()
