# evals/video_classification_frozen/provenance.py

"""Utilities for recording and validating frozen-probe experiment provenance.

Tracks input checksums, code version, a wandb-safe configuration, and the video-to-study index used
during evaluation. This module does not send data to wandb; `eval.py` controls what is logged.
"""

import hashlib
import os
import subprocess

import pandas as pd

# Finds the repository root, to be later used by `git_state()`.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# wandb settings for the probe: deliberately prevent wandb from automatically uploading potentially
# sensitive local-environment information. The commit is recorded in the run's config instead.
PRIVATE_WANDB_SETTINGS = {
    "console": "off",
    "x_disable_meta": True,
    "disable_git": True,
    "save_code": False,
    "disable_code": True,
    "x_save_requirements": False,
}

INDEX_COLUMNS = ("split", "video_path", "study_id", "patient_id")


def sha256(path, chunk=1 << 20):
    """Computes SHA-256 checksum of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(encoder_checkpoint, manifests, normalization, settings):
    """Build a fingerprint that uniquely identifies a probe experiment: the SHA-256 hashes of the encoder
    checkpoint and manifests (`None` if not given), the EF normalization, and the settings that affect
    training and evaluation.
    
    The fingerprint is saved with the probe heads and must match when the probe is resumed or evaluated.
    """
    return {
        "encoder_sha256": sha256(encoder_checkpoint),
        "manifests_sha256": {name: sha256(path) if path else None for name, path in manifests.items()},
        "normalization": dict(normalization),
        "settings": settings,
    }


def differences(saved, current, ignore=()):
    """Compares two fingerprints recursively, leaving out `ignore`: some fields may intentionally be allowed to differ."""
    found = []

    def walk(a, b, name):
        if name in ignore:
            return
        # If both values are dictionaries, recursively checks every key (can catch a key that exist only in one config).
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                walk(a.get(key), b.get(key), f"{name}.{key}" if name else key)
        elif a != b:
            found.append(name)

    walk(saved, current, "")
    return found


def check_fingerprint(saved, current, what, ignore=()):
    """Verify that a probe is resumed or evaluated with the same setup it was trained with.

    Raises an error if the encoder, data, or relevant settings differ from the saved fingerprint.
    `what` identifies the probe in error messages.
    """
    if saved is None:
        raise ValueError(f"{what} has no fingerprint (it predates them), so this run cannot be checked "
                         "against it: start the probe in a new folder.")
    diffs = differences(saved, current, ignore)
    if diffs:
        raise ValueError(f"{what} was trained with another {', '.join(diffs)} than this config gives: "
                         "a probe is continued and tested only as it was trained. Start a new folder "
                         "for a new probe.")


def git_state():
    """Return the git commit and whether the working tree has tracked changes.

    Returns `("unknown", None)` when the repository state cannot be determined.
    Untracked files are ignored when determining whether the working tree is dirty.
    """
    try:
        # Gets the current git commit hash.
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        # Checks whether tracked files have local modifications.
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True).stdout
        return commit, bool(status.strip())
    # Git failure handling: if the code isn't being run inside a git repo, or git isn't installed,
    # it doesn't crash the experiment.
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None


def public_config(value):
    """Return a wandb-safe copy of the config with absolute paths reduced to file names.
    This prevents local filesystem paths from being sent to wandb."""
    # Recursive: handles nested dictionaries/lists/tuples.
    if isinstance(value, dict):
        return {k: public_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [public_config(v) for v in value]
    if isinstance(value, str) and (os.path.isabs(value) or value.startswith("~")):
        return os.path.basename(value.rstrip("/"))
    return value


def load_video_index(path):
    """Load and validate the video index produced by `data/make_probe_manifests.py`: one row per video
    with its split, study and patient, indexed by video path.

    Raises an error if a video appears more than once or if a patient appears in more than one split.
    """
    # Reads everything as strings to prevent accidentally converting numeric identifiers.
    index = pd.read_csv(path, dtype={c: str for c in INDEX_COLUMNS})
    # Required-column validation: if any required field is missing, it raises an error.
    missing = [c for c in INDEX_COLUMNS if c not in index.columns]
    if missing:
        raise ValueError(f"The video index {os.path.basename(path)} lacks the columns {missing}.")
    # Duplicate video check: a particular video must appear only once.
    if index.video_path.duplicated().any():
        raise ValueError(f"The video index lists {int(index.video_path.duplicated().sum())} videos twice.")
    # Patient leakage check.
    shared = int((index.groupby("patient_id").split.nunique() > 1).sum())
    if shared:
        raise ValueError(f"{shared} patients appear in more than one split of the video index.")
    return index.set_index("video_path")


def study_ids(index, samples, split):
    """Return the study ID for each video in `samples` using the video index.

    Raises an error if a video is missing from the index or belongs to a different split than `split`.
    """
    # Checks that every video exists.
    known = pd.Index(samples).isin(index.index)
    if not known.all():
        raise ValueError(f"{int((~known).sum())} videos of the {split} manifest are not in the video index.")
    # Checks split consistency.
    rows = index.loc[list(samples)]
    wrong = rows.split != split
    if wrong.any():
        raise ValueError(f"{int(wrong.sum())} videos of the {split} manifest are listed under another split.")
    return rows.study_id.to_numpy()


def split_counts(index):
    """Return the numbers of videos, studies, and patients in each split."""
    return {
        split: {"videos": int(len(rows)), "studies": int(rows.study_id.nunique()),
                "patients": int(rows.patient_id.nunique())}
        for split, rows in index.groupby("split", sort=True)
    }
