# tests/data/supervision_faults.py

"""Fault injection for the supervised workers of build_manifests.py and scan_acquisition.py, run by
tests/data/test_supervision.py in a subprocess under a hard deadline (not a test module itself):

    python tests/data/supervision_faults.py <spec.json>

The fault is chosen by the file's name: `crash1_` (the worker process exits on the first attempt),
`crash_` (on every attempt), `stall1_` / `stall_` (sleeps 600 s on the first / every attempt),
`raise_` (raises), `flaky1_` (the decode fails on the first attempt only). Every attempt appends one
byte to `<file>.attempts`, so the test can count them. Synthetic files only.
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "data"))

import build_manifests as bm  # noqa: E402
import scan_acquisition as sa  # noqa: E402

# The real functions, kept before `main` patches the modules (worker processes never run `main`).
REAL_FINGERPRINT, REAL_SCAN = bm.fingerprint_and_validate, sa.scan_file


def attempt(path):
    with open(path + ".attempts", "a") as f:
        f.write("x")
    return os.path.getsize(path + ".attempts")


def fault(path):
    """Act out the fault the file's name asks for; True when the decode should fail this time."""
    name, n = os.path.basename(path), attempt(path)
    if name.startswith("crash_") or (name.startswith("crash1_") and n == 1):
        os._exit(17)
    if name.startswith("stall_") or (name.startswith("stall1_") and n == 1):
        time.sleep(600)
    if name.startswith("raise_"):
        raise ValueError("injected")
    return name.startswith("flaky1_") and n == 1


def fault_validate(path):
    """A stand-in verifier: every file decodes (20 frames at 25 fps) unless its fault says otherwise."""
    return bm.Check("undecodable") if fault(path) else bm.Check(None, 20, 25.0)


def faulty_fingerprint(path, validate=bm.verify_video):
    if fault(path):
        st = os.stat(path)
        return (st.st_size, st.st_mtime_ns), bm.Check("undecodable")
    return REAL_FINGERPRINT(path, validate)


def faulty_scan(path, **kwargs):
    fault(path)
    return REAL_SCAN(path, **kwargs)


def verdicts_json(verdicts):
    """{video_path: [reason, n_frames, fps]} with None for missing values."""
    return {p: [None if bm.pd.isna(r.reason) else str(r.reason), None if bm.pd.isna(r.n_frames) else int(r.n_frames),
                None if bm.pd.isna(r.fps) else float(r.fps)] for p, r in verdicts.iterrows()}


def main():
    with open(sys.argv[1]) as f:
        spec = json.load(f)
    bm.VERIFY_RETRY_DELAY = spec.get("retry_delay", 0.1)
    if spec["run"] == "build":  # the real CLI, the real decoder for files without a fault
        bm.fingerprint_and_validate = faulty_fingerprint
        sys.argv = ["build_manifests.py", *spec["argv"]]
        bm.main()
    elif spec["run"] == "scan":  # the real CLI, the real header reader for files without a fault
        sa.scan_file = faulty_scan
        sa.main(spec["argv"])
    else:
        validate = bm.verify_video if spec.get("validate") == "real" else fault_validate
        try:
            verdicts, stats = bm.verify_archive(spec["paths"], validate, spec.get("cache"), **spec["kwargs"])
            out = {"verdicts": verdicts_json(verdicts), "stats": stats}
        except Exception as e:
            out = {"error": type(e).__name__, "message": str(e)}
        print("RESULT " + json.dumps(out, default=str))


if __name__ == "__main__":
    main()
