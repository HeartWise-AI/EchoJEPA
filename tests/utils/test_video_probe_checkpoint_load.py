# tests/utils/test_video_probe_checkpoint_load.py

"""A probe checkpoint that runs code when it is unpickled must be refused.

`check_finished`, `restore_best` and `checkpoint_reference` are what resume, `--test_only`,
validation scoring and the end of training call for `latest.pt` and `best.pt`. The multi-view
probe calls the same three. `load_checkpoint`, `run_test` and `load_selected_head` read those
files through `robust_checkpoint_loader` and must keep refusing the same file.
"""

import os
import pickle
import tempfile
import unittest
from unittest import mock

import torch

from evals.video_classification_frozen import eval as probe
from evals.video_classification_frozen import score_validation
from evals.video_classification_frozen_multi import eval as multi

RAN = []
MARKER = None


def mark_ran():
    RAN.append(True)
    with open(MARKER, "w") as handle:
        handle.write("ran")


class RunsOnLoad:
    """Unpickling it calls `mark_ran` and writes `MARKER`."""

    def __reduce__(self):
        return mark_ran, ()


class TestVideoProbeCheckpointLoad(unittest.TestCase):

    def setUp(self):
        self._saved_env = {
            name: os.environ.pop(name, None)
            for name in ("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
        }
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = os.path.join(self.tmp.name, "probe")
        os.makedirs(self.folder)
        global MARKER
        MARKER = os.path.join(self.tmp.name, "marker")
        RAN.clear()
        if os.path.exists(MARKER):
            os.remove(MARKER)

    def tearDown(self):
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def save(self, name, payload):
        path = os.path.join(self.folder, name)
        torch.save(payload, path)
        return path

    def plain(self, name):
        return self.save(name, {
            "epoch": 2,
            "best_epoch": 2,
            "best_val_acc": 0.5,
            "val_acc_per_head": [0.5],
            "weight": torch.zeros(2),
        })

    def hostile(self, name):
        return self.save(name, {
            "epoch": 2,
            "best_epoch": 2,
            "best_val_acc": 0.5,
            "val_acc_per_head": [0.5],
            "payload": RunsOnLoad(),
        })

    def assert_refused(self, call):
        RAN.clear()
        if os.path.exists(MARKER):
            os.remove(MARKER)
        try:
            call()
        except pickle.UnpicklingError:
            pass
        else:
            self.fail(f"the checkpoint load returned after running its payload: {RAN}, "
                      f"marker={os.path.exists(MARKER)}")
        self.assertEqual(RAN, [])
        self.assertFalse(os.path.exists(MARKER))

    def test_the_fixture_runs_when_pickle_is_enabled(self):
        path = self.hostile("best.pt")
        torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        self.assertEqual(RAN, [True])
        self.assertTrue(os.path.exists(MARKER))

    def test_a_tensor_checkpoint_still_loads(self):
        self.plain("latest.pt")
        best = self.plain("best.pt")
        probe.check_finished(self.folder, 2)
        probe.restore_best(self.folder, 2)
        ref = probe.checkpoint_reference(best, "regression", ["head0"], {})
        self.assertEqual(ref["epoch"], 2)
        self.assertEqual(ref["head_name"], "head0")
        self.assertFalse(os.path.exists(MARKER))

    def test_finished_check_refuses_code_in_latest(self):
        self.hostile("latest.pt")
        self.plain("best.pt")
        self.assert_refused(lambda: probe.check_finished(self.folder, 2))

    def test_finished_check_refuses_code_in_best(self):
        self.plain("latest.pt")
        self.hostile("best.pt")
        self.assert_refused(lambda: probe.check_finished(self.folder, 2))

    def test_resume_refuses_code_in_best(self):
        self.hostile("best.pt")
        self.assert_refused(lambda: probe.restore_best(self.folder, 2))

    def test_checkpoint_reference_refuses_code(self):
        path = self.hostile("best.pt")
        self.assert_refused(lambda: probe.checkpoint_reference(path, "regression", ["head0"], {}))

    def test_validation_scoring_refuses_code_in_the_probe_folder(self):
        root, encoder, index = self.scoring_files()
        folder = os.path.join(root, "video_classification_frozen")
        body = {"epoch": 2, "best_epoch": 2, "best_val_acc": 0.5, "payload": RunsOnLoad()}
        for name in ("latest.pt", "best.pt"):
            torch.save(body, os.path.join(folder, name))
        with mock.patch("src.utils.checkpoint_loader.time.sleep"):
            self.assert_refused(lambda: score_validation.score_validation(
                self.scoring_config(root, encoder, index)))

    def test_resume_loader_refuses_code_in_latest(self):
        path = self.hostile("latest.pt")
        with mock.patch("src.utils.checkpoint_loader.time.sleep"):
            self.assert_refused(lambda: probe.load_checkpoint(
                device=torch.device("cpu"), r_path=path, classifiers=[], opt=[], scaler=None))

    def test_run_test_refuses_code_in_best(self):
        path = self.hostile("best.pt")
        with mock.patch("src.utils.checkpoint_loader.time.sleep"):
            self.assert_refused(lambda: probe.run_test(
                device=torch.device("cpu"), encoder=None, classifiers=None, dataset=None, batch_size=1,
                num_workers=0, checkpoint_path=path, task_type="regression", use_bfloat16=False,
                target_mean=0.0, target_std=1.0, head_names=["head0"], folder=self.folder, rank=0,
                world_size=1))

    def test_multi_selected_head_refuses_code_in_best(self):
        path = self.hostile("best.pt")
        with mock.patch("src.utils.checkpoint_loader.time.sleep"):
            self.assert_refused(lambda: multi.load_selected_head(path, None, None))

    def test_validation_scoring_refuses_code_in_best_after_the_finished_check(self):
        root, encoder, index = self.scoring_files()
        folder = os.path.join(root, "video_classification_frozen")
        body = {"epoch": 2, "best_epoch": 2, "best_val_acc": 0.5, "val_acc_per_head": [0.5],
                "weight": torch.zeros(2)}
        for name in ("latest.pt", "best.pt"):
            torch.save(body, os.path.join(folder, name))
        best = os.path.join(folder, "best.pt")
        real_fingerprint = probe.shared_fingerprint

        def swap_then_fingerprint(*args, **kwargs):
            torch.save({"epoch": 2, "payload": RunsOnLoad()}, best)
            return real_fingerprint(*args, **kwargs)

        with mock.patch("src.utils.checkpoint_loader.time.sleep"), \
                mock.patch("evals.video_classification_frozen.score_validation.probe.shared_fingerprint",
                           side_effect=swap_then_fingerprint):
            self.assert_refused(lambda: score_validation.score_validation(self.scoring_config(root, encoder, index)))

    def scoring_files(self):
        root = os.path.join(self.tmp.name, "run")
        os.makedirs(os.path.join(root, "video_classification_frozen"))
        index = os.path.join(self.tmp.name, "video_index.csv")
        with open(index, "w") as handle:
            handle.write("split,video_path,study_id,patient_id\ntrain,v.mp4,s1,p1\n")
        encoder = os.path.join(self.tmp.name, "encoder.pt")
        with open(encoder, "wb") as handle:
            handle.write(b"encoder")
        for name in ("train.csv", "val.csv"):
            open(os.path.join(self.tmp.name, name), "w").close()
        return root, encoder, index

    def scoring_config(self, root, encoder, index):
        return {
            "eval_name": "video_classification_frozen",
            "folder": root,
            "model_kwargs": {
                "checkpoint": encoder or os.path.join(self.tmp.name, "encoder.pt"),
                "module_name": "encoder",
                "pretrain_kwargs": {},
                "wrapper_kwargs": {},
            },
            "experiment": {
                "classifier": {"task_type": "regression", "num_targets": 1},
                "data": {
                    "dataset_train": os.path.join(self.tmp.name, "train.csv"),
                    "dataset_val": os.path.join(self.tmp.name, "val.csv"),
                    "video_index": index or os.path.join(self.tmp.name, "video_index.csv"),
                    "target_mean": 50.0,
                    "target_std": 10.0,
                },
                "optimization": {
                    "num_epochs": 2,
                    "batch_size": 1,
                    "use_bfloat16": False,
                    "multihead_kwargs": [{"lr": 1e-4, "weight_decay": 0.01}],
                },
            },
        }
