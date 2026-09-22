# tests/utils/test_wandb_logging.py

"""Tests for W&B run initialization, checkpoint resume behavior, and input-clip logging."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from src.utils import wandb_logging


class _FakeRun(SimpleNamespace):
    """A wandb run that records how its x-axis was configured."""

    def define_metric(self, name, step_metric=None):
        self.defined_metrics.append((name, step_metric))


class _FakeWandb:
    def __init__(self):
        self.calls = []

    def init(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeRun(
            id="fresh-id" if kwargs["id"] is None else kwargs["id"],
            entity=kwargs["entity"],
            project=kwargs["project"],
            url="https://example.invalid/run",
            defined_metrics=[],
        )


class _FailingWandb:
    """Stands in for a wandb that cannot reach the server."""

    def __init__(self):
        self.calls = []

    def init(self, **kwargs):
        self.calls.append(kwargs)
        raise RuntimeError("network is unreachable")


class TestInitWandb(unittest.TestCase):
    def setUp(self):
        self.meta = {"wandb_project": "project", "wandb_entity": "entity"}

    def test_entity_is_required_on_nonzero_ranks_too(self):
        with self.assertRaises(ValueError):
            wandb_logging.init_wandb(
                {"wandb_project": "project"}, {}, rank=1, folder="unused"
            )

    def test_explicit_missing_checkpoint_id_does_not_use_stale_sidecar(self):
        fake_wandb = _FakeWandb()
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "wandb_run_id.txt"), "w") as id_file:
                id_file.write("stale-id")

            with mock.patch.object(wandb_logging, "wandb", fake_wandb):
                run, run_id = wandb_logging.init_wandb(
                    self.meta,
                    {},
                    rank=0,
                    folder=folder,
                    resuming_training=True,
                    checkpoint_run_id=None,
                    checkpoint_has_wandb_run_id=True,
                )

            self.assertEqual(run.id, "fresh-id")
            self.assertEqual(run_id, "fresh-id")
            self.assertIsNone(fake_wandb.calls[0]["id"])
            self.assertEqual(fake_wandb.calls[0]["resume"], "never")

    def test_fresh_run_ignores_stale_sidecar(self):
        """Jacques' case: a fresh run in a folder that still holds an old run id."""
        fake_wandb = _FakeWandb()
        with tempfile.TemporaryDirectory() as folder:
            id_path = os.path.join(folder, "wandb_run_id.txt")
            with open(id_path, "w") as id_file:
                id_file.write("old-run-id")

            with mock.patch.object(wandb_logging, "wandb", fake_wandb):
                run, run_id = wandb_logging.init_wandb(
                    self.meta,
                    {},
                    rank=0,
                    folder=folder,
                    resuming_training=False,
                )

            self.assertEqual(run.id, "fresh-id")
            self.assertEqual(run_id, "fresh-id")
            # Never reconnects to the old experiment.
            self.assertIsNone(fake_wandb.calls[0]["id"])
            self.assertEqual(fake_wandb.calls[0]["resume"], "never")
            # And the fresh id replaces the stale sidecar.
            with open(id_path) as id_file:
                self.assertEqual(id_file.read().strip(), "fresh-id")

    def test_charts_are_pinned_to_an_explicit_step_metric(self):
        """Without this, a resumed run replays steps wandb silently drops."""
        fake_wandb = _FakeWandb()
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(wandb_logging, "wandb", fake_wandb):
                run, _ = wandb_logging.init_wandb(self.meta, {}, rank=0, folder=folder)

        self.assertIn((wandb_logging.STEP_METRIC, None), run.defined_metrics)
        self.assertIn(("*", wandb_logging.STEP_METRIC), run.defined_metrics)

    def test_scalars_carry_the_step_as_a_metric_not_as_wandb_step(self):
        logged = {}

        def _log(payload, **kwargs):
            logged["payload"] = payload
            logged["kwargs"] = kwargs

        run = SimpleNamespace(log=_log)
        wandb_logging.log_scalars(run, {"train/loss": 1.5}, step=42)

        self.assertEqual(logged["payload"][wandb_logging.STEP_METRIC], 42)
        self.assertEqual(logged["payload"]["train/loss"], 1.5)
        # An explicit wandb step is what a resume cannot rewind past.
        self.assertNotIn("step", logged["kwargs"])

    def test_failed_resume_disables_logging_instead_of_killing_the_job(self):
        """A resume that cannot reattach costs logging, never the training job."""
        failing = _FailingWandb()
        with tempfile.TemporaryDirectory() as folder:
            id_path = os.path.join(folder, "wandb_run_id.txt")
            with open(id_path, "w") as id_file:
                id_file.write("existing-id")

            with mock.patch.object(wandb_logging, "wandb", failing):
                run, run_id = wandb_logging.init_wandb(
                    self.meta,
                    {},
                    rank=0,
                    folder=folder,
                    resuming_training=True,
                    checkpoint_run_id="existing-id",
                    checkpoint_has_wandb_run_id=True,
                )

            self.assertIsNone(run)
            # The caller still learns which run this job belongs to.
            self.assertEqual(run_id, "existing-id")
            # It still refused to quietly fork a new run.
            self.assertEqual(failing.calls[0]["id"], "existing-id")
            self.assertEqual(failing.calls[0]["resume"], "must")
            # And it left the sidecar alone, so a later resume can still reattach.
            with open(id_path) as id_file:
                self.assertEqual(id_file.read().strip(), "existing-id")

    def test_failed_legacy_resume_still_reports_the_sidecar_id(self):
        """The id lives only in the sidecar, so losing it here is permanent.

        The checkpoint has no `wandb_run_id` field, the sidecar has the real id,
        and wandb.init then fails. If init_wandb reported None, save_checkpoint
        would write `wandb_run_id: None` -- which the next resume reads as
        "explicitly no run" and refuses to fall back to the sidecar for.
        """
        failing = _FailingWandb()
        with tempfile.TemporaryDirectory() as folder:
            id_path = os.path.join(folder, "wandb_run_id.txt")
            with open(id_path, "w") as id_file:
                id_file.write("legacy-id")

            with mock.patch.object(wandb_logging, "wandb", failing):
                run, run_id = wandb_logging.init_wandb(
                    self.meta,
                    {},
                    rank=0,
                    folder=folder,
                    resuming_training=True,
                    checkpoint_run_id=None,          # legacy checkpoint...
                    checkpoint_has_wandb_run_id=False,  # ...with no such field
                )

            self.assertIsNone(run)
            # Recovered from the sidecar and handed back, so the next checkpoint
            # records it and the association survives.
            self.assertEqual(run_id, "legacy-id")
            self.assertEqual(failing.calls[0]["id"], "legacy-id")
            with open(id_path) as id_file:
                self.assertEqual(id_file.read().strip(), "legacy-id")

    def test_missing_wandb_on_legacy_resume_still_reports_the_sidecar_id(self):
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "wandb_run_id.txt"), "w") as id_file:
                id_file.write("legacy-id")

            with mock.patch.object(wandb_logging, "wandb", None):
                run, run_id = wandb_logging.init_wandb(
                    self.meta,
                    {},
                    rank=0,
                    folder=folder,
                    resuming_training=True,
                    checkpoint_run_id=None,
                    checkpoint_has_wandb_run_id=False,
                )

            self.assertIsNone(run)
            self.assertEqual(run_id, "legacy-id")

    def test_missing_wandb_on_resume_disables_logging_instead_of_killing_the_job(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(wandb_logging, "wandb", None):
                run, run_id = wandb_logging.init_wandb(
                    self.meta,
                    {},
                    rank=0,
                    folder=folder,
                    resuming_training=True,
                    checkpoint_run_id="existing-id",
                    checkpoint_has_wandb_run_id=True,
                )

            self.assertIsNone(run)
            self.assertEqual(run_id, "existing-id")

    def test_failed_fresh_init_remains_nonfatal(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(wandb_logging, "wandb", _FailingWandb()):
                run, run_id = wandb_logging.init_wandb(
                    self.meta,
                    {},
                    rank=0,
                    folder=folder,
                    resuming_training=False,
                )
        self.assertIsNone(run)
        # A fresh run that never opened belongs to no run at all.
        self.assertIsNone(run_id)

    def test_legacy_checkpoint_can_use_sidecar_id(self):
        fake_wandb = _FakeWandb()
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "wandb_run_id.txt"), "w") as id_file:
                id_file.write("legacy-id")

            with mock.patch.object(wandb_logging, "wandb", fake_wandb):
                run, run_id = wandb_logging.init_wandb(
                    self.meta,
                    {},
                    rank=0,
                    folder=folder,
                    resuming_training=True,
                    checkpoint_run_id=None,
                    checkpoint_has_wandb_run_id=False,
                )

            self.assertEqual(run.id, "legacy-id")
            self.assertEqual(run_id, "legacy-id")
            self.assertEqual(fake_wandb.calls[0]["id"], "legacy-id")
            self.assertEqual(fake_wandb.calls[0]["resume"], "must")

    def test_logging_is_disabled_unless_a_project_is_configured(self):
        fake_wandb = _FakeWandb()
        for meta in ({}, {"wandb_project": None}, {"wandb_project": None, "wandb_entity": "entity"}):
            with self.subTest(meta=meta), tempfile.TemporaryDirectory() as folder:
                with mock.patch.object(wandb_logging, "wandb", fake_wandb):
                    result = wandb_logging.init_wandb(meta, {}, rank=0, folder=folder)

                self.assertEqual(result, (None, None))
                # Nothing is opened and nothing is written to the run folder.
                self.assertEqual(os.listdir(folder), [])
        self.assertEqual(fake_wandb.calls, [])

    def test_only_rank_zero_opens_a_run(self):
        """Nonzero ranks must not open duplicate runs, even when resuming."""
        fake_wandb = _FakeWandb()
        for rank in (1, 3):
            with self.subTest(rank=rank), tempfile.TemporaryDirectory() as folder:
                with mock.patch.object(wandb_logging, "wandb", fake_wandb):
                    result = wandb_logging.init_wandb(
                        self.meta,
                        {},
                        rank=rank,
                        folder=folder,
                        resuming_training=True,
                        checkpoint_run_id="existing-id",
                        checkpoint_has_wandb_run_id=True,
                    )

                self.assertEqual(result, (None, None))
                self.assertEqual(os.listdir(folder), [])
        self.assertEqual(fake_wandb.calls, [])

    def test_run_landing_in_another_destination_is_reported(self):
        class _RedirectingWandb(_FakeWandb):
            def init(self, **kwargs):
                run = super().init(**kwargs)
                run.entity = "personal-account"
                return run

        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(wandb_logging, "wandb", _RedirectingWandb()):
                with self.assertLogs(level="WARNING") as logs:
                    wandb_logging.init_wandb(self.meta, {}, rank=0, folder=folder)

        self.assertTrue(any("personal-account/project" in line for line in logs.output))


def _clips(batch, frames=4, size=8, seed=0):
    """A (B, C, T, H, W) batch whose frames are all distinct."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 3, frames, size, size, generator=generator)


class TestClipHelpers(unittest.TestCase):
    def test_select_clips_takes_the_first_fpc_batch(self):
        first, second = _clips(2), _clips(2, frames=8)
        self.assertIs(wandb_logging._select_clips([first, second]), first)
        self.assertIs(wandb_logging._select_clips((first,)), first)
        self.assertIs(wandb_logging._select_clips(first), first)

    def test_select_clips_rejects_what_it_cannot_display(self):
        self.assertIsNone(wandb_logging._select_clips([]))
        self.assertIsNone(wandb_logging._select_clips(torch.zeros(2, 3, 8, 8)))
        self.assertIsNone(wandb_logging._select_clips("not a tensor"))

    def test_denormalize_recovers_the_original_pixels(self):
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        # Every uint8 value, in every channel and frame.
        pixels = torch.arange(256, dtype=torch.float32).reshape(1, 1, 1, 16, 16)
        pixels = pixels.expand(1, 3, 4, 16, 16).contiguous()
        normalized = (pixels / 255.0 - torch.tensor(mean).view(1, 3, 1, 1, 1)) / torch.tensor(std).view(
            1, 3, 1, 1, 1
        )
        before = normalized.clone()

        out = wandb_logging.denormalize_clips(normalized, (mean, std))

        self.assertEqual(out.dtype, torch.uint8)
        # (B, C, T, H, W) -> (B, T, C, H, W), the layout wandb.Video expects.
        self.assertEqual(tuple(out.shape), (1, 4, 3, 16, 16))
        self.assertTrue(torch.equal(out.permute(0, 2, 1, 3, 4), pixels.to(torch.uint8)))
        # The tensor being trained on is left untouched.
        self.assertTrue(torch.equal(normalized, before))

    def test_denormalize_clamps_out_of_range_values(self):
        clips = torch.tensor([-100.0, 100.0]).view(1, 1, 1, 1, 2).expand(1, 3, 1, 1, 2)
        out = wandb_logging.denormalize_clips(clips, ((0.5,) * 3, (0.5,) * 3))
        self.assertEqual(out.flatten()[:2].tolist(), [0, 255])

    def test_unique_frame_fraction(self):
        clips = _clips(2)
        self.assertEqual(wandb_logging.unique_frame_fraction(clips), 1.0)

        # Clip 0 repeats frame 0 once (3/4 unique); clip 1 stays fully unique.
        clips[0, :, 1] = clips[0, :, 0]
        self.assertAlmostEqual(wandb_logging.unique_frame_fraction(clips), (0.75 + 1.0) / 2)

    def test_log_input_clips_uploads_and_analyses_only_the_sample(self):
        """Repeated frames outside the uploaded sample must not affect the metric."""
        clips = _clips(4)
        clips[2:, :, 1] = clips[2:, :, 0]  # only clips that are not uploaded repeat frames
        logged = []
        fake_wandb = SimpleNamespace(Video=lambda data, fps, format: ("video", data.shape, format))
        run = SimpleNamespace(log=lambda payload, **kwargs: logged.append((payload, kwargs)))

        with mock.patch.object(wandb_logging, "wandb", fake_wandb):
            wandb_logging.log_input_clips(
                run, [clips], step=6, normalize=((0.5,) * 3, (0.5,) * 3), num_videos=2
            )

        [(payload, kwargs)] = logged
        videos = {k: v for k, v in payload.items() if k.startswith("train/input_clips/")}
        self.assertEqual(sorted(videos), ["train/input_clips/0", "train/input_clips/1"])
        self.assertEqual(videos["train/input_clips/0"], ("video", (4, 3, 8, 8), "gif"))
        self.assertEqual(payload["data/unique_frame_frac"], 1.0)
        self.assertEqual(payload[wandb_logging.STEP_METRIC], 6)
        self.assertNotIn("step", kwargs)


if __name__ == "__main__":
    unittest.main()
