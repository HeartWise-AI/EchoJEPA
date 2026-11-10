# tests/models/test_radar_training.py

"""RADAR training end to end on synthetic cached tokens: loader, pooler, caption student and momentum teacher, loss,
schedule and optimizer checks over several optimizer steps, in one and two CPU processes.

- Every epoch has a globally skipped step (no structure can contribute) and contributing steps, and structure
  availability changes from step to step; a skipped step changes no parameter, optimizer state or teacher.
- Training two epochs equals training one, stopping, and resuming for the second, also with dropout in the caption
  encoder (each rank gets back its own RNG states), in one process and in two; the logs of an interrupted epoch are
  dropped at resume. A resume with another config, another number of processes, or other training code is refused
  (other code is accepted with allow_code_change, and recorded).
- Two gloo processes (DDP) reproduce the single-process losses, skip decisions and final weights.
- Loader workers start by spawn (or forkserver), never by the platform's default fork, and do not change what is
  trained: two spawned workers give the zero-worker losses and weights, epoch after epoch.
- The routing registry is the only source of the view family map: the old `data.view_families` key is refused, and
  another map goes through `data.routing`, which changes the run's fingerprint.
- A0 and A1 run on flags of a provisional policy; A2 and A3 are refused until the caption table says approved.
- A captioned structure without a usable clip takes no part in training: its row is not valid, and the null token it
  falls back on changes neither the loss nor any gradient.
- Each step and epoch logs the exact-caption and normal-normal pair counts; `data.normal_fraction` lowers the latter.
- With `meta.wandb_project`, rank 0 logs each step and epoch to one W&B run, which a resume continues; the run's
  config holds no path, `meta` is outside the fingerprint, and without a project nothing is logged.
- With RADAR_CUDA_TESTS=1 on a machine with a GPU: the stochastic resume on CUDA (CUDA RNG states).
"""

import copy
import datetime
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from evals.radar_frozen import train as radar_train
from src.datasets.radar_study_dataset import RadarStudyDataset, collate, read_table
from src.datasets.radar_token_cache import TokenCache
from src.models.radar.routing import load_routing
from src.models.radar.schedule import step_schedule
from src.models.radar.text_encoder import HashTokenizer, MomentumTeacher
from src.utils import wandb_logging
from tests.models.radar_synthetic import DIM, GRID, make_synthetic, seal_captions, synthetic_config
from tests.utils.test_vjepa_2_1_wandb import FakeWandb

SHIPPED_POLICY = os.path.join(radar_train.REPO, "configs", "radar", "radar_normality_policy_v0.yaml")


def _state(summary):
    return {k: v.detach().clone() for k, v in summary["model"].state_dict().items()}


def _same(a, b, atol):
    return all(torch.allclose(a[k], b[k], rtol=0.0, atol=atol) if atol else torch.equal(a[k], b[k]) for k in a)


def _worker(rank, world_size, init_file, config, out_dir, result_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        summary = radar_train.train(config, out_dir)
        torch.save({"losses": summary["losses"], "skipped": summary["skipped_steps"], "state": _state(summary),
                    "teacher": {k: v.clone() for k, v in summary["teacher"].state_dict().items()}},
                   os.path.join(result_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _resume_worker(rank, world_size, init_file, config, root, result_dir):
    """Two epochs straight, then one epoch, a stop and a resume, in two processes."""
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(seconds=60))
    try:
        straight = radar_train.train(config, os.path.join(root, "straight"))
        first = radar_train.train(config, os.path.join(root, "resumed"), stop_after_epoch=1)
        second = radar_train.train(config, os.path.join(root, "resumed"))
        torch.save({"straight": straight["losses"], "resumed": first["losses"] + second["losses"],
                    "straight_state": _state(straight), "resumed_state": _state(second)},
                   os.path.join(result_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _steps(path):
    with open(path) as f:
        return [json.loads(line)["step"] for line in f]


class TestLearningRate(unittest.TestCase):

    def test_warmup_peak_and_last_step_floor(self):
        base, ratio = 1e-3, 1e-2
        for schedule in [step_schedule(16, 4, 2, warmup_fraction=0.25),      # the synthetic run: 8 steps, warmup 2
                         step_schedule(7000, 48, 20),                         # 2,900 steps, warmup 145
                         step_schedule(16, 4, 2, warmup_fraction=0.0),
                         {"total_steps": 3, "warmup_steps": 2},               # one step after warmup: the last
                         step_schedule(4, 4, 1, warmup_fraction=0.9)]:        # one step in all
            with self.subTest(**schedule):
                warmup, total = schedule["warmup_steps"], schedule["total_steps"]
                rates = [radar_train.learning_rate(base, k, schedule, ratio) for k in range(total)]
                self.assertAlmostEqual(rates[-1], base * ratio, places=15)
                if warmup:
                    self.assertAlmostEqual(rates[warmup - 1], base, places=15)
                    self.assertTrue(all(a < b for a, b in zip(rates[:warmup - 1], rates[1:warmup])))
                elif total > 1:
                    self.assertAlmostEqual(rates[0], base, places=15)
                after = rates[max(0, warmup - 1):]
                self.assertTrue(all(a >= b for a, b in zip(after, after[1:])))
                self.assertTrue(all(base * ratio - 1e-18 <= r <= base + 1e-18 for r in rates))


class TestRadarTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.paths = make_synthetic(os.path.join(cls.root, "data"))
        cls.config = synthetic_config(cls.paths)
        cls.straight = radar_train.train(cls.config, os.path.join(cls.root, "straight"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_several_steps_with_skipped_and_contributing_batches(self):
        s = self.straight
        self.assertEqual(s["schedule"], {"steps_per_epoch": 4, "total_steps": 8, "warmup_steps": 2})
        self.assertEqual(s["steps"], 8)
        self.assertGreaterEqual(s["skipped_steps"], 2)                 # at least one per epoch
        self.assertGreaterEqual(s["optimizer_steps"], 2)
        self.assertEqual(s["skipped_steps"] + s["optimizer_steps"], 8)
        with open(os.path.join(self.root, "straight", "log.jsonl")) as f:
            lines = [json.loads(line) for line in f]
        self.assertEqual(len(lines), 8)
        lv_rows = {line["structures"]["lv"]["rows"] for line in lines}
        self.assertGreater(len(lv_rows), 1)                            # availability changes between steps
        opt = self.config["optimization"]
        for name, lr in lines[-1]["lr"].items():                       # the last step is at the configured floor
            self.assertAlmostEqual(lr, opt["min_lr"], places=15, msg=name)
        for name, lr in lines[1]["lr"].items():                        # the end of warmup is at the base rate
            self.assertAlmostEqual(lr, opt["lr"], places=15, msg=name)
        for line in lines:
            self.assertTrue(0.001 <= line["temperature"] <= 0.5)
            if not line["contributed"]:
                self.assertTrue(all(st["skipped"] for st in line["structures"].values()))
            for st in line["structures"].values():
                self.assertTrue(0 <= st["exact_pairs"] <= st["rows"] * (st["rows"] - 1) // 2)
                self.assertTrue(0 <= st["normal_pairs"] <= st["rows"] * (st["rows"] - 1) // 2)
        self.assertGreater(max(line["structures"]["pericardium"]["normal_pairs"] for line in lines), 0)
        self.assertEqual({line["structures"]["lv"]["normal_pairs"] for line in lines}, {0})   # LV captions: unknown
        with open(os.path.join(self.root, "straight", "epochs.jsonl")) as f:
            epochs = [json.loads(line) for line in f]
        self.assertEqual([e["epoch"] for e in epochs], [0, 1])
        self.assertIsNotNone(epochs[0]["structures"]["lv"]["median_exact_pairs"])
        self.assertIsNone(epochs[0]["structures"]["pulmonic"]["median_exact_pairs"])
        self.assertEqual(epochs[0]["structures"]["lv"]["median_normal_pairs"], 0)
        self.assertIsNone(epochs[0]["structures"]["pulmonic"]["median_normal_pairs"])
        self.assertGreaterEqual(epochs[0]["structures"]["lv"]["contributing_steps"], 1)
        self.assertEqual(epochs[0]["structures"]["pulmonic"]["contributing_steps"], 0)

    def test_teacher_differs_from_student(self):
        model, teacher = self.straight["model"], self.straight["teacher"]
        student = dict(model.text.named_parameters())
        moved = [n for n, p in teacher.model.named_parameters() if not torch.allclose(p, student[n])]
        self.assertTrue(moved)

    def test_teacher_takes_one_step_per_optimizer_step(self):
        with mock.patch.object(MomentumTeacher, "update", autospec=True, side_effect=MomentumTeacher.update) as update:
            summary = radar_train.train(self.config, os.path.join(self.root, "counted"))
        self.assertEqual(update.call_count, summary["optimizer_steps"])

    def test_a_fresh_run_leaves_no_older_checkpoint_to_resume(self):
        out = os.path.join(self.root, "fresh")
        radar_train.train(self.config, out, stop_after_epoch=1)
        self.assertTrue(os.path.exists(os.path.join(out, radar_train.CHECKPOINT)))
        with mock.patch.object(radar_train, "train_step", side_effect=RuntimeError("died")):
            with self.assertRaisesRegex(RuntimeError, "died"):
                radar_train.train(self.config, out, resume=False)
        self.assertFalse(os.path.exists(os.path.join(out, radar_train.CHECKPOINT)))

    def test_no_bias_gets_weight_decay(self):
        groups = radar_train.param_groups(self.straight["model"], self.config["optimization"])
        names = {id(p): n for n, p in self.straight["model"].named_parameters()}
        decayed = [names[id(p)] for g in groups if g["weight_decay"] for p in g["params"]]
        self.assertTrue(decayed)
        self.assertEqual([n for n in decayed if n.endswith("bias")], [])

    def test_the_code_digest_covers_what_the_pooler_is_built_from(self):
        files = radar_train.code_files()
        for path in ("src/models/utils/modules.py", "src/models/utils/pos_embs.py", "src/models/radar/loss.py"):
            self.assertIn(path, files)

    def test_skipped_step_changes_nothing(self):
        routing = load_routing()
        cache = TokenCache(self.paths["cache_dir"], token_grid=GRID, embed_dim=DIM)
        ds = RadarStudyDataset(read_table(self.paths["clips"]), read_table(self.paths["captions"], ("study_id",)),
                               routing, cache, num_clips=4)
        torch.manual_seed(0)
        model, tokenizer = radar_train.build_model(self.config, routing)
        optimizer = torch.optim.AdamW(radar_train.param_groups(model, self.config["optimization"]), lr=1e-3)
        teacher = radar_train.MomentumTeacher(model.text, 0.9)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        teacher_before = {k: v.clone() for k, v in teacher.state_dict().items()}
        batch = collate([ds[12], ds[13], ds[14], ds[15]], tokenizer)   # no captions at all
        loss, stats, contributed = radar_train.train_step(model, model, teacher, batch, optimizer, "none")
        self.assertFalse(contributed)
        self.assertEqual(loss, 0.0)
        self.assertTrue(_same(before, model.state_dict(), 0.0))
        self.assertTrue(_same(teacher_before, teacher.state_dict(), 0.0))
        self.assertEqual(optimizer.state_dict()["state"], {})

    def test_resume_reproduces_straight_training(self):
        out = os.path.join(self.root, "resumed")
        first = radar_train.train(self.config, out, stop_after_epoch=1)
        self.assertEqual(first["steps"], 4)
        with open(os.path.join(out, "log.jsonl"), "a") as f:          # an epoch interrupted after one step
            f.write(json.dumps({"epoch": 1, "step": 4, "loss": 9.0}) + "\n")
        with open(os.path.join(out, "epochs.jsonl"), "a") as f:
            f.write(json.dumps({"epoch": 1, "structures": {}}) + "\n")
        second = radar_train.train(self.config, out)
        self.assertEqual(second["steps"], 8)
        self.assertEqual(first["losses"] + second["losses"], self.straight["losses"])
        self.assertEqual(first["skipped_steps"] + (second["skipped_steps"] - first["skipped_steps"]),
                         self.straight["skipped_steps"])
        self.assertTrue(_same(_state(second), _state(self.straight), 1e-6))
        self.assertEqual(_steps(os.path.join(out, "log.jsonl")), list(range(8)))
        with open(os.path.join(out, "epochs.jsonl")) as f:
            self.assertEqual([json.loads(line)["epoch"] for line in f], [0, 1])
        with open(os.path.join(out, "resumes.jsonl")) as f:
            self.assertEqual([json.loads(line)["step"] for line in f], [4])
        changed = synthetic_config(self.paths, lr=2e-3)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            radar_train.train(changed, out)

    def test_normal_fraction_lowers_the_logged_normal_pairs(self):
        config = copy.deepcopy(self.config)
        config["data"]["normal_fraction"] = 0.01           # nearly every normal caption sits out
        out = os.path.join(self.root, "normal_fraction")
        radar_train.train(config, out, stop_after_epoch=1)

        def pairs(folder):
            with open(os.path.join(self.root, folder, "log.jsonl")) as f:
                return sum(st["normal_pairs"] for line in map(json.loads, f) if line["epoch"] == 0
                           for st in line["structures"].values())
        self.assertLess(pairs("normal_fraction"), pairs("straight"))

    def test_wandb_logs_steps_and_epochs_and_a_resume_continues_the_run(self):
        config = {**copy.deepcopy(self.config),
                  "meta": {"wandb_project": "radar", "wandb_entity": "team", "wandb_run_name": "a3"}}
        out, fake = os.path.join(self.root, "wandb"), FakeWandb()
        with mock.patch.object(wandb_logging, "wandb", fake):
            first = radar_train.train(config, out, stop_after_epoch=1)
            second = radar_train.train(config, out)
        self.assertEqual(first["fingerprint"], self.straight["fingerprint"])          # meta is run control
        self.assertEqual(second["losses"], self.straight["losses"][4:])
        self.assertEqual([c["resume"] for c in fake.calls], ["never", "must"])
        self.assertEqual(fake.calls[1]["id"], fake.runs[0].id)                         # the id came from latest.pt
        self.assertTrue(all(run.finished for run in fake.runs))
        steps = [p for run in fake.runs for p in run.logged if "train/loss" in p]
        self.assertEqual([p[wandb_logging.STEP_METRIC] for p in steps], list(range(8)))
        self.assertEqual([p["train/loss"] for p in steps], self.straight["losses"])
        epochs = [p for run in fake.runs for p in run.logged if "structures/lv/contributing_steps" in p]
        self.assertEqual(len(epochs), 2)
        self.assertIn("structures/pericardium/median_normal_pairs", epochs[0])
        self.assertNotIn("structures/pulmonic/median_rows", epochs[0])                 # never contributes: no median

        def strings(value):
            if isinstance(value, dict):
                return [s for v in value.values() for s in strings(v)]
            if isinstance(value, (list, tuple)):
                return [s for v in value for s in strings(v)]
            return [value] if isinstance(value, str) else []
        self.assertFalse([s for s in strings(fake.calls[0]["config"]) if os.sep in s])   # no path leaves the host
        self.assertEqual(fake.calls[0]["config"]["data"]["captions"], "captions.csv")

    def test_without_a_wandb_project_nothing_is_logged(self):
        fake = FakeWandb()
        with mock.patch.object(wandb_logging, "wandb", fake):
            radar_train.train(self.config, os.path.join(self.root, "no_wandb"), stop_after_epoch=1)
        self.assertEqual(fake.calls, [])

    def test_resume_with_other_code(self):
        out = os.path.join(self.root, "code_change")
        radar_train.train(self.config, out, stop_after_epoch=1)
        with mock.patch.object(radar_train, "code_digest", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "code digest"):
                radar_train.train(self.config, out)
            second = radar_train.train(self.config, out, allow_code_change=True)
        self.assertEqual(second["steps"], 8)
        with open(os.path.join(out, "resumes.jsonl")) as f:
            record = json.loads(f.readline())
        self.assertEqual(record["code_sha256"], "0" * 64)
        self.assertEqual(record["checkpoint_code_sha256"], radar_train.code_digest())

    def test_stochastic_resume_one_process(self):
        config = synthetic_config(self.paths, text={"dropout": 0.3})
        straight = radar_train.train(config, os.path.join(self.root, "dropout_straight"))
        self.assertNotEqual(straight["losses"], self.straight["losses"])        # dropout is active
        out = os.path.join(self.root, "dropout_resumed")
        first = radar_train.train(config, out, stop_after_epoch=1)
        second = radar_train.train(config, out)
        self.assertEqual(first["losses"] + second["losses"], straight["losses"])
        self.assertTrue(_same(_state(second), _state(straight), 0.0))

    def test_stochastic_resume_two_processes(self):
        config = synthetic_config(self.paths, text={"dropout": 0.3})
        root = tempfile.mkdtemp(dir=self.root)
        mp.spawn(_resume_worker, args=(2, os.path.join(root, "init"), config, root, root), nprocs=2, join=True)
        for rank in range(2):
            with self.subTest(rank=rank):
                result = torch.load(os.path.join(root, f"rank{rank}.pt"))
                self.assertEqual(result["resumed"], result["straight"])
                self.assertTrue(_same(result["resumed_state"], result["straight_state"], 0.0))
        checkpoint = torch.load(os.path.join(root, "resumed", radar_train.CHECKPOINT), weights_only=True)
        self.assertEqual(checkpoint["world_size"], 2)
        self.assertFalse(torch.equal(checkpoint["rng"][0]["torch"], checkpoint["rng"][1]["torch"]))  # own streams
        with self.assertRaisesRegex(ValueError, "same"):
            radar_train.train(config, os.path.join(root, "resumed"))          # 2 processes wrote it, 1 resumes

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("RADAR_CUDA_TESTS") == "1",
                         "set RADAR_CUDA_TESTS=1 on a machine with a free GPU")
    def test_stochastic_resume_cuda(self):
        config = synthetic_config(self.paths, text={"dropout": 0.3}, device="cuda")
        straight = radar_train.train(config, os.path.join(self.root, "cuda_straight"))
        out = os.path.join(self.root, "cuda_resumed")
        first = radar_train.train(config, out, stop_after_epoch=1)
        second = radar_train.train(config, out)
        checkpoint = torch.load(os.path.join(out, radar_train.CHECKPOINT), weights_only=True)
        self.assertIsNotNone(checkpoint["rng"][0]["cuda"])
        self.assertTrue(torch.allclose(torch.tensor(first["losses"] + second["losses"]),
                                       torch.tensor(straight["losses"]), atol=1e-5))
        self.assertTrue(_same({k: v.cpu() for k, v in _state(second).items()},
                              {k: v.cpu() for k, v in _state(straight).items()}, 1e-5))

    def test_loader_workers_start_safely(self):
        dataset = RadarStudyDataset(read_table(self.paths["clips"]), read_table(self.paths["captions"], ("study_id",)),
                                    load_routing(), TokenCache(self.paths["cache_dir"], token_grid=GRID,
                                                               embed_dim=DIM), num_clips=4)
        sampler = radar_train.StudyBatchSampler(len(dataset), 4)
        _, tokenizer = radar_train.build_model(self.config, load_routing())
        self.assertIsNone(radar_train.make_loader(dataset, sampler, tokenizer, 0).multiprocessing_context)
        for method in ("spawn", "forkserver"):
            loader = radar_train.make_loader(dataset, sampler, tokenizer, 2, method)
            self.assertEqual(loader.multiprocessing_context.get_start_method(), method)
            self.assertFalse(loader.persistent_workers)
        with self.assertRaisesRegex(ValueError, "worker_start_method"):
            radar_train.make_loader(dataset, sampler, tokenizer, 2, "fork")

        config = synthetic_config(self.paths, num_workers=2)
        summary = radar_train.train(config, os.path.join(self.root, "workers"))
        self.assertEqual(summary["loader_workers"], {"num_workers": 2, "start_method": "spawn"})
        self.assertEqual(summary["fingerprint"], self.straight["fingerprint"])        # run control only
        self.assertEqual(summary["losses"], self.straight["losses"])
        self.assertTrue(_same(_state(summary), _state(self.straight), 0.0))

    def test_view_families_come_only_from_the_registry(self):
        config = copy.deepcopy(self.config)
        config["data"]["view_families"] = {"A4C_RV": "A4C"}
        with self.assertRaisesRegex(ValueError, "data.routing"):
            radar_train.train(config, os.path.join(self.root, "old_key"))
        registry = yaml.safe_load(open(radar_train.DEFAULT_ROUTING, encoding="utf-8"))
        registry["view_families"]["PSAX_AV"] = "PSAX"
        other = os.path.join(self.root, "other_families.yaml")
        with open(other, "w", encoding="utf-8") as f:
            yaml.safe_dump(registry, f)
        self.assertEqual(load_routing(other).view_families["PSAX_AV"], "PSAX")
        cache = TokenCache(self.paths["cache_dir"], token_grid=GRID, embed_dim=DIM)
        prints = [radar_train.fingerprint(self.config, load_routing(path), cache, 16)["sha256"]
                  for path in (radar_train.DEFAULT_ROUTING, other)]
        self.assertNotEqual(*prints)

    def test_the_run_records_what_fixed_its_bags(self):
        out = os.path.join(self.root, "bags")
        config = copy.deepcopy(self.config)
        config["optimization"]["epochs"] = 1
        with mock.patch.object(radar_train, "RadarStudyDataset", wraps=RadarStudyDataset) as made:
            radar_train.train(config, out)
        recorded = made.call_args
        dataset = RadarStudyDataset(*recorded.args, **recorded.kwargs)
        with open(os.path.join(out, "fingerprint.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["bags"], {"train": dataset.bag_record()})

    def test_the_code_digest_covers_how_bags_are_chosen(self):
        source = radar_train.choose_bag.__module__
        self.assertEqual(source, "src.datasets.study_bags")
        path = os.path.abspath(sys.modules[source].__file__)
        real_open, digest = open, radar_train.code_digest()

        def edited(name, *args, **kwargs):
            if os.path.abspath(str(name)) == path:
                return io.BytesIO(b"another way of choosing")
            return real_open(name, *args, **kwargs)

        with mock.patch("builtins.open", edited):
            self.assertNotEqual(radar_train.code_digest(), digest)

    def test_captions_of_studies_outside_the_clips_stop_the_run(self):
        captions = read_table(self.paths["captions"], ("study_id",))
        other = pd.concat([captions, captions.iloc[:1].assign(study_id="ELSEWHERE")], ignore_index=True)
        config = copy.deepcopy(self.config)
        config["data"]["captions"] = seal_captions(other, os.path.join(self.root, "elsewhere"), self.paths["clips"])
        with self.assertRaisesRegex(ValueError, "no clip in this split"):
            radar_train.train(config, os.path.join(self.root, "elsewhere_out"))

    def test_a_table_that_is_not_the_builds_stops_the_run(self):
        folder = os.path.join(self.root, "edited")
        path = seal_captions(read_table(self.paths["captions"], ("study_id",)), folder, self.paths["clips"])
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n")
        config = copy.deepcopy(self.config)
        config["data"]["captions"] = path
        with self.assertRaisesRegex(ValueError, "changed since it was built"):
            radar_train.train(config, os.path.join(self.root, "edited_out"))
        os.remove(os.path.join(folder, "build_info.json"))
        with self.assertRaisesRegex(ValueError, "No build_info.json"):
            radar_train.train(config, os.path.join(self.root, "edited_out"))

    def test_the_arm_must_be_stated(self):
        for loss in ({}, {"use_normal": False}, {"use_exact": "false"}, {"use_exact": None}):
            config = copy.deepcopy(self.config)
            config["loss"] = loss
            with self.assertRaisesRegex(ValueError, "loss.use_exact must be set"):
                radar_train.build_model(config, load_routing())
        config = copy.deepcopy(self.config)
        del config["loss"]
        with self.assertRaisesRegex(ValueError, "loss.use_exact must be set"):
            radar_train.build_model(config, load_routing())

    def test_every_arm_records_the_rules_of_its_table(self):
        config = copy.deepcopy(self.config)
        config["loss"] = {"use_exact": False}
        config["optimization"]["epochs"] = 1
        out = os.path.join(self.root, "table_rules")
        radar_train.train(config, out)
        with open(os.path.join(out, "fingerprint.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["caption_policy_digest"], "synthetic")

    def test_a2_and_a3_wait_for_an_approved_policy(self):
        captions = read_table(self.paths["captions"], ("study_id",))
        provisional = seal_captions(captions.assign(normality_status="provisional"),
                                    os.path.join(self.root, "provisional"), self.paths["clips"])
        missing = seal_captions(captions.drop(columns=["normality_status"]), os.path.join(self.root, "no_status"),
                                self.paths["clips"])
        arms = {"a0": {"use_exact": False}, "a1": {"use_exact": True}, "a2": {"use_exact": True, "use_normal": True},
                "a3": {"use_exact": True, "use_normal": True, "use_abnormal_teacher": True, "teacher_momentum": 0.9}}
        for name, loss in arms.items():
            config = copy.deepcopy(self.config)
            config["loss"] = loss
            config["optimization"]["epochs"] = 1
            for table in (provisional, missing):
                config["data"]["captions"] = table
                out = os.path.join(self.root, f"{name}_{os.path.basename(os.path.dirname(table))}")
                if name in ("a0", "a1"):
                    self.assertGreater(len(radar_train.train(config, out)["losses"]), 0)
                else:
                    with self.assertRaisesRegex(ValueError, "approved policy"):
                        radar_train.train(config, out)

    def test_a2_needs_one_digest_matching_an_approved_policy(self):
        captions = read_table(self.paths["captions"], ("study_id",))
        policy, digest = self.paths["normality_policy"], captions.policy_digest.iloc[0]
        loss = {"use_normal": True}
        self.assertEqual(radar_train.check_normality_approved(captions, loss, policy), digest)
        mixed = [digest] * (len(captions) - 1) + ["other"]
        one = "need one policy digest"
        for name, (table, message) in {
                "no column": (captions.drop(columns=["policy_digest"]), one),
                "one blank": (captions.assign(policy_digest=[digest] * (len(captions) - 1) + [""]), one),
                "mixed": (captions.assign(policy_digest=mixed), one),
                "blank": (captions.assign(policy_digest=""), "is not the approved"),
                "missing": (captions.assign(policy_digest=float("nan")), "is not the approved"),
                "another policy": (captions.assign(policy_digest="other"), "is not the approved")}.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                radar_train.check_normality_approved(table, loss, policy)
        with self.assertRaisesRegex(ValueError, "data.normality_policy"):
            radar_train.check_normality_approved(captions, loss, None)
        with self.assertRaisesRegex(ValueError, "not approved"):
            radar_train.check_normality_approved(captions, loss, SHIPPED_POLICY)

    def test_a_stale_approval_is_refused(self):
        with open(self.paths["normality_policy"], encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        raw["approval"]["digest"] = "another"
        stale = os.path.join(self.root, "stale.yaml")
        with open(stale, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f)
        captions = read_table(self.paths["captions"], ("study_id",))
        with self.assertRaisesRegex(ValueError, "approve again"):
            radar_train.check_normality_approved(captions, {"use_normal": True}, stale)

    def test_a0_and_a1_need_no_policy(self):
        bare = read_table(self.paths["captions"], ("study_id",)).drop(columns=["policy_digest", "normality_status"])
        self.assertIsNone(radar_train.check_normality_approved(bare, {"use_exact": True}, None))

    def test_the_run_records_its_policy(self):
        config = copy.deepcopy(self.config)
        config["optimization"]["epochs"] = 1
        out = os.path.join(self.root, "recorded")
        radar_train.train(config, out)
        with open(os.path.join(out, "fingerprint.json"), encoding="utf-8") as f:
            record = json.load(f)
        digest = read_table(self.paths["captions"], ("study_id",)).policy_digest.iloc[0]
        self.assertEqual(record["normality_policy_digest"], digest)
        self.assertNotIn("normality_policy", record["config"]["data"])       # the digest identifies it, not the path

    def test_captions_too_long_to_read_whole_are_left_out(self):
        tokenizer = HashTokenizer(max_length=5)
        long_a, long_b = "un deux trois quatre cinq FEVG 60", "un deux trois quatre cinq FEVG 40"
        self.assertEqual(tokenizer([long_a])[0].tolist(), tokenizer([long_b])[0].tolist())   # the same tokens
        table = pd.DataFrame({"structure": ["lv", "lv", "lv"], "caption": [long_a, long_b, "VG normal"],
                              "ef_status": ["appended", "stated", "appended"]})
        kept, report = radar_train.fit_captions(table, tokenizer, max_fraction=1.0)
        self.assertEqual(kept.caption.tolist(), ["VG normal"])
        self.assertEqual(report["structures"]["lv"], {"rows": 3, "overlength": 2, "longest": 8,
                                                      "overlength_with_appended_ef": 1})

    def test_a_caption_that_just_fits_is_kept(self):
        tokenizer = HashTokenizer(max_length=3)
        table = pd.DataFrame({"structure": ["lv"], "caption": ["VG normal"]})
        kept, report = radar_train.fit_captions(table, tokenizer)
        self.assertEqual((len(kept), report["structures"]["lv"]["longest"]), (1, 3))
        self.assertEqual(tokenizer(["VG normal"])[0].shape[1], 3)

    def test_too_many_long_captions_stop_the_run(self):
        table = pd.DataFrame({"structure": ["lv"] * 100, "caption": ["un deux trois quatre"] * 2 + ["VG"] * 98})
        with self.assertRaisesRegex(ValueError, "Raise max_length"):
            radar_train.fit_captions(table, HashTokenizer(max_length=3))
        self.assertEqual(len(radar_train.fit_captions(table, HashTokenizer(max_length=3), 0.05)[0]), 98)

    def test_a_run_reports_its_caption_lengths_and_trains_only_what_fits(self):
        captions = read_table(self.paths["captions"], ("study_id",))
        long_caption = " ".join(["mot"] * 40)
        row = captions.index[captions.structure == "aortic"][0]
        captions.loc[row, "caption"] = long_caption
        config = copy.deepcopy(self.config)
        config["data"]["captions"] = seal_captions(captions, os.path.join(self.root, "long"), self.paths["clips"])
        config["text"]["max_overlength_fraction"] = 1.0
        config["optimization"]["epochs"] = 1
        out = os.path.join(self.root, "lengths")
        with mock.patch.object(radar_train, "RadarStudyDataset", wraps=RadarStudyDataset) as dataset:
            radar_train.train(config, out)
        trained = dataset.call_args.args[1]
        self.assertEqual(len(trained), len(captions) - 1)
        self.assertNotIn(long_caption, set(trained.caption))
        with open(os.path.join(out, "caption_lengths.json"), encoding="utf-8") as f:
            report = json.load(f)
        self.assertEqual(report["structures"]["aortic"]["overlength"], 1)
        self.assertEqual(report["structures"]["lv"]["overlength"], 0)

    def test_rows_without_a_clip_take_no_part_in_the_loss(self):
        routing = load_routing()
        cache = TokenCache(self.paths["cache_dir"], token_grid=GRID, embed_dim=DIM)
        dataset = RadarStudyDataset(read_table(self.paths["clips"]), read_table(self.paths["captions"], ("study_id",)),
                                    routing, cache, num_clips=4)
        torch.manual_seed(0)
        model, tokenizer = radar_train.build_model(self.config, routing)
        model.train()
        teacher = MomentumTeacher(model.text, 0.9)
        batch = collate([dataset[i] for i in range(4)], tokenizer)
        names = [s.name for s in routing.trained]
        lost = names.index("pericardium")
        self.assertTrue(batch["has_caption"][:, lost].any())
        batch["eligible"][:, lost] = False

        def run():
            model.zero_grad(set_to_none=True)
            loss, stats, valid = model(batch, teacher)
            loss.backward()
            return loss.detach(), stats, valid, {n: p.grad.clone() for n, p in model.named_parameters()
                                                 if p.grad is not None}

        loss, stats, valid, grads = run()
        self.assertTrue(stats["lv"]["loss"] is not None and torch.isfinite(loss))
        self.assertFalse(valid[:, lost].any())
        self.assertIsNone(stats["pericardium"]["loss"])
        model.pooler.null_token.copy_(3.0 * torch.randn(1, 1, DIM))   # not constant: LayerNorm would remove it
        other_loss, _, _, other_grads = run()
        self.assertTrue(torch.equal(loss, other_loss))
        self.assertEqual(sorted(grads), sorted(other_grads))
        for name, grad in grads.items():
            self.assertTrue(torch.equal(grad, other_grads[name]), name)

    def test_bf16_autocast_step_is_finite(self):
        config = synthetic_config(self.paths, amp="bf16", epochs=1)
        summary = radar_train.train(config, os.path.join(self.root, "bf16"))
        self.assertTrue(all(torch.isfinite(torch.tensor(summary["losses"]))))

    def test_two_processes_match_one(self):
        result_dir = tempfile.mkdtemp(dir=self.root)
        mp.spawn(_worker, args=(2, os.path.join(result_dir, "init"), self.config, os.path.join(self.root, "ddp"),
                                result_dir), nprocs=2, join=True)
        reference = _state(self.straight)
        for rank in range(2):
            with self.subTest(rank=rank):
                result = torch.load(os.path.join(result_dir, f"rank{rank}.pt"))
                self.assertEqual(result["skipped"], self.straight["skipped_steps"])
                self.assertTrue(torch.allclose(torch.tensor(result["losses"]), torch.tensor(self.straight["losses"]),
                                               atol=1e-5))
                self.assertTrue(_same(result["state"], reference, 1e-5))
                teacher = {k: v for k, v in self.straight["teacher"].state_dict().items()}
                self.assertTrue(_same(result["teacher"], teacher, 1e-5))


if __name__ == "__main__":
    unittest.main()
