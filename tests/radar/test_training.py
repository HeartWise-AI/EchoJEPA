"""RADAR skeleton on CPU: encoder loading and dimension checks, the record loader and its guards, a forward/backward
step, and checkpoint save/resume. A vit_tiny on 4-frame 32 px synthetic clips stands in for ViT-B."""

import contextlib
import copy
import json
import logging
import os
import tempfile
import time
import unittest
from unittest import mock

import torch
import yaml

import app.radar.train as train_module
from app.radar.data import CAPTION_UNVALIDATED, RadarClipDataset, SplitGuardError, collate, select_records
from app.radar.loss import ContrastiveLoss
from app.radar.model import build_model, build_video_backbone, load_video_weights
from app.radar.records import write_records
from app.radar.train import _loader, check_config, run, trainable_state
from evals.video_classification_frozen import provenance
from evals.video_classification_frozen.modelcustom import vjepa_2_1_encoder
from tests.radar.synthetic import ONTOLOGY, approved_clip_records, candidate, record, replace, write_video

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OPTIONS = dict(patch_size=16, tubelet_size=2, uniform_power=True, use_rope=True, use_sdpa=True,
               img_temporal_dim_size=1, interpolate_rope=True, modality_embedding=True, n_output_distillation=1)


def tiny_config(tmp, records, **changes):
    cfg = {
        "meta": {"seed": 0, "output_dir": os.path.join(tmp, "run"), "resume": True,
                 "dev_allow_unreviewed_candidates": False},
        "data": {"records": records, "validation": False, "frames_per_clip": 4, "frame_step": 1, "resolution": 32,
                 "random_clip_sampling": True, "batch_size": 4, "num_workers": 0},
        "visual": {"model_name": "vit_tiny", "checkpoint": None, "allow_random_init": True, "embed_dim": 192,
                   "freeze": True, "pooling": "attentive", "pool_heads": 3, "encoder": dict(OPTIONS)},
        "text": {"type": "tiny", "dim": 32, "depth": 1, "heads": 2, "max_length": 32, "vocab_size": 512},
        "projection": {"dim": 16, "hidden": None},
        "loss": {"init_temperature": 0.07, "learnable_temperature": True, "max_logit_scale": 100.0,
                 "same_study": "ignore", "equal_concepts": "ignore"},
        "optimization": {"epochs": 2, "lr": 1e-3, "final_lr": 1e-4, "warmup_steps": 1, "weight_decay": 0.01,
                         "grad_clip": 1.0, "checkpoint_every": 100},
    }
    for dotted, value in changes.items():
        section, key = dotted.split(".")
        cfg[section][key] = value
    return cfg


def split_table(tmp):
    """A stand-in archive patient split table and its SHA-256, the `split_source` bound records carry."""
    path = os.path.join(tmp, "patient_splits.csv")
    with open(path, "w") as f:
        f.write("patient_id,split\nsynthetic,train\n")
    return path, provenance.sha256(path)


def val_records(tmp, n, start):
    """Approved val-split records whose patients and studies are not the training ones."""
    return [replace(r, split="val", study_key=f"val-study-{i}", patient_key=f"val-patient-{i}")
            for i, r in enumerate(approved_clip_records(tmp, n=n, start=start))]


@contextlib.contextmanager
def text_dropout(p):
    """Every model `run` builds gets dropout `p` in its tiny text encoder (built with none); yields the initial
    trainable state of each, in build order."""
    initial = []

    def build(*args, **kwargs):
        model, meta = build_model(*args, **kwargs)
        for m in model.text.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = p
            elif isinstance(m, torch.nn.MultiheadAttention):
                m.dropout = p
        initial.append({k: v.clone() for k, v in trainable_state(model, model.frozen_prefixes()).items()})
        return model, meta

    with mock.patch("app.radar.train.build_model", build):
        yield initial


def differences(a, b, where=""):
    """Paths at which two nested checkpoint entries differ; tensors are compared bitwise."""
    if isinstance(a, torch.Tensor):
        return [] if isinstance(b, torch.Tensor) and torch.equal(a, b) else [where]
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            return [where]
        return [d for k in a for d in differences(a[k], b[k], f"{where}/{k}")]
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            return [where]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{where}/{i}")]
    return [] if a == b else [where]


class Quiet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.WARNING)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name


class TestVideoEncoderLoading(Quiet):
    def save_checkpoint(self, model_name="vit_tiny"):
        torch.manual_seed(1)
        backbone = build_video_backbone(model_name, 32, 4, OPTIONS)
        path = os.path.join(self.tmp, f"{model_name}.pt")
        state = {f"module.backbone.{k}": v for k, v in backbone.state_dict().items()}
        torch.save({"target_encoder": state, "epoch": 7}, path)
        return path, backbone

    def test_loads_like_the_frozen_probe(self):
        path, source = self.save_checkpoint()
        backbone = build_video_backbone("vit_tiny", 32, 4, OPTIONS)
        self.assertEqual(load_video_weights(backbone, path), {"checkpoint_key": "target_encoder", "epoch": 7})
        probe = vjepa_2_1_encoder.init_module(32, 4, path, {"encoder": {"model_name": "vit_tiny", **OPTIONS}}, {})
        for (k, a), (_, b) in zip(sorted(backbone.state_dict().items()), sorted(probe.model.state_dict().items())):
            self.assertTrue(torch.equal(a, b), k)
        self.assertTrue(torch.equal(backbone.state_dict()["patch_embed.proj.weight"],
                                    source.state_dict()["patch_embed.proj.weight"]))

    def test_a_checkpoint_of_another_size_fails_with_the_dimensions(self):
        path, _ = self.save_checkpoint("vit_tiny")
        backbone = build_video_backbone("vit_small", 32, 4, OPTIONS)
        with self.assertRaisesRegex(ValueError, r"embed_dim: checkpoint 192 vs config 384"):
            load_video_weights(backbone, path)

    def test_config_width_and_options_are_checked(self):
        cfg = tiny_config(self.tmp, "unused")
        cfg["visual"]["embed_dim"] = 768
        with self.assertRaisesRegex(ValueError, "visual.embed_dim is 768 but vit_tiny is 192 wide"):
            build_model(cfg, 32, 4)
        cfg = tiny_config(self.tmp, "unused")
        cfg["visual"]["encoder"]["pred_depth"] = 4
        with self.assertRaisesRegex(ValueError, "Unknown V-JEPA 2.1 encoder options"):
            build_model(cfg, 32, 4)
        cfg = tiny_config(self.tmp, "unused")
        cfg["visual"]["allow_random_init"] = False
        with self.assertRaisesRegex(ValueError, "visual.checkpoint is required"):
            build_model(cfg, 32, 4)
        cfg = tiny_config(self.tmp, "unused")
        cfg["visual"]["checkpoint"], _ = self.save_checkpoint()
        cfg["visual"]["checkpoint_key"] = "ema_encoder"
        with self.assertRaisesRegex(KeyError, "no `ema_encoder`"):
            build_model(cfg, 32, 4)


class TestRecordSelection(Quiet):
    def test_approved_only_by_default_and_dev_flag_is_recorded(self):
        recs = approved_clip_records(self.tmp, n=2) + [record(9, caption=candidate("Coupe A4C."))]
        items, stats = select_records(recs, split="train", purpose="training")
        self.assertEqual(len(items), 2)
        self.assertEqual((stats["approved_captions"], stats["skipped_without_approved_caption"]), (2, 1))
        items, stats = select_records(recs, split="train", purpose="training", allow_unreviewed_candidates=True)
        self.assertEqual(items[-1].caption_source, CAPTION_UNVALIDATED)
        self.assertTrue(stats["allow_unreviewed_candidates"])
        self.assertEqual(stats["unvalidated_candidate_captions"], 1)

    def test_split_guard(self):
        recs = approved_clip_records(self.tmp, n=2)
        with self.assertRaises(SplitGuardError):
            select_records(recs, split="test", purpose="training")
        with self.assertRaisesRegex(SplitGuardError, "1 test-split records"):
            select_records(recs + [replace(recs[0], record_id="rc-" + "f" * 24, split="test")], split="train",
                           purpose="training")
        _, source = split_table(self.tmp)
        bound = [replace(r, split_source=source) for r in recs + val_records(self.tmp, 2, start=10)]
        items, stats = select_records(bound, split="val", purpose="validation_loss", split_source=source)
        self.assertEqual((len(items), stats["split"], stats["purpose"]), (2, "val", "validation_loss"))
        # The leak the guard exists for: a training patient's other study under val.
        leak = replace(bound[-1], record_id="rc-" + "e" * 24, study_key="study-9", patient_key=recs[0].patient_key)
        with self.assertRaisesRegex(SplitGuardError, "1 patients have records in more than one split"):
            select_records(bound + [leak], split="train", purpose="training")


class TestStep(Quiet):
    def test_loader_and_forward_backward(self):
        recs = approved_clip_records(self.tmp, n=4, studies=2,
                                     captions=["Coupe A4C, mode B.", "coupe a4c, mode b", "Coupe PLAX.", "Coupe A2C."])
        items, _ = select_records(recs, split="train", purpose="training")
        dataset = RadarClipDataset(items, frames_per_clip=4, resolution=32, frame_step=1, seed=0)
        batch = collate([dataset[i] for i in range(4)])
        self.assertEqual(tuple(batch["video"].shape), (4, 3, 4, 32, 32))
        self.assertEqual(batch["clip_indices"].tolist()[0], [0, 1, 2, 3])
        self.assertEqual(batch["caption_key"][0], batch["caption_key"][1])  # identical after normalization

        torch.manual_seed(0)
        model, _ = build_model(tiny_config(self.tmp, "unused"), 32, 4)
        loss_fn = ContrastiveLoss()
        tokens = model.text.tokenize(batch["caption"])
        video, text = model(batch["video"], batch["clip_indices"], tokens)
        self.assertEqual(tuple(video.shape), (4, 16))
        self.assertTrue(torch.allclose(video.norm(dim=-1), torch.ones(4), atol=1e-5))
        mask = loss_fn.mask(batch["caption_key"], batch["study_key"], batch["concept_key"])
        self.assertEqual(int(mask[0, 1]), 1)  # identical captions are positives
        self.assertEqual(int(mask[0, 2]), -1)  # clips 0 and 2 share a study: ignored, not negative
        loss, _ = loss_fn(video, text, mask)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.video.backbone.parameters()))  # frozen encoder
        self.assertIsNotNone(model.video.pool.query_tokens.grad)
        self.assertIsNotNone(model.text.tokens.weight.grad)
        self.assertIsNotNone(loss_fn.logit_scale.grad)

    def test_padding_tokens_are_left_out_of_the_mean_pool(self):
        torch.manual_seed(0)
        cfg = tiny_config(self.tmp, "unused")
        cfg["visual"]["pooling"] = "mean"
        model, _ = build_model(cfg, 32, 4)
        video = torch.randn(1, 3, 4, 32, 32)
        padded = model.video(video, torch.tensor([[0, 1, -1, -1]]))
        tokens = model.video.backbone(video)
        self.assertTrue(torch.allclose(padded, tokens[:, :4].mean(1), atol=1e-6))  # first tubelet's 2 x 2 tokens


class TestTraining(Quiet):
    def records_file(self, n=8, **kw):
        recs = approved_clip_records(self.tmp, n=n, studies=3, **kw)
        path = os.path.join(self.tmp, "records.jsonl")
        write_records(path, recs, ONTOLOGY)
        return path

    def trained(self, cfg):
        return {k: v.clone() for k, v in torch.load(os.path.join(cfg["meta"]["output_dir"], "latest.pt"),
                                                    weights_only=False)["model"].items()}

    def test_resume_matches_an_uninterrupted_run(self):
        path = self.records_file()
        whole = tiny_config(self.tmp, path)
        whole["meta"]["output_dir"] = os.path.join(self.tmp, "whole")
        self.assertEqual(run(whole)["step"], 4)  # 8 clips / batch 4 x 2 epochs

        parts = copy.deepcopy(whole)
        parts["meta"]["output_dir"] = os.path.join(self.tmp, "parts")
        self.assertEqual(run(parts, stop_after_steps=3), {"step": 3, "loss": mock.ANY, "stopped": True})
        self.assertEqual(run(parts)["step"], 4)
        a, b = self.trained(whole), self.trained(parts)
        self.assertEqual(sorted(a), sorted(b))
        self.assertFalse(any(k.startswith("video.backbone.") for k in a))  # frozen encoder is not re-saved
        for k in a:
            self.assertTrue(torch.allclose(a[k], b[k], atol=1e-5), k)

        with open(os.path.join(parts["meta"]["output_dir"], "provenance.json")) as f:
            prov = json.load(f)
        self.assertEqual(prov["resumed_from_step"], 3)
        self.assertEqual(prov["last_checkpoint_step"], 4)
        self.assertEqual(prov["dev_flags"], {"allow_unreviewed_candidates": False, "visual_random_init": True})
        self.assertEqual(prov["caption_sources"], {"approved": 8, "unvalidated_candidate": 0})
        self.assertEqual(len(prov["records_sha256"]), 64)
        self.assertIn("commit", prov["git"])
        with open(os.path.join(parts["meta"]["output_dir"], "log.jsonl")) as f:
            steps = [json.loads(line)["step"] for line in f]
        self.assertEqual(steps, [1, 2, 3, 4])

    def test_epoch_batches_keep_random_sampler_order_and_resume_skips_whole_batches(self):
        class Indices(torch.utils.data.Dataset):
            def __len__(self):
                return 10

            def set_epoch(self, epoch):
                pass

        for epoch in (0, 1):
            sampler = torch.utils.data.RandomSampler(range(10), generator=torch.Generator().manual_seed(
                7 * 1_000_003 + epoch))
            expected = list(torch.utils.data.BatchSampler(sampler, 4, drop_last=True))
            self.assertEqual(list(_loader(Indices(), 4, 7, epoch, 0).batch_sampler), expected)
            self.assertEqual(list(_loader(Indices(), 4, 7, epoch, 0, skip=1).batch_sampler), expected[1:])
        before = torch.get_rng_state()
        iter(_loader(Indices(), 4, 7, 0, 0))
        self.assertTrue(torch.equal(torch.get_rng_state(), before))  # worker seeds come from the loader's own

    def test_resume_with_text_dropout_and_validation_is_exact(self):
        train = approved_clip_records(self.tmp, n=12, studies=3)
        val = val_records(self.tmp, 4, start=20)
        self.assertFalse({r.patient_key for r in train} & {r.patient_key for r in val})
        splits, source = split_table(self.tmp)
        path = os.path.join(self.tmp, "records.jsonl")
        write_records(path, [replace(r, split_source=source) for r in train + val], ONTOLOGY)
        # Distinct captions with cross-study pairs as negatives give a nonzero loss (all synthetic captions share one
        # concept set, so equal_concepts=ignore would mask every pair). No weight decay: only gradients move weights.
        cfg = tiny_config(self.tmp, path, **{"data.validation": True, "optimization.epochs": 3,
                                             "optimization.weight_decay": 0.0, "loss.equal_concepts": "off",
                                             "data.patient_splits": splits})

        def train_in(name, stops=()):
            c = copy.deepcopy(cfg)
            c["meta"]["output_dir"] = out = os.path.join(self.tmp, name)
            for stop in stops:
                self.assertEqual(run(c, stop_after_steps=stop), {"step": stop, "loss": mock.ANY, "stopped": True})
            self.assertEqual(run(c)["step"], 9)  # 12 clips / batch 4 x 3 epochs
            with open(os.path.join(out, "log.jsonl")) as f:
                log = [json.loads(line) for line in f]
            return torch.load(os.path.join(out, "latest.pt"), weights_only=False), log

        with text_dropout(0.3):
            model, _ = train_module.build_model(cfg, 32, 4)
        tokens = model.train().text.tokenize(["Coupe apicale 4 cavités (A4C), mode B."])
        self.assertFalse(torch.equal(model.encode_text(tokens), model.encode_text(tokens)))  # dropout is on

        with text_dropout(0.3) as initial:
            whole, log = train_in("whole")
        losses = [line["loss"] for line in log if "loss" in line]
        self.assertEqual(len(losses), 9)
        self.assertTrue(all(loss > 0 for loss in losses))
        self.assertGreater(len(set(losses)), 1)
        self.assertEqual([line["step"] for line in log if "val_loss" in line], [3, 6, 9])
        self.assertTrue(all(line["val_loss"] > 0 for line in log if "val_loss" in line))
        (start,) = initial
        self.assertEqual(sorted(start), sorted(whole["model"]))
        self.assertEqual([k for k in start if torch.equal(start[k], whole["model"][k])], [])  # all were trained

        for name, stops in (("mid-epoch", (4,)), ("epoch boundary", (6,)), ("twice", (2, 3))):
            with self.subTest(name), text_dropout(0.3):
                parts, parts_log = train_in(name, stops)
                self.assertEqual(differences({k: whole[k] for k in ("model", "loss", "optimizer", "step")},
                                             {k: parts[k] for k in ("model", "loss", "optimizer", "step")}), [])
                self.assertEqual(parts_log, log)

    def test_resume_refuses_another_config_or_records(self):
        path = self.records_file()
        cfg = tiny_config(self.tmp, path)
        run(cfg, stop_after_steps=1)
        changed = copy.deepcopy(cfg)
        changed["optimization"]["lr"] = 5e-4
        with self.assertRaisesRegex(ValueError, "config_sha256"):
            run(changed)
        with open(path, "a") as f:
            f.write("\n")
        with self.assertRaisesRegex(ValueError, "records_sha256"):
            run(cfg)
        no_resume = copy.deepcopy(cfg)
        no_resume["meta"]["resume"] = False
        with self.assertRaises(FileExistsError):
            run(no_resume)

    def test_unreviewed_captions_need_the_dev_flag_and_are_recorded(self):
        recs = [record(i, study=f"s{i}", video_path=write_video(os.path.join(self.tmp, f"u{i}.mp4"), seed=i),
                       caption=candidate(f"Coupe A4C numéro {i}.")) for i in range(4)]
        path = os.path.join(self.tmp, "unreviewed.jsonl")
        write_records(path, recs, ONTOLOGY)
        cfg = tiny_config(self.tmp, path, **{"optimization.epochs": 1})
        with self.assertRaisesRegex(ValueError, "0 captioned training clips"):
            run(cfg)
        cfg["meta"]["dev_allow_unreviewed_candidates"] = True
        run(cfg)
        with open(os.path.join(cfg["meta"]["output_dir"], "provenance.json")) as f:
            prov = json.load(f)
        self.assertTrue(prov["dev_flags"]["allow_unreviewed_candidates"])
        self.assertEqual(prov["caption_sources"], {"approved": 0, "unvalidated_candidate": 4})

    def test_validation_use_is_logged(self):
        train = approved_clip_records(self.tmp, n=4, studies=2)
        splits, source = split_table(self.tmp)
        path = os.path.join(self.tmp, "with_val.jsonl")
        write_records(path, [replace(r, split_source=source) for r in train + val_records(self.tmp, 2, start=20)],
                      ONTOLOGY)
        cfg = tiny_config(self.tmp, path, **{"data.validation": True, "optimization.epochs": 1})
        with self.assertRaisesRegex(ValueError, "data.validation needs data.patient_splits"):
            run(cfg)
        cfg["data"]["patient_splits"] = splits
        run(cfg)
        out = cfg["meta"]["output_dir"]
        with open(os.path.join(out, "provenance.json")) as f:
            uses = json.load(f)["split_uses"]
        self.assertEqual([(u["split"], u["purpose"], u["used"]) for u in uses],
                         [("train", "training", 4), ("val", "validation_loss", 2)])
        with open(os.path.join(out, "log.jsonl")) as f:
            self.assertIn("val_loss", f.read().splitlines()[-1])


class TestStalledRead(Quiet):
    """A read that stalls in a loader worker ends training or validation after data.loader_timeout seconds."""

    STALL_S = 10  # each stalled item; a regression waits this per item and then fails, never hangs

    def setUp(self):
        super().setUp()
        splits, source = split_table(self.tmp)
        self.train = [replace(r, split_source=source) for r in approved_clip_records(self.tmp, n=8, studies=4)]
        self.val = [replace(r, split_source=source) for r in val_records(self.tmp, 4, start=20)]
        self.path = os.path.join(self.tmp, "records.jsonl")
        write_records(self.path, self.train + self.val, ONTOLOGY)
        self.cfg = tiny_config(self.tmp, self.path, **{"data.validation": True, "data.patient_splits": splits,
                                                       "data.num_workers": 1, "data.loader_timeout": 1,
                                                       "optimization.epochs": 1})

    def run_stalling(self, records):
        stalled, original = {r.record_id for r in records}, RadarClipDataset.__getitem__

        def getitem(dataset, index):  # inherited by the forked workers
            if dataset.items[index].record_id in stalled:
                time.sleep(self.STALL_S)
            return original(dataset, index)

        start = time.monotonic()
        with mock.patch.object(RadarClipDataset, "__getitem__", getitem), \
                self.assertRaisesRegex(RuntimeError, "timed out after 1 seconds"):
            run(self.cfg)
        self.assertLess(time.monotonic() - start, self.STALL_S)

    def test_a_stalled_training_read_times_out(self):
        self.run_stalling(self.train)

    def test_a_stalled_validation_read_times_out(self):
        self.run_stalling(self.val)
        with open(os.path.join(self.cfg["meta"]["output_dir"], "log.jsonl")) as f:
            logged = [json.loads(line) for line in f]
        self.assertEqual(len([e for e in logged if "loss" in e]), 2)  # the epoch trained; validation stalled
        with open(os.path.join(self.cfg["meta"]["output_dir"], "provenance.json")) as f:
            self.assertEqual(json.load(f)["loader"], {"num_workers": 1, "timeout_s": 1})

    def test_without_workers_there_is_no_timeout_and_bad_values_are_refused(self):
        cfg = tiny_config(self.tmp, self.path, **{"optimization.epochs": 1})
        run(cfg)
        with open(os.path.join(cfg["meta"]["output_dir"], "provenance.json")) as f:
            self.assertEqual(json.load(f)["loader"], {"num_workers": 0, "timeout_s": 0})
        for bad in (-1, "900", float("inf")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "loader_timeout"):
                run(tiny_config(self.tmp, self.path, **{"data.loader_timeout": bad, "meta.output_dir":
                                                        os.path.join(self.tmp, f"bad-{bad}")}))


class TestShippedConfig(unittest.TestCase):
    def test_skeleton_config_matches_the_ef_probe_encoder_and_has_placeholders(self):
        with open(os.path.join(ROOT, "configs", "radar", "skeleton-vitb16.yaml")) as f:
            cfg = yaml.safe_load(f)
        with open(os.path.join(ROOT, "configs", "eval", "vitb16", "lvef-a4c-224px-16f.yaml")) as f:
            probe = yaml.safe_load(f)["model_kwargs"]["pretrain_kwargs"]["encoder"]
        self.assertEqual(cfg["visual"]["encoder"], {k: v for k, v in probe.items()
                                                    if k not in ("model_name", "checkpoint_key")})
        self.assertEqual((cfg["visual"]["model_name"], cfg["visual"]["checkpoint_key"]),
                         (probe["model_name"], probe["checkpoint_key"]))
        self.assertEqual(os.path.basename(cfg["visual"]["checkpoint"]), "vjepa2_1_vitb_mimic_pt169_c60.pt")
        self.assertFalse(cfg["meta"]["dev_allow_unreviewed_candidates"])
        self.assertEqual(cfg["loss"]["same_study"], "ignore")
        with self.assertRaisesRegex(ValueError, "placeholder"):
            check_config(cfg)


if __name__ == "__main__":
    unittest.main()
