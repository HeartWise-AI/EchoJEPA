# tests/models/test_radar_empty_rank.py

"""The assembled RADAR model under DDP when a rank has no caption row.

Four consecutive optimizer steps over global batches of four studies (each of two gloo ranks reads every other
study of the batch), with availability changing from step to step:

1. rank 1 has no captioned study, rank 0 has two (the global batch contributes through the LV);
2. no captioned study on either rank (a globally skipped step);
3. rank 0 has no captioned study, rank 1 has two;
4. both ranks have a captioned study.

Each step's loss, skip decisions and parameter gradients equal those of one process reading the whole global batch,
and the weights after each optimizer step agree. The collectives have a short timeout so a mismatch fails the test
instead of hanging it.
"""

import datetime
import os
import shutil
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from evals.radar_frozen import train as radar_train
from src.datasets.radar_study_dataset import RadarStudyDataset, collate, read_table
from src.datasets.radar_token_cache import TokenCache
from src.models.radar.routing import load_routing
from src.models.radar.text_encoder import MomentumTeacher
from tests.models.radar_synthetic import DIM, GRID, make_synthetic, synthetic_config

# Studies S00-S05 are captioned, S06-S15 are not; rank r reads positions r, r + 2 of each global batch.
STEPS = [["S00", "S12", "S01", "S13"],
         ["S12", "S13", "S14", "S15"],
         ["S12", "S03", "S13", "S04"],
         ["S00", "S02", "S14", "S15"]]


def _setup(paths, config):
    routing = load_routing()
    cache = TokenCache(paths["cache_dir"], token_grid=GRID, embed_dim=DIM)
    dataset = RadarStudyDataset(read_table(paths["clips"]), read_table(paths["captions"], ("study_id",)), routing,
                                cache, num_clips=4)
    torch.manual_seed(0)
    model, tokenizer = radar_train.build_model(config, routing)
    return dataset, model, tokenizer


def _run(paths, config, rank=0, world_size=1):
    """Per step: (loss, skipped structures, gradients, weights after the step)."""
    dataset, model, tokenizer = _setup(paths, config)
    position = {item: i for i, item in enumerate(s for s, _ in dataset.studies)}
    optimizer = torch.optim.AdamW(radar_train.param_groups(model, config["optimization"]), lr=1e-2)
    wrapped = model
    if world_size > 1:
        wrapped = torch.nn.parallel.DistributedDataParallel(model)
    teacher = MomentumTeacher(model.text, 0.9)
    model.train()
    out = []
    for studies in STEPS:
        batch = collate([dataset[position[s]] for s in studies[rank::world_size]], tokenizer)
        loss, stats, _ = wrapped(batch, teacher)
        loss.backward()
        grads = {n: p.grad.clone() if p.grad is not None else None for n, p in model.named_parameters()}
        contributed = any(s["loss"] is not None for s in stats.values())
        if contributed:
            optimizer.step()
            model.loss.clamp_temperature_()
            teacher.update(model.text)
        optimizer.zero_grad(set_to_none=True)
        out.append({"loss": float(loss.detach()), "skipped": {n: s["skipped"] for n, s in stats.items()},
                    "contributed": contributed, "grads": grads,
                    "weights": {k: v.detach().clone() for k, v in model.state_dict().items()}})
    return out


def _worker(rank, world_size, init_file, paths, config, out_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(seconds=60))
    try:
        torch.save(_run(paths, config, rank, world_size), os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


class TestEmptyCaptionRank(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.paths = make_synthetic(os.path.join(cls.root, "data"))
        cls.config = synthetic_config(cls.paths)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_fixture_covers_the_cases(self):
        dataset, _, _ = _setup(self.paths, self.config)
        captioned = {s for s, _ in dataset.studies if any(k[0] == s for k in dataset.captions)}
        per_rank = [[sum(s in captioned for s in step[r::2]) for r in range(2)] for step in STEPS]
        self.assertEqual(per_rank, [[2, 0], [0, 0], [0, 2], [1, 1]])

    def test_two_processes_match_one(self):
        reference = _run(self.paths, self.config)
        self.assertEqual([r["contributed"] for r in reference], [True, False, True, True])
        out_dir = tempfile.mkdtemp(dir=self.root)
        mp.spawn(_worker, args=(2, os.path.join(out_dir, "init"), self.paths, self.config, out_dir), nprocs=2,
                 join=True)
        for rank in range(2):
            result = torch.load(os.path.join(out_dir, f"rank{rank}.pt"))
            for k, (got, ref) in enumerate(zip(result, reference)):
                with self.subTest(rank=rank, step=k):
                    self.assertAlmostEqual(got["loss"], ref["loss"], places=5)
                    self.assertEqual(got["skipped"], ref["skipped"])
                    for name, g in ref["grads"].items():
                        if g is None:
                            self.assertIsNone(got["grads"][name], name)
                        else:
                            self.assertTrue(torch.allclose(got["grads"][name], g, rtol=1e-4, atol=5e-5), name)
                    for name, w in ref["weights"].items():
                        self.assertTrue(torch.allclose(got["weights"][name], w, atol=1e-5), name)


if __name__ == "__main__":
    unittest.main()
