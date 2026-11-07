# tests/datasets/test_study_bags.py

"""RADAR and its matched baseline read the same clips: the real datasets on one synthetic clip table, compared by clip
id across epochs, evaluation bags, row orders, worker counts and ranks. A shared family map alone is not enough, a
`shared_bags` run takes the registry's map and nothing else, whatever its embedding map holds, a clip that cannot be
read stops it, and equal records mean equal bags."""

import copy
import random
import unittest

import numpy as np
import pandas as pd
import torch

from src.datasets.radar_study_dataset import RadarStudyDataset, StudyBatchSampler
from src.datasets.study_bags import bag_record, choose_bag, evaluation_rng, training_rng
from src.datasets.video_study_dataset import VideoStudyDataset, shared_bag_options
from src.models.radar.routing import DEFAULT_ROUTING, load_routing

ROUTING = load_routing()
VIEWS = [v for v in ROUTING.view_classes if v != "OTHER"]
STUDIES, BUDGET, SEED, EVAL_SEED = 24, 5, 3, 11


def clip_table(order=0):
    """Studies of 3 to 30 clips, whose ids are video paths as in the real table, in row order `order`."""
    rng = random.Random(0)
    rows = []
    for s in range(STUDIES):
        views = rng.sample(VIEWS, rng.randint(1, 8))
        for c in range(rng.randint(3, 30)):
            rows.append({"study_id": f"S{s:02d}", "clip_id": f"/clips/S{s:02d}/{rng.getrandbits(40):x}.mp4",
                         "view": rng.choice(views), "acquisition": "bmode", "patient_id": f"P{s:02d}", "lvef": 60.0})
    return pd.DataFrame(rows).sample(frac=1, random_state=order).reset_index(drop=True)


class Cache:
    """What RadarStudyDataset asks of a token cache."""
    shape = (1, 1)

    def __contains__(self, clip_id):
        return True

    def load(self, clip_id):
        return torch.zeros(1, 1)


class Probe(VideoStudyDataset):
    """The study probe's dataset, with a clip that holds its own position in the table instead of pixels."""

    def __init__(self, table, **options):
        self.positions = {path: float(i) for i, path in enumerate(sorted(table.clip_id))}
        self.unreadable = set()
        super().__init__(table, path_column="clip_id", group_column="study_id", target_columns=["lvef"],
                         view_column="view", num_videos=BUDGET, frames_per_clip=1, **options)

    def load_video(self, path):
        if path in self.unreadable:
            return None
        return [torch.full((1, 1, 1, 1), self.positions[path])], [np.zeros(1, dtype=np.int64)]


def radar(table, training=True, **options):
    captions = pd.DataFrame(columns=["study_id", "structure", "caption", "caption_id", "normality"])
    return RadarStudyDataset(table, captions, ROUTING, Cache(), num_clips=BUDGET, training=training, seed=SEED,
                             eval_seed=EVAL_SEED, **options)


def probe(table, training=True, **options):
    return Probe(table, sampling_mode="shared_bags", routing=ROUTING, seed=SEED, eval_seed=EVAL_SEED,
                 training=training, **options)


def probe_bag(item, paths):
    """The clip ids a probe item read, in slot order."""
    clips, _, mask = item[:3]
    return [paths[int(value)] for value, real in zip(clips.flatten(), mask) if real]


def everything(items):
    return items


def radar_bags(dataset, **loader):
    """{study: clip ids} read through a DataLoader."""
    out = {}
    for batch in torch.utils.data.DataLoader(dataset, collate_fn=everything, **loader):
        out.update({item["study_id"]: item["clip_ids"] for item in batch})
    return out


def probe_bags(dataset, **loader):
    paths = sorted(dataset.positions)
    out = {}
    for batch in torch.utils.data.DataLoader(dataset, collate_fn=everything, **loader):
        for item in batch:
            bag = probe_bag(item, paths)
            out[bag[0].split("/")[2]] = bag
    return out


class TestSameClips(unittest.TestCase):
    def setUp(self):
        self.table = clip_table()

    def read_both(self, first, second):
        paths = sorted(second.positions)
        ours = {s: first[i]["clip_ids"] for i, (s, _) in enumerate(first.studies)}
        theirs = {study: probe_bag(second[i], paths) for i, study in enumerate(second.study_ids)}
        return ours, theirs

    def test_training_bags_are_the_same_clips_in_the_same_slots(self):
        first, second = radar(self.table), probe(self.table, shuffle_videos=True)
        seen = []
        for epoch in range(4):
            first.set_epoch(epoch)
            second.set_epoch(epoch)
            ours, theirs = self.read_both(first, second)
            self.assertEqual(ours, theirs)
            seen.append(ours)
        self.assertNotEqual(seen[0], seen[1])

    def test_evaluation_bags_are_the_same_clips_for_every_bag_index(self):
        first, second = radar(self.table, training=False), probe(self.table, training=False)
        seen = []
        for bag in range(3):
            first.eval_bag_index = second.eval_bag_index = bag
            ours, theirs = self.read_both(first, second)
            self.assertEqual(ours, theirs)
            seen.append(ours)
        self.assertNotEqual(seen[0], seen[1])

    def test_the_row_order_of_either_table_changes_nothing(self):
        ours, ours_probe = self.read_both(radar(clip_table(order=1)), probe(clip_table(order=2)))
        theirs, theirs_probe = self.read_both(radar(clip_table(order=3)), probe(clip_table(order=4)))
        self.assertEqual(ours, theirs)
        self.assertEqual(ours_probe, theirs_probe)

    def test_each_rank_reads_the_same_bag_as_one_process(self):
        first, second = radar(self.table), probe(self.table)
        first.set_epoch(1)
        second.set_epoch(1)
        whole, _ = self.read_both(first, second)
        ours, theirs = {}, {}
        for rank in range(2):
            ours.update(radar_bags(first, batch_sampler=StudyBatchSampler(len(first), 4, rank, 2, SEED)))
            sampler = torch.utils.data.distributed.DistributedSampler(second, num_replicas=2, rank=rank, seed=SEED)
            theirs.update(probe_bags(second, sampler=sampler, batch_size=2))
        self.assertEqual(ours, {s: whole[s] for s in ours})
        self.assertEqual(theirs, whole)

    def test_workers_read_the_same_bags_and_see_the_epoch(self):
        first, second = radar(self.table), probe(self.table)
        workers = {"num_workers": 2, "multiprocessing_context": "spawn", "persistent_workers": False, "batch_size": 4}
        seen = []
        for epoch in range(2):
            first.set_epoch(epoch)
            second.set_epoch(epoch)
            whole, _ = self.read_both(first, second)
            self.assertEqual(radar_bags(first, **workers), whole)
            self.assertEqual(probe_bags(second, **workers), whole)
            seen.append(whole)
        self.assertNotEqual(*seen)

    def test_a_shared_family_map_alone_gives_other_clips(self):
        balanced = Probe(self.table, sampling_mode="view_balanced", view_family_map=ROUTING.view_families,
                         eval_seed=EVAL_SEED)
        ours, theirs = self.read_both(radar(self.table, training=False), balanced)
        self.assertNotEqual(ours, theirs)

    def test_the_old_modes_do_not_read_the_new_streams(self):
        balanced = Probe(self.table, sampling_mode="view_balanced", eval_seed=EVAL_SEED)
        self.assertEqual(balanced.selection, "view_balanced")
        self.assertEqual(balanced.bag_record()["registry_sha256"], None)


class TestLabelMap(unittest.TestCase):
    """`view_labels_map` numbers the view embedding: it changes no bag, however little of the table it holds."""
    PARTIAL = {"A4C": 0, "Other": 1}

    def setUp(self):
        self.table = clip_table()

    def test_an_incomplete_map_reads_the_same_clips_and_records_the_same_thing(self):
        paths = sorted(self.table.clip_id)
        for training in (True, False):
            with self.subTest(training=training):
                mapped = probe(self.table, training=training, view_labels_map=self.PARTIAL)
                ours = radar(self.table, training=training)
                self.assertEqual({s: ours[i]["clip_ids"] for i, (s, _) in enumerate(ours.studies)},
                                 {s: probe_bag(mapped[i], paths) for i, s in enumerate(mapped.study_ids)})
                self.assertEqual(mapped.bag_record(), ours.bag_record())

    def test_the_map_still_gives_the_slots_their_classes(self):
        mapped = probe(self.table, view_labels_map=self.PARTIAL)
        self.assertEqual({c for study in mapped.studies for c in study[3]}, {0, 1})
        self.assertLessEqual({m["family"] for m in mapped.video_metadata.values()},
                             {ROUTING.view_families.get(v, v) for v in VIEWS})

    def test_an_older_mode_keeps_its_fallback(self):
        balanced = Probe(self.table, sampling_mode="view_balanced", view_labels_map=self.PARTIAL)
        self.assertEqual({m["family"] for m in balanced.video_metadata.values()}, {"A4C", "Other"})


class TestUnreadableClips(unittest.TestCase):
    """A shared bag is read whole or the run stops; nothing else can match the bag the run records."""

    def setUp(self):
        self.table = clip_table()
        self.paths = sorted(self.table.clip_id)

    def bag_of(self, dataset, index=0):
        study = dataset.studies[index]
        metadata = [dataset.video_metadata[(study[0], path)] for path in study[1]]
        return [study[1][i] for i in dataset.select(study[1], [m["family"] for m in metadata], study[0])]

    def test_one_selected_clip_that_fails_stops_training_and_evaluation(self):
        for training in (True, False):
            with self.subTest(training=training):
                dataset = probe(self.table, training=training)
                dataset.unreadable = {self.bag_of(dataset)[2]}
                with self.assertRaisesRegex(RuntimeError, "failed to load") as stopped:
                    dataset[0]
                with self.assertRaisesRegex(RuntimeError, "failed to load"):
                    dataset.get_item_study(0)              # what the evaluation reads
                self.assertNotIn(next(iter(dataset.unreadable)), str(stopped.exception))    # a path can name a patient

    def test_a_study_that_fails_is_never_replaced(self):
        dataset = probe(self.table, training=True)
        dataset.unreadable = set(dataset.studies[0][1])
        with self.assertRaisesRegex(RuntimeError, "failed to load"):
            dataset[0]

    def test_a_clip_outside_the_bag_changes_nothing(self):
        dataset = probe(self.table, training=True)
        bag = self.bag_of(dataset)
        dataset.unreadable = set(dataset.studies[0][1]) - set(bag)
        self.assertEqual(probe_bag(dataset[0], self.paths), bag)

    def test_an_older_mode_still_skips_and_replaces(self):
        dataset = Probe(self.table, sampling_mode="view_balanced", training=True)
        study = dataset.studies[0][1]
        dataset.num_videos = len(study)                          # every clip of the study is selected
        dataset.unreadable = {study[0]}
        self.assertEqual(int(dataset.get_item_study(0)[2].sum()), len(study) - 1)
        dataset.unreadable = set(study)
        self.assertIsNone(dataset.get_item_study(0))
        self.assertNotEqual(probe_bag(dataset[0], self.paths)[0].split("/")[2], "S00")    # another study took its slot


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.table = clip_table()

    def test_the_registry_gives_the_families(self):
        views = ["A4C", "A4C_ZOOM", "PSAX", "PSAX_AV"]
        table = pd.DataFrame({"study_id": "S", "clip_id": [f"/c/{v}.mp4" for v in views], "view": views,
                              "patient_id": "P", "lvef": 60.0})
        dataset = probe(table)
        dataset.num_videos = 3
        for epoch in range(20):
            dataset.set_epoch(epoch)
            paths = table.clip_id.sort_values().tolist()
            families = [dataset.video_metadata[("S", path)]["family"] for path in paths]
            chosen = [paths[i] for i in dataset.select(paths, families, "S")]
            self.assertIn("/c/PSAX_AV.mp4", chosen)
            self.assertIn("/c/PSAX.mp4", chosen)
            self.assertEqual(sum("A4C" in path for path in chosen), 1)

    def test_what_would_read_other_clips_is_refused(self):
        unknown = self.table.assign(view="A4C_RV")
        doubled = pd.concat([self.table, self.table.iloc[:1].assign(study_id="S99", patient_id="P99")],
                            ignore_index=True)
        cases = {
            "needs the routing registry": (self.table, {"routing": None}),
            "remove `view_family_map`": (self.table, {"view_family_map": {"A4C_ZOOM": "A4C"}}),
            "remove `views`": (self.table, {"views": ["A4C"]}),
            "num_clips_per_video": (self.table, {"num_clips_per_video": 2}),
            "does not know": (unknown, {}),
            "each clip once": (doubled, {}),
            "study id": (self.table.assign(study_id="'" + self.table.study_id.astype(str) + "'"), {}),
        }
        for message, (table, options) in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                Probe(table, **{"sampling_mode": "shared_bags", "routing": ROUTING, **options})

    def test_the_mode_and_the_registry_go_together(self):
        self.assertEqual(shared_bag_options({"sampling_mode": "random"}, 1), {})
        options = shared_bag_options({"sampling_mode": "shared_bags", "routing": DEFAULT_ROUTING}, 1)
        self.assertEqual((options["routing"].sha256, options["seed"]), (ROUTING.sha256, 1))
        for data in ({"sampling_mode": "shared_bags"}, {"sampling_mode": "random", "routing": DEFAULT_ROUTING}):
            with self.subTest(data), self.assertRaisesRegex(ValueError, "go together"):
                shared_bag_options(data, 1)


class TestRecord(unittest.TestCase):
    def setUp(self):
        self.table = clip_table()

    def test_both_runs_record_the_same_thing(self):
        self.assertEqual(radar(self.table).bag_record(), probe(self.table).bag_record())
        self.assertEqual(radar(self.table, training=False).bag_record(),
                         probe(self.table, training=False).bag_record())
        self.assertEqual(radar(clip_table(order=1)).bag_record(), probe(clip_table(order=2)).bag_record())

    def test_a_training_record_holds_the_training_seed_and_an_evaluation_record_the_other(self):
        self.assertEqual(probe(self.table).bag_record()["seed"], SEED)
        self.assertEqual(probe(self.table, training=False).bag_record()["seed"], EVAL_SEED)

    def test_anything_that_changes_the_bags_changes_the_record(self):
        candidates = [("S1", "/a.mp4", "A4C"), ("S1", "/b.mp4", "PSAX"), ("S2", "/c.mp4", "A2C")]
        record = bag_record(candidates, "registry", "view_balanced", 12, 0)
        self.assertEqual(record, bag_record(candidates[::-1], "registry", "view_balanced", 12, 0))
        others = [bag_record(candidates[:2], "registry", "view_balanced", 12, 0),
                  bag_record([("S1", "/a.mp4", "A2C"), *candidates[1:]], "registry", "view_balanced", 12, 0),
                  bag_record([("S0", "/a.mp4", "A4C"), *candidates[1:]], "registry", "view_balanced", 12, 0),
                  bag_record(candidates, "another", "view_balanced", 12, 0),
                  bag_record(candidates, "registry", "random", 12, 0),
                  bag_record(candidates, "registry", "view_balanced", 8, 0),
                  bag_record(candidates, "registry", "view_balanced", 12, 1)]
        for other in others:
            self.assertNotEqual(record, other)

    def test_the_streams_are_named_by_everything_they_depend_on(self):
        def first(rng):
            return copy.copy(rng).random()

        streams = [training_rng(0, 0, "S"), training_rng(1, 0, "S"), training_rng(0, 1, "S"), training_rng(0, 0, "T"),
                   evaluation_rng(0, 0, "S"), evaluation_rng(1, 0, "S"), evaluation_rng(0, 1, "S"),
                   evaluation_rng(0, 0, "T")]
        self.assertEqual(len({first(rng) for rng in streams}), len(streams))

    def test_a_bag_is_a_function_of_its_candidates_not_their_order(self):
        candidates = [(f"/c{i}.mp4", v) for i, v in enumerate(["A4C", "A4C_ZOOM", "PSAX", "PSAX_AV", "A2C"] * 3)]
        bag = choose_bag(candidates, 4, "view_balanced", training_rng(0, 0, "S"), ROUTING.view_families)
        self.assertEqual(bag, choose_bag(candidates[::-1], 4, "view_balanced", training_rng(0, 0, "S"),
                                         ROUTING.view_families))
        self.assertEqual(len(set(bag)), 4)
