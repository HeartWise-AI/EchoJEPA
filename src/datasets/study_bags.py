# src/datasets/study_bags.py

"""Which clips of a study are read: the one definition RADAR and its matched baseline share.

A bag depends only on the study's candidate clips, the view families, the budget and a random stream named by the seed
and the study, with the epoch in training and the bag index in evaluation. It never depends on the rank, the worker or
a table's row order (`choose_bag` sorts the candidates by clip id first), so it is the same in every process and in
both pipelines. A view's family comes from the routing registry, and a family only steers the choice.

`bag_record` says what fixed a split's bags, and each run stores it: two runs with equal records read the same clips of
a study in the same epoch or evaluation bag. Equal records say nothing about the order of the studies or the batches.
"""

import hashlib
import random

BAG_VERSION = 1    # raise when a stream or the selection changes: bags of another version are other clips


def check_study_ids(ids):
    """Refuse a study id that is blank or has spaces or quotes around it. Every reader of a clip table takes its ids as
    written, and an id names a study and a random stream, so an id that one reader repaired would be another study to
    the next one. Nothing here repairs an id: a clip table is corrected where it is made."""
    texts = ["" if value is None or value != value else str(value) for value in ids]
    bad = sum(not text or text[0].isspace() or text[-1].isspace() or text[0] in "'\"" or text[-1] in "'\""
              for text in texts)
    if bad:
        raise ValueError(f"{bad} clips have a study id that is blank or has spaces or quotes around it: correct the "
                         f"clip table, or its readers will not agree on which study a clip belongs to.")


def training_rng(seed, epoch, study_id):
    """The random stream that chooses one study's training bag."""
    return random.Random(f"train:{seed}:{epoch}:{study_id}")


def evaluation_rng(eval_seed, bag, study_id):
    """The random stream that chooses one study's evaluation bag."""
    return random.Random(f"eval:{eval_seed}:{bag}:{study_id}")


def select_clips(clips, num_clips, mode, rng):
    """Positions (into `clips`, a list of (clip_id, view family)) of the clips read for one study."""
    if mode == "random":
        return rng.sample(range(len(clips)), min(num_clips, len(clips)))
    if mode != "view_balanced":
        raise ValueError(f"selection must be view_balanced or random, not {mode!r}.")
    by_view = {}
    for i, (_, view) in enumerate(clips):
        by_view.setdefault(view, []).append(i)
    views = sorted(by_view)
    rng.shuffle(views)
    for members in by_view.values():
        rng.shuffle(members)
    chosen = []
    while len(chosen) < min(num_clips, len(clips)):
        for view in views:
            if by_view[view] and len(chosen) < num_clips:
                chosen.append(by_view[view].pop())
    return chosen


def choose_bag(candidates, num_clips, mode, rng, view_families):
    """The clip ids read for one study, in slot order. `candidates`: its (clip_id, view) pairs, in any order."""
    ordered = sorted(candidates)
    grouped = [(clip, view_families.get(view, view)) for clip, view in ordered]
    return [ordered[i][0] for i in select_clips(grouped, num_clips, mode, rng)]


def bag_record(candidates, registry_sha256, selection, num_clips, seed):
    """What fixes one split's bags. `candidates`: its (study_id, clip_id, view) triples, in any order. `seed`: the
    training seed for a training split, the evaluation seed for any other."""
    digest = hashlib.sha256()
    for study, clip, view in sorted(candidates):
        digest.update(f"{study}\0{clip}\0{view}\n".encode("utf-8"))
    return {"version": BAG_VERSION, "registry_sha256": registry_sha256, "candidates_sha256": digest.hexdigest(),
            "selection": selection, "num_clips": int(num_clips), "seed": int(seed)}
