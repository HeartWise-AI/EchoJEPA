# evals/radar_frozen/val_monitor.py

"""Per-epoch validation monitoring of RADAR training (the `validation` config section; off when absent).

Monitoring only: it reads the cached tokens of a validation split, changes no parameter and no checkpoint, and
`evals/radar_frozen/train.py` treats the section as run control (outside the fingerprint), so turning it on or off
never changes what a run trains or whether it resumes. The training random streams are saved before and put back
after each pass, so a run with validation trains bit for bit as one without.

Each pass embeds every validation study from `num_bags` evaluation bags (`zero_shot_eval.embed_studies`: the same
bags as the zero-shot evaluation, from `eval_seed`), averages each structure's embedding over the bags that hold a
clip it may read, and reports:

- video -> caption retrieval per trained structure, with de-duplicated captions, as DeepCORO_CLIP's validation does
  (`runners/video_constrative_learning_runner.py`): the distinct B-mode captions of the structure's evaluable studies
  are the gallery, each study's own caption is its target, and Recall@k (`recall_k`, default 1, 5, 10), MRR and the
  median rank are taken over those studies; `retrieval_mean` is the mean over the structures with at least two
  studies. Many studies share a caption (a normal one, mostly), so the gallery is far smaller than the studies, and
  chance Recall@1 is about 1 / gallery size, not 1 / studies;
- with `labels` (a study label table, `data/make_study_targets.py`), the zero-shot AUROC of each binary task of
  `prompts` (default `configs/radar/prompts_v0.yaml`) whose structure is trained, scored as
  `zero_shot_eval.py` scores it (mean_similarity pooling, the model's temperature), and their mean.

The report holds counts and metrics only, no caption and no study identifier.

Config: validation {clips, captions, split (default val), cache_dir (default data.cache_dir), every (epochs, default
1; the last epoch is always validated), num_bags (default 1), eval_seed (default 0), batch_size (default 32),
recall_k (default [1, 5, 10]), labels, prompts}. Clips only: a run with Doppler stills is validated on its clips
(as `zero_shot_eval --clip_only`), and one whose queries read demographics is refused.
"""

import os
import random
import statistics

import numpy as np
import torch
import torch.nn.functional as F

from evals.radar_frozen.zero_shot_eval import (
    CAPTION_COLUMNS,
    DEFAULT_PROMPTS,
    embed_studies,
    encode_task,
    pool_bags,
    read_labels,
    score_task,
)
from evals.video_classification_frozen.metrics import auroc
from src.datasets.radar_study_dataset import RadarStudyDataset, read_table
from src.datasets.radar_token_cache import TokenCache
from src.models.radar.zero_shot import encode_prompts, load_prompt_tasks

DEFAULT_RECALL_K = (1, 5, 10)


def rank_metrics(video, gallery, target, recall_k=DEFAULT_RECALL_K):
    """Recall@k, MRR and median rank of `target` [N] (gallery index of each video's caption) when the gallery
    [M, P] is ranked by cosine similarity to each video [N, P]. A tie with the target counts in the target's favour."""
    sims = F.normalize(video.float(), dim=-1) @ F.normalize(gallery.float(), dim=-1).T
    own = sims.gather(1, target[:, None])
    rank = (sims > own).sum(dim=1)                                      # 0 = first
    out = {f"recall@{k}": float((rank < k).float().mean()) for k in recall_k}
    out["mrr"] = float((1.0 / (rank.float() + 1.0)).mean())
    out["median_rank"] = float(statistics.median((rank + 1).tolist()))
    return out


def _rng_state():
    return (torch.get_rng_state(), random.getstate(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _set_rng_state(state):
    torch.set_rng_state(state[0])
    random.setstate(state[1])
    if state[2] is not None:
        torch.cuda.set_rng_state_all(state[2])


class Validator:
    """Builds the validation split once; `run(model, tokenizer)` returns one pass's report."""

    def __init__(self, section, config, routing, device):
        data, m = config["data"], config["model"]
        if (m.get("stills") or {}).get("demo_film"):
            raise ValueError("validation reads clips only, and this model's queries read demographics "
                             "(model.stills.demo_film).")
        for key in ("clips", "captions"):
            if not section.get(key):
                raise ValueError(f"validation.{key} is required.")
        self.every = int(section.get("every", 1))
        self.num_bags = int(section.get("num_bags", 1))
        self.batch_size = int(section.get("batch_size", 32))
        self.recall_k = tuple(int(k) for k in section.get("recall_k", DEFAULT_RECALL_K))
        if self.every < 1 or self.num_bags < 1 or not self.recall_k or min(self.recall_k) < 1:
            raise ValueError("validation.every, num_bags and recall_k must be at least 1.")
        self.amp = config["optimization"].get("amp", "none")
        self.device = device
        self.structures = [s.name for s in routing.trained]
        split = section.get("split", "val")
        clips = read_table(section["clips"])
        if "split" in clips.columns:
            clips = clips[clips.split.astype(str) == split]
        if clips.empty:
            raise ValueError(f"validation.clips has no clip of split {split!r}.")
        studies = set(clips.study_id.astype(str))
        captions = read_table(section["captions"], id_columns=("study_id",))
        if "split" in captions.columns:
            captions = captions[captions.split.astype(str) == split]
        captions = captions[captions.study_id.astype(str).isin(studies)
                            & captions.structure.isin(self.structures)]
        if "scope" in captions.columns:
            captions = captions[captions.scope == "bmode"]
        cache = TokenCache(section.get("cache_dir") or data["cache_dir"], token_grid=m["token_grid"],
                           embed_dim=m["embed_dim"])
        self.dataset = RadarStudyDataset(clips, captions[list(CAPTION_COLUMNS)], routing, cache,
                                         num_clips=data.get("num_clips", 12),
                                         selection=data.get("selection", "view_balanced"), training=False,
                                         eval_seed=int(section.get("eval_seed", 0)),
                                         acquisition_classes=data.get("acquisition_classes", ("bmode", "color_flow")))
        self.study_ids = [study for study, _ in self.dataset.studies]
        self.tasks, self.labels = {}, None
        if section.get("labels"):
            tasks = load_prompt_tasks(section.get("prompts") or DEFAULT_PROMPTS)
            self.tasks = {n: t for n, t in tasks.items() if t.mode == "binary" and t.structure in self.structures}
            if self.tasks:
                self.labels, _ = read_labels(section["labels"], self.tasks, self.study_ids)

    def due(self, epoch, last_epoch):
        """Whether the pass runs after zero-indexed `epoch` of a run whose last epoch is `last_epoch` - 1."""
        return (epoch + 1) % self.every == 0 or epoch + 1 == last_epoch

    @torch.no_grad()
    def run(self, model, tokenizer):
        state, was_training = _rng_state(), model.training
        model.eval()
        try:
            return self._run(model, tokenizer)
        finally:
            model.train(was_training)
            _set_rng_state(state)

    def _run(self, model, tokenizer):
        z, has = embed_studies(model, self.dataset, self.device, self.amp, self.num_bags, self.batch_size)
        report = {"studies": len(self.study_ids), "num_bags": self.num_bags, "retrieval": {}}
        for s, name in enumerate(self.structures):
            support = has[:, :, s]                                       # [bags, N]
            count = support.sum(dim=0)
            rows = [i for i, study in enumerate(self.study_ids)
                    if count[i] > 0 and (study, name, "bmode") in self.dataset.captions]
            entry = {"studies": len(rows)}
            report["retrieval"][name] = entry
            if len(rows) < 2:
                continue
            index = torch.tensor(rows)
            weight = support[:, index].float()[..., None]
            video = (z[:, index, s] * weight).sum(dim=0) / count[index].float()[:, None]
            texts = [self.dataset.captions[(self.study_ids[i], name, "bmode")][0] for i in rows]
            gallery_texts = sorted(set(texts))
            position = {t: j for j, t in enumerate(gallery_texts)}
            gallery = encode_prompts(model.text, tokenizer, gallery_texts).cpu()
            target = torch.tensor([position[t] for t in texts])
            entry.update(unique_captions=len(gallery_texts), **rank_metrics(video, gallery, target, self.recall_k))
        scored = [e for e in report["retrieval"].values() if "mrr" in e]
        report["retrieval_mean"] = ({k: float(np.mean([e[k] for e in scored]))
                                     for k in [f"recall@{k}" for k in self.recall_k] + ["mrr", "median_rank"]}
                                    if scored else {})
        if self.tasks:
            temperature = float(model.loss.effective_temperature())
            report["zero_shot"] = {}
            for name, task in self.tasks.items():
                s = self.structures.index(task.structure)
                encoded = encode_task(task, model.text, tokenizer)
                per_bag = torch.stack([score_task(task, z[b, :, s], encoded, temperature)
                                       for b in range(self.num_bags)])
                pooled, bags = pool_bags(per_bag, has[:, :, s])
                label = self.labels[task.target].to_numpy(dtype=float)
                keep = np.isfinite(label) & (bags > 0)
                value = auroc(label[keep] > 0.5, pooled[keep]) if keep.any() else float("nan")
                report["zero_shot"][name] = {"n": int(keep.sum()), "positives": int((label[keep] > 0.5).sum()),
                                             "auroc": None if not np.isfinite(value) else float(value)}
            values = [e["auroc"] for e in report["zero_shot"].values() if e["auroc"] is not None]
            report["zero_shot_mean_auroc"] = float(np.mean(values)) if values else None
        return report


def scalars(report):
    """The report as flat W&B scalars under `val/`."""
    out = {}
    for name, entry in report["retrieval"].items():
        for key, value in entry.items():
            out[f"val/retrieval/{name}/{key}"] = value
    for key, value in report.get("retrieval_mean", {}).items():
        out[f"val/retrieval/mean/{key}"] = value
    for name, entry in report.get("zero_shot", {}).items():
        if entry["auroc"] is not None:
            out[f"val/zero_shot/{name}/auroc"] = entry["auroc"]
    if report.get("zero_shot_mean_auroc") is not None:
        out["val/zero_shot/mean_auroc"] = report["zero_shot_mean_auroc"]
    return out


def jsonl_path(out_dir):
    return os.path.join(out_dir, "validation.jsonl")
