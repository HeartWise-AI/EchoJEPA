# evals/radar_frozen/train.py

"""RADAR v0 training: structure-level alignment of frozen EchoJEPA tokens with report captions.

Reads a finished token cache with the clip and caption tables of the training split, and trains the structure
pooler, the caption encoder and the temperature with the structure loss (`src/models/radar/loss.py`). It runs in
one process or several (torchrun, or a caller that has initialised torch.distributed): the device count is whatever
is launched.

Each optimizer step reads one global batch of studies (`StudyBatchSampler`). A step to which no structure
contributes anywhere (every structure skipped on the gathered batch, a decision every rank shares) changes nothing:
no optimizer step, no temperature clamp, no teacher update. The learning-rate schedule still counts it, so the
schedule's length is fixed by the number of studies (`step_schedule`). After each optimizer step the temperature is
clamped into its bounds and the momentum teacher, when the loss uses it, takes one EMA step.

The teacher differs from the released RADAR code in one way: the release updates its momentum encoder once per organ
inside each step (0.995 per update, about ten updates per step), and this one once per optimizer step. With the
release's 0.995 the teacher here therefore moves about ten times more slowly per step; `loss.teacher_momentum`:
0.995 ** 10 (about 0.95) gives the release's rate per step. It only matters for A3.

Randomness: every rank seeds with `seed` to build the model, so initialisation is identical everywhere, then
reseeds its stochastic stream (dropout) from (seed, rank), so ranks draw independent dropout masks.

`latest.pt`, written by rank 0 at the end of each epoch, holds the model, teacher, optimizer, step counters, every
rank's RNG states (CPU torch, Python and, on GPU, the rank's CUDA generator), the number of processes, and a
fingerprint of everything that defines the run: the config without run-control keys, `PROTOCOL_VERSION`, the
routing table, the token cache's meta (which carries the digest of its index, and so of every clip's token file
hash), the clip and caption tables and the number of studies. Resumes start at an epoch boundary and are exact:
each rank gets back its own RNG states. A resume is refused when the fingerprint differs, when the number of
processes or the device type differs (that would be a new stochastic trajectory, so a new run), and when the
source of the RADAR training code differs (`code_digest`), unless `allow_code_change` is set (for a change known not
to alter training, such as logging); both code digests are then logged.

Loader workers (`num_workers` > 0) start with `worker_start_method`, spawn (default) or forkserver, never the
platform's default fork: forking after CUDA and NCCL are initialised can deadlock (the PyTorch DDP documentation
asks for spawn or forkserver). Workers are started for each epoch rather than kept alive, so they receive the
dataset with its current epoch (its bags depend on it). Workers do not change what is trained: `num_workers` and
`worker_start_method` are run control, outside the fingerprint.

The normality flags feed A2 and A3 only (`loss.use_normal`, `loss.use_abnormal_teacher`). Those arms are refused unless
every caption row says `normality_status` = approved, every row carries the same `policy_digest`, and that digest is
the one of the approved policy named by `data.normality_policy` (for the current rules: an approval does not survive a
change to them). The digest goes into the run's fingerprint. A0 and A1 run on any caption table, without a policy.

Flags and caption ids describe the whole caption, so the encoder must read the whole caption: a truncated one could
give two different targets the same tokens. `fit_captions` measures every caption with the configured tokenizer and
leaves out those over `text.max_length`, for every arm. `caption_lengths.json` reports them per structure, with the LV
captions whose appended EF would have been cut, and the run stops if any structure loses more than
`text.max_overlength_fraction` (default 0.01): that calls for a longer `max_length`, not for a silent loss.

Logs (rank 0): `log.jsonl`, one line per step (epoch, step, loss, learning rates, temperature, contributed,
per-structure rows, skip reason, exact-caption pairs and normal-normal pairs); `epochs.jsonl`, per epoch and structure
the contributing steps and the medians of rows, exact-caption pairs and normal-normal pairs over them. Both pair
counts are taken in every arm, so the logs of two arms show the same quantities: exact-caption pairs are positives
in A1 and negatives in A0, normal-normal pairs are positives in A2 and A3 only, and their count is what
`data.normal_fraction` reduces. A resume drops the lines
written after the checkpoint (an interrupted epoch), and a run that starts from scratch drops them all with any older
checkpoint, so the logs always describe the checkpointed trajectory. `resumes.jsonl` records each resume (epoch, step,
and the code digests of the checkpoint and of the resuming code).

Weights & Biases (rank 0, optional): with `meta.wandb_project` and `meta.wandb_entity` (and `meta.wandb_run_name`),
each step's loss, temperature and learning rates and each epoch's per-structure medians are also logged to a run,
through `src.utils.wandb_logging`: the config goes up with its paths cut to file names, and no caption, study
identifier or environment metadata is sent. A resume continues the same W&B run (its id is in the checkpoint). `meta`
is run control, outside the fingerprint, and a failure to log never stops training.

Config: data {cache_dir, clips, captions, split, routing, normality_policy, num_clips, selection, acquisition_classes,
normal_fraction (share of normal captions a training epoch keeps, drawn per epoch, default 1)}
(another view family map means another registry through `routing`; the old `view_families` key is refused);
model {token_grid, embed_dim, depth, num_heads, mlp_ratio, proj_dim, activation_checkpointing};
text {backbone, max_length, max_overlength_fraction, ...: see `build_text_encoder`}; loss {use_exact (required:
false = A0, true = A1 and the base of A2/A3), use_normal,
use_abnormal_teacher, release_gate, min_pairs, init_temperature, temperature_bounds, teacher_momentum}; optimization
{epochs, global_batch, lr, text_lr, min_lr, weight_decay, warmup_fraction, seed, amp (none or bf16), num_workers,
worker_start_method}; meta {wandb_project, wandb_entity, wandb_run_name}; device (auto, cpu or cuda).

Doppler stills (optional; a config without these sections trains exactly as v0): data.stills {table (the split's
`stills.csv`), cache_dir (the stills' token cache), demographics (optional), num_stills, cond_columns,
heatmap_column}; model.stills {token_grid (e.g. [1, 14, 14]), cond (default true: the pooler reads cond_columns),
heatmap_bias, demo_film}; heads {targets, weight, target_weights, demographics, hidden_dim} for the per-still
measurement regression (`src/models/radar/heads.py`). They need a routing registry with still routing
(`configs/radar/radar_routing_v1.yaml`). A step contributes when a structure or a measurement target does. See
docs/radar/RADAR_stills_integration.md.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import statistics
import sys
from functools import partial

import pandas as pd
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel

from src.datasets.radar_study_dataset import (
    RadarStudyDataset,
    StudyBatchSampler,
    check_caption_build,
    collate,
    file_sha256,
    read_table,
)
from src.datasets.radar_stills import DEMO_DIM, StillSource, same_encoder
from src.datasets.radar_token_cache import TokenCache
from src.datasets.study_bags import choose_bag
from src.models.radar.heads import StillMeasurementHeads
from src.models.radar.loss import StructureContrastiveLoss
from src.models.radar.model import RadarModel
from src.models.radar.routing import DEFAULT_ROUTING, load_routing
from src.models.radar.schedule import step_schedule
from src.models.radar.structure_pooler import StructurePooler
from src.models.radar.text_encoder import MomentumTeacher, build_text_encoder, build_tokenizer
from src.utils.optimizer_coverage import check_optimizer_coverage
from src.utils.wandb_logging import PRIVATE_WANDB_SETTINGS, finish_wandb, init_wandb, log_scalars

CHECKPOINT = "latest.pt"
PROTOCOL_VERSION = 2   # bump when a code change alters what a given config trains
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN_CONTROL = {"optimization": ("num_workers", "worker_start_method")}   # keys that do not change what is trained
RUN_CONTROL_SECTIONS = ("meta",)                                         # whole sections that do not either
WORKER_START_METHODS = ("spawn", "forkserver")
AMP_MODES = ("none", "bf16")


def _world():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


def _device(config):
    choice = config.get("device", "auto")
    if choice == "cpu" or (choice == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", _world()[1] % max(1, torch.cuda.device_count())))
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def code_files():
    """The repository-relative source files of the modules RADAR training runs, the pooler's parts included."""
    modules = {__name__, RadarStudyDataset.__module__, choose_bag.__module__, TokenCache.__module__,
               StructureContrastiveLoss.__module__,
               RadarModel.__module__, load_routing.__module__, step_schedule.__module__, StructurePooler.__module__,
               build_text_encoder.__module__, check_optimizer_coverage.__module__, StillSource.__module__,
               StillMeasurementHeads.__module__, "src.datasets.still_preprocess", "src.utils.distributed",
               "src.models.utils.modules", "src.models.utils.pos_embs"}
    return sorted(os.path.relpath(os.path.abspath(sys.modules[m].__file__), REPO) for m in modules)


def code_digest():
    """sha256 over the source files (repository-relative paths and contents) of `code_files`."""
    h = hashlib.sha256()
    for path in code_files():
        h.update(path.encode() + b"\0")
        with open(os.path.join(REPO, path), "rb") as f:
            h.update(f.read() + b"\0")
    return h.hexdigest()


def fingerprint(config, routing, cache, n_studies, normality_digest=None, bags=None, caption_digest=None,
                stills=None):
    """What defines the run: the config without run-control keys and data paths, and the contents behind them.
    `normality_digest`: the approved policy behind the flags of A2 and A3. `bags`: {split: `bag_record`}, the same
    in the matched baseline's run when both read the same clips. `caption_digest`: the rules the caption table was
    built with, recorded in every arm, approved or not. `stills`: the still stream's record (absent without stills,
    so a v0 config keeps its v0 fingerprint)."""
    semantic = copy.deepcopy(config)
    for section, keys in RUN_CONTROL.items():
        for key in keys:
            semantic.get(section, {}).pop(key, None)
    for section in RUN_CONTROL_SECTIONS:
        semantic.pop(section, None)
    data = semantic.get("data", {})
    record = {
        "config": semantic,
        "protocol_version": PROTOCOL_VERSION,
        "routing_sha256": routing.sha256,
        "cache_meta": cache.meta,
        "clips_sha256": file_sha256(data.pop("clips")),
        "captions_sha256": file_sha256(data.pop("captions")),
        "caption_policy_digest": caption_digest,
        "n_studies": n_studies,
        "normality_policy_digest": normality_digest,
        "bags": bags,
    }
    if stills is not None:
        still_paths = data.get("stills", {})
        record["stills"] = {**stills, "table_sha256": file_sha256(still_paths.pop("table")),
                            "cache_meta": TokenCache(still_paths.pop("cache_dir")).meta,
                            "demographics_sha256": (file_sha256(still_paths.pop("demographics"))
                                                    if still_paths.get("demographics") else None)}
    data.pop("cache_dir", None)
    data.pop("routing", None)
    data.pop("normality_policy", None)
    record["sha256"] = hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()
    return record


def load_approved_policy(path):
    """The policy at `path`, which must be approved for the current rules (`load_policy` refuses a stale approval)."""
    if not path:
        raise ValueError("A2 and A3 need data.normality_policy, the approved policy behind the flags.")
    folder = os.path.join(REPO, "data")
    if folder not in sys.path:
        sys.path.append(folder)
    import radar_normality
    policy = radar_normality.load_policy(path)
    if policy.status != "approved":
        raise ValueError(f"{path} is not approved: audit the flags and fill in its approval block first.")
    return policy


def check_normality_approved(captions, loss, policy_path=None):
    """Refuse A2 and A3 on flags that do not come from one approved policy. Returns its digest; None for A0 and A1."""
    if not (loss.get("use_normal") or loss.get("use_abnormal_teacher")):
        return None
    status = set(captions["normality_status"]) if "normality_status" in captions.columns else {"missing"}
    if status != {"approved"}:
        raise ValueError(f"A2 and A3 need normality flags from an approved policy, but the caption table says "
                         f"{sorted(map(str, status))}. Review the cutoffs and audit the flags first; A0 and A1 need "
                         f"neither.")
    digests = (set(captions["policy_digest"].fillna("").astype(str).str.strip())
               if "policy_digest" in captions.columns else set())
    if len(digests) != 1:
        raise ValueError(f"A2 and A3 need one policy digest in the caption table, but it holds {len(digests)}.")
    policy = load_approved_policy(policy_path)
    digest = next(iter(digests))
    if digest != policy.digest:
        raise ValueError(f"The caption table's policy digest {digest[:12]} is not the approved policy's "
                         f"{policy.digest[:12]}: rebuild the captions.")
    return digest


def fit_captions(captions, tokenizer, max_fraction=0.01):
    """(the captions the encoder reads whole, a report per structure). See the module docstring."""
    tokens = tokenizer.lengths(captions["caption"].tolist())
    table = captions.assign(tokens=tokens, over=[n > tokenizer.max_length for n in tokens])
    report = {"max_length": tokenizer.max_length, "max_overlength_fraction": max_fraction, "structures": {}}
    for structure, rows in table.groupby("structure", sort=True):
        entry = {"rows": len(rows), "overlength": int(rows.over.sum()), "longest": int(rows.tokens.max())}
        if "ef_status" in rows.columns:
            entry["overlength_with_appended_ef"] = int((rows.over & (rows.ef_status == "appended")).sum())
        report["structures"][structure] = entry
    worst = sorted(s for s, e in report["structures"].items() if e["overlength"] / e["rows"] > max_fraction)
    if worst:
        counts = {s: report["structures"][s]["overlength"] for s in worst}
        raise ValueError(f"Captions over text.max_length = {tokenizer.max_length} exceed {max_fraction:.1%} of "
                         f"{worst}: {counts}. Raise max_length.")
    return captions[~table.over.to_numpy()], report


def build_model(config, routing):
    m, t, lc = config["model"], config["text"], config.get("loss", {})
    if not isinstance(lc.get("use_exact"), bool):
        raise ValueError("loss.use_exact must be set to true or false: false is A0 (identity targets, CLIP InfoNCE), "
                         "true is A1 (identity plus exact-caption targets) and the base of A2 and A3. A default "
                         "would turn a config that forgot it into the other arm.")
    trained = [s.name for s in routing.trained]
    still_model, heads_config = m.get("stills"), config.get("heads")
    still_data = config["data"].get("stills")
    if bool(still_model) != bool(still_data) or (heads_config and not still_data):
        raise ValueError("data.stills and model.stills go together, and heads need them.")
    if still_model and not routing.still_tags:
        raise ValueError("Stills need a routing registry with still_structs (configs/radar/radar_routing_v1.yaml).")
    cond_dim = len(still_data.get("cond_columns", [])) if still_data else 0
    still_kwargs = {}
    if still_model:
        still_kwargs = {"still_token_grid": tuple(still_model["token_grid"]),
                        "num_still_types": len(routing.still_types),
                        "still_cond_dim": cond_dim if still_model.get("cond", True) else 0,
                        "still_heatmap_bias": bool(still_model.get("heatmap_bias", False)),
                        "demo_dim": DEMO_DIM if still_model.get("demo_film", False) else 0}
    pooler = StructurePooler(
        embed_dim=m["embed_dim"], num_structures=len(trained), token_grid=tuple(m["token_grid"]),
        num_view_classes=len(routing.view_classes),
        num_acquisition_classes=len(config["data"].get("acquisition_classes", ("bmode", "color_flow"))),
        depth=m.get("depth", 2), num_heads=m.get("num_heads", 12), mlp_ratio=m.get("mlp_ratio", 4.0),
        proj_dim=m.get("proj_dim", 256), use_activation_checkpointing=m.get("activation_checkpointing", False),
        **still_kwargs)
    heads = None
    if heads_config:
        heads = StillMeasurementHeads(m["embed_dim"], heads_config["targets"], still_cond_dim=cond_dim,
                                      demo_dim=DEMO_DIM if heads_config.get("demographics", True) else 0,
                                      hidden_dim=heads_config.get("hidden_dim", 256))
    text_encoder, tokenizer = build_text_encoder(t, m.get("proj_dim", 256))
    loss = StructureContrastiveLoss(
        trained, use_exact=lc["use_exact"], use_normal=lc.get("use_normal", False),
        use_abnormal_teacher=lc.get("use_abnormal_teacher", False), release_gate=lc.get("release_gate", False),
        min_pairs=lc.get("min_pairs", 2), init_temperature=lc.get("init_temperature", 0.07),
        temperature_bounds=tuple(lc.get("temperature_bounds", (0.001, 0.5))))
    if heads is None:
        return RadarModel(pooler, text_encoder, loss), tokenizer
    return RadarModel(pooler, text_encoder, loss, heads, heads_config.get("weight", 1.0),
                      heads_config.get("target_weights")), tokenizer


def read_stills(section, routing, split, heads=None, scaler=None, demo_stats=None):
    """The still source of one split from `data.stills` (see the module docstring). The targets' scaler and the age
    statistics are fitted on the split given here unless passed (evaluation passes the training split's)."""
    table = pd.read_csv(section["table"], dtype={"study_id": str, "still_id": str, "still_tag": str,
                                                 "still_type": str})
    if "split" in table.columns:
        table = table[table.split.astype(str) == split]
    demographics = (pd.read_csv(section["demographics"], dtype={"study_id": str})
                    if section.get("demographics") else None)
    cache = TokenCache(section["cache_dir"])
    return StillSource(table, routing, cache, [s.name for s in routing.trained],
                       num_stills=section.get("num_stills", 8), cond_columns=section.get("cond_columns", []),
                       target_columns=(heads or {}).get("targets", []), scaler=scaler,
                       heatmap_column=section.get("heatmap_column"), demographics=demographics, demo_stats=demo_stats)


def param_groups(model, opt):
    """AdamW groups: vision side (pooler, temperature) at `lr`, caption encoder at `text_lr`; weight decay only on
    matrices that are not embeddings, learned queries or biases."""
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        part = "text" if name.startswith("text.") else "vision"
        no_decay = p.ndim < 2 or "embed" in name or name.endswith(("queries", "bias"))    # `proj_bias` is 2-D
        groups.setdefault((part, "no_decay" if no_decay else "decay"), []).append(p)
    out = []
    for (part, kind), params in sorted(groups.items()):
        base = opt["text_lr"] if part == "text" else opt["lr"]
        out.append({"params": params, "lr": base, "base_lr": base, "name": f"{part}_{kind}",
                    "weight_decay": opt.get("weight_decay", 0.05) if kind == "decay" else 0.0})
    return out


def learning_rate(base, step, schedule, min_ratio):
    """The learning rate of zero-indexed `step`: linear warmup reaching `base` at step `warmup - 1`, then cosine from
    `base` at step `warmup` to `base * min_ratio` at the last step, `total - 1` (when the last step is the only one
    after warmup, it is at the floor)."""
    warmup, total = schedule["warmup_steps"], schedule["total_steps"]
    floor = base * min_ratio
    if warmup and step < warmup:
        return floor + (base - floor) * (step + 1) / warmup
    span = total - 1 - warmup
    progress = 1.0 if span <= 0 else min(1.0, (step - warmup) / span)
    return floor + (base - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_loader(dataset, sampler, tokenizer, num_workers=0, start_method="spawn"):
    """The training DataLoader. With workers, they start by `start_method` (spawn or forkserver, never fork: see the
    module docstring) and are not persistent, so each epoch's workers get the dataset's current epoch."""
    if start_method not in WORKER_START_METHODS:
        raise ValueError(f"worker_start_method must be one of {WORKER_START_METHODS}, not {start_method!r}.")
    workers = {"multiprocessing_context": start_method, "persistent_workers": False} if num_workers > 0 else {}
    return torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers,
                                       collate_fn=partial(collate, tokenizer=tokenizer), **workers)


def _to(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def train_step(wrapped, model, teacher, batch, optimizer, amp):
    """One global step. Returns (loss, stats, contributed)."""
    with torch.autocast(device_type=batch["tokens"].device.type, dtype=torch.bfloat16, enabled=amp == "bf16"):
        loss, stats, _ = wrapped(batch, teacher)
    loss.backward()
    contributed = any(s["loss"] is not None for s in stats.values())
    if model.last_measurement is not None:
        contributed = contributed or any(s["n"] > 0 for s in model.last_measurement.values())
    if contributed:
        optimizer.step()
        model.loss.clamp_temperature_()
        if teacher is not None:
            teacher.update(model.text)
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach()), stats, contributed


def _save(path, state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def _seed_rank(seed, rank):
    """Seed this rank's torch (CPU and CUDA) and Python streams from (seed, rank)."""
    value = int.from_bytes(hashlib.sha256(f"radar:{seed}:{rank}".encode()).digest()[:8], "big")
    torch.manual_seed(value)
    random.seed(value)


def _rng_state(device):
    """This rank's RNG states: CPU torch, Python, and its CUDA generator on GPU."""
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def _set_rng_state(state, device):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def _all_rng_states(device, world_size):
    """Every rank's RNG states, in rank order (a collective: every rank calls it)."""
    state = _rng_state(device)
    if world_size == 1:
        return [state]
    states = [None] * world_size
    dist.all_gather_object(states, state)
    return states


def _keep_jsonl(path, keep):
    """Rewrite a JSON-lines log with only the records `keep` accepts (all dropped when `keep` is None)."""
    if not os.path.exists(path):
        return
    lines = []
    if keep is not None:
        with open(path) as f:
            lines = [line for line in f if line.strip() and keep(json.loads(line))]
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.writelines(lines)
    os.replace(tmp, path)


def train(config, out_dir, resume=True, stop_after_epoch=None, allow_code_change=False):
    """Train (or resume) and return a summary. `stop_after_epoch` ends this call after that many completed epochs
    without changing the schedule (run control, as for a preempted job)."""
    world_size, rank = _world()
    device = _device(config)
    opt = config["optimization"]
    amp = opt.get("amp", "none")
    if amp not in AMP_MODES:
        raise ValueError(f"amp must be one of {AMP_MODES}.")
    seed = int(opt.get("seed", 0))
    torch.manual_seed(seed)
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    data = config["data"]
    if "view_families" in data:
        raise ValueError("data.view_families was removed: the routing registry holds the view family map. "
                         "To change it, point data.routing at another registry.")
    routing = load_routing(data.get("routing") or DEFAULT_ROUTING)
    m = config["model"]
    cache = TokenCache(data["cache_dir"], token_grid=m["token_grid"], embed_dim=m["embed_dim"])
    split = data.get("split", "train")
    clips = read_table(data["clips"])
    if "split" in clips.columns:
        clips = clips[clips.split.astype(str) == split]
    build = check_caption_build(data["captions"], routing, data["clips"], split)
    captions = read_table(data["captions"], id_columns=("study_id",))
    if captions.empty:
        raise ValueError("The caption table is empty.")
    normality_digest = check_normality_approved(captions, config.get("loss", {}), data.get("normality_policy"))
    captions, lengths = fit_captions(captions, build_tokenizer(config["text"]),
                                     config["text"].get("max_overlength_fraction", 0.01))
    stills = read_stills(data["stills"], routing, split, config.get("heads")) if data.get("stills") else None
    if stills is not None:
        same_encoder(cache.meta, stills.cache.meta)
    dataset = RadarStudyDataset(clips, captions, routing, cache, num_clips=data.get("num_clips", 12),
                                selection=data.get("selection", "view_balanced"), training=True, seed=seed,
                                acquisition_classes=data.get("acquisition_classes", ("bmode", "color_flow")),
                                normal_fraction=data.get("normal_fraction", 1.0), stills=stills)
    global_batch = int(opt["global_batch"])
    sampler = StudyBatchSampler(len(dataset), global_batch, rank, world_size, seed)
    schedule = step_schedule(len(dataset), global_batch, opt["epochs"], opt.get("warmup_fraction", 0.05))
    record = fingerprint(config, routing, cache, len(dataset), normality_digest, {split: dataset.bag_record()},
                         build.get("policy_digest"), stills.record() if stills is not None else None)
    code = code_digest()

    model, tokenizer = build_model(config, routing)
    model.to(device)
    optimizer = torch.optim.AdamW(param_groups(model, opt), lr=opt["lr"])
    check_optimizer_coverage(model, optimizer)
    wrapped = model
    if world_size > 1:
        wrapped = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    teacher = None
    if model.loss.use_abnormal_teacher:
        teacher = MomentumTeacher(model.text, config.get("loss", {}).get("teacher_momentum", 0.995)).to(device)

    _seed_rank(seed, rank)

    start_epoch, step, optimizer_steps, skipped = 0, 0, 0, 0
    wandb_id, state = None, {}
    path = os.path.join(out_dir, CHECKPOINT)
    if resume and os.path.exists(path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["fingerprint"]["sha256"] != record["sha256"]:
            raise ValueError(f"{path} was written by a different run (fingerprint differs); use a new out_dir.")
        if state["world_size"] != world_size or state["device_type"] != device.type:
            raise ValueError(f"{path} was written by {state['world_size']} {state['device_type']} process(es), not "
                             f"{world_size} {device.type}: an exact resume needs the same; use a new out_dir.")
        if state["code_sha256"] != code and not allow_code_change:
            raise ValueError(f"{path} was written by other RADAR training code (code digest differs); resume with "
                             "allow_code_change only for a change that does not alter training.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if teacher is not None:
            teacher.load_state_dict(state["teacher"])
        start_epoch, step = state["epoch"], state["step"]
        optimizer_steps, skipped = state["optimizer_steps"], state["skipped_steps"]
        _set_rng_state(state["rng"][rank], device)
    wandb_run, wandb_id = init_wandb(config.get("meta") or {}, config, rank, out_dir,
                                     resuming_training=bool(start_epoch), checkpoint_run_id=state.get("wandb_run_id"),
                                     checkpoint_has_wandb_run_id="wandb_run_id" in state,
                                     settings=PRIVATE_WANDB_SETTINGS)

    loader = make_loader(dataset, sampler, tokenizer, opt.get("num_workers", 0),
                         opt.get("worker_start_method", "spawn"))
    context = loader.multiprocessing_context
    min_ratio = opt.get("min_lr", 1e-6) / opt["lr"]
    losses = []
    last_epoch = opt["epochs"] if stop_after_epoch is None else min(opt["epochs"], stop_after_epoch)
    if rank == 0:
        with open(os.path.join(out_dir, "fingerprint.json"), "w") as f:
            json.dump({**record, "code_sha256": code}, f, indent=2, sort_keys=True, default=str)
        with open(os.path.join(out_dir, "caption_lengths.json"), "w") as f:
            json.dump(lengths, f, indent=2, sort_keys=True)
        resumed_at = (step, start_epoch) if start_epoch else None
        if not start_epoch:         # a fresh run: nothing older may be resumed from if it dies in its first epoch
            for stale in (path, os.path.join(out_dir, "resumes.jsonl")):
                if os.path.exists(stale):
                    os.remove(stale)
        _keep_jsonl(os.path.join(out_dir, "log.jsonl"), resumed_at and (lambda r: r["step"] < resumed_at[0]))
        _keep_jsonl(os.path.join(out_dir, "epochs.jsonl"), resumed_at and (lambda r: r["epoch"] < resumed_at[1]))
        if start_epoch:
            with open(os.path.join(out_dir, "resumes.jsonl"), "a") as f:
                f.write(json.dumps({"epoch": start_epoch, "step": step, "checkpoint_code_sha256": state["code_sha256"],
                                    "code_sha256": code}) + "\n")
    for epoch in range(start_epoch, last_epoch):
        model.train()
        sampler.set_epoch(epoch)
        dataset.set_epoch(epoch)
        contributing = {name: [] for name in model.loss.structures}
        for batch in loader:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(group["base_lr"], step, schedule, min_ratio)
            loss, stats, contributed = train_step(wrapped, model, teacher, _to(batch, device), optimizer, amp)
            losses.append(loss)
            optimizer_steps += int(contributed)
            skipped += int(not contributed)
            for name, stat in stats.items():
                if stat["loss"] is not None:
                    contributing[name].append((stat["rows"], stat["exact_pairs"], stat["normal_pairs"]))
            if rank == 0:
                line = {"epoch": epoch, "step": step, "loss": loss, "contributed": contributed,
                        "temperature": float(model.loss.temperature.detach()),
                        "lr": {g["name"]: g["lr"] for g in optimizer.param_groups},
                        "structures": {n: {"rows": s["rows"], "skipped": s["skipped"],
                                           "exact_pairs": s["exact_pairs"], "normal_pairs": s["normal_pairs"]}
                                       for n, s in stats.items()}}
                if model.last_measurement is not None:
                    line["measurement"] = model.last_measurement
                with open(os.path.join(out_dir, "log.jsonl"), "a") as f:
                    f.write(json.dumps(line) + "\n")
                log_scalars(wandb_run, {"train/loss": loss, "train/temperature": line["temperature"],
                                        "train/contributed": int(contributed), "train/epoch": epoch,
                                        **{f"lr/{name}": lr for name, lr in line["lr"].items()}}, step=step)
            step += 1
        rng = _all_rng_states(device, world_size)
        if rank == 0:
            summary = {"epoch": epoch, "structures": {
                name: {"contributing_steps": len(steps),
                       "median_rows": statistics.median(r for r, _, _ in steps) if steps else None,
                       "median_exact_pairs": statistics.median(p for _, p, _ in steps) if steps else None,
                       "median_normal_pairs": statistics.median(p for _, _, p in steps) if steps else None}
                for name, steps in contributing.items()}}
            with open(os.path.join(out_dir, "epochs.jsonl"), "a") as f:
                f.write(json.dumps(summary) + "\n")
            log_scalars(wandb_run, {f"structures/{name}/{key}": value
                                    for name, entry in summary["structures"].items()
                                    for key, value in entry.items() if value is not None}, step=step)
            _save(path, {"model": model.state_dict(), "teacher": teacher.state_dict() if teacher else None,
                         "optimizer": optimizer.state_dict(), "epoch": epoch + 1, "step": step,
                         "optimizer_steps": optimizer_steps, "skipped_steps": skipped, "fingerprint": record,
                         "code_sha256": code, "world_size": world_size, "device_type": device.type, "rng": rng,
                         "wandb_run_id": wandb_id})
        if world_size > 1:
            dist.barrier()
    finish_wandb(wandb_run)
    return {"epochs_completed": last_epoch, "steps": step, "optimizer_steps": optimizer_steps,
            "skipped_steps": skipped, "losses": losses, "schedule": schedule, "fingerprint": record["sha256"],
            "loader_workers": {"num_workers": loader.num_workers,
                               "start_method": context.get_start_method() if context is not None else None},
            "model": model, "teacher": teacher}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--allow_code_change", action="store_true",
                        help="resume although the RADAR training code changed (only for changes that do not alter "
                             "training)")
    args = parser.parse_args(argv)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    started = False
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        cuda = torch.cuda.is_available() and config.get("device", "auto") != "cpu"
        dist.init_process_group("nccl" if cuda else "gloo")
        started = True
    try:
        summary = train(config, args.out_dir, resume=not args.no_resume, allow_code_change=args.allow_code_change)
        if _world()[1] == 0:
            print(json.dumps({k: v for k, v in summary.items() if k not in ("model", "teacher", "losses")}))
    finally:
        if started:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
