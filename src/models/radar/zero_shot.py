# src/models/radar/zero_shot.py

"""Zero-shot scoring of RADAR structure embeddings with caption-style prompts, without a task head.

Three prompt modes, as in the prompt files of the two reference papers:

- `binary` (RADAR): a positive and a negative prompt set. Each set is encoded, mean-pooled and re-normalised; the
  probability of the finding is the softmax over the two cosine similarities, scaled by the temperature
  (`binary_probability`). RADAR's supplement reports 0.894 AUC with one prompt and 0.933 with 500 or more, so a set
  holds many prompts.
- `categorical`: one prompt set per ordered grade; softmax over the grades' mean-pooled embeddings, the grade
  prediction being the argmax or the expectation of the grade values (`categorical_probabilities`).
- `regression` (EchoCLIP, Methods): a template with a `<#>` slot is filled with every value of a range, the prompts are
  encoded, the values are ranked by similarity, and the median of the best `keep` fraction (0.2 in EchoCLIP) is the
  estimate (`regression_estimate`). EchoCLIP does this per frame and then pools; here one study embedding per
  structure is scored.

`PromptTask` reads one task of a prompt file (`configs/radar/prompts_v0.yaml`). `held_out` prompts (EchoCLIP's
`testers`) are never used to choose settings, only to report. The wording of a prompt is a clinical choice: a file
holds only wording that a clinician has approved, and a task without approved wording is not in it.

Inputs are L2-normalised embeddings in the shared space: structure embeddings [N, P] from the pooler and prompt
embeddings [M, P] from the caption encoder (`encode_prompts`). Everything is computed in float32.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import yaml

MODES = ("binary", "categorical", "regression")
SLOT = "<#>"


@dataclass(frozen=True)
class PromptTask:
    name: str
    mode: str
    structure: str
    label_sources: tuple
    values: tuple = ()               # categorical: grade of each prompt; regression: unused
    value_range: tuple = ()          # regression: inclusive integer range filling SLOT
    held_out: tuple = ()
    negative_sources: tuple = field(default=())   # binary: the negative prompt set


def load_prompt_tasks(path):
    """{name: PromptTask} of a prompt file; refuses a task that cannot be scored as written."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or not isinstance(raw.get("tasks"), dict) or not raw["tasks"]:
        raise ValueError(f"{path}: needs a non-empty `tasks` mapping.")
    tasks = {}
    for name, spec in raw["tasks"].items():
        mode = spec.get("mode")
        if mode not in MODES:
            raise ValueError(f"{name}: mode must be one of {MODES}, got {mode!r}.")
        sources = tuple(spec.get("label_sources") or ())
        if not sources or not all(isinstance(s, str) and s.strip() for s in sources):
            raise ValueError(f"{name}: label_sources must be non-empty text.")
        if not isinstance(spec.get("structure"), str):
            raise ValueError(f"{name}: needs the `structure` whose embedding it scores.")
        values, value_range, negative = (), (), tuple(spec.get("negative_sources") or ())
        if mode == "binary" and not negative:
            raise ValueError(f"{name}: a binary task needs negative_sources.")
        if mode == "categorical":
            values = tuple(spec.get("values") or ())
            if len(values) != len(sources):
                raise ValueError(f"{name}: values must have one entry per label source.")
        if mode == "regression":
            value_range = tuple(spec.get("range") or ())
            if len(value_range) != 2 or value_range[0] >= value_range[1]:
                raise ValueError(f"{name}: a regression task needs range [low, high] with low < high.")
            if any(SLOT not in s for s in sources):
                raise ValueError(f"{name}: every regression prompt needs the {SLOT} slot.")
        tasks[name] = PromptTask(name, mode, spec["structure"], sources, values, value_range,
                                 tuple(spec.get("held_out") or ()), negative)
    return tasks


def regression_prompts(templates, low, high):
    """[(value, prompt)] for every integer of [low, high] in every template, values in increasing order."""
    return [(v, t.replace(SLOT, str(v))) for t in templates for v in range(int(low), int(high) + 1)]


@torch.no_grad()
def encode_prompts(text_encoder, tokenizer, prompts, batch_size=256):
    """L2-normalised embeddings [M, P] of `prompts`, in order, with the encoder in eval mode."""
    was_training = text_encoder.training
    text_encoder.eval()
    try:
        out = []
        device = next(text_encoder.parameters()).device
        for start in range(0, len(prompts), batch_size):
            ids, mask = tokenizer(list(prompts[start:start + batch_size]))
            out.append(text_encoder(ids.to(device), mask.to(device)).float())
        return torch.cat(out)
    finally:
        text_encoder.train(was_training)


def pool(embeddings):
    """The mean of a prompt set, L2-normalised: RADAR's prompt ensemble."""
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("A prompt set needs at least one embedding [M, P].")
    return F.normalize(embeddings.float().mean(dim=0), dim=-1)


def binary_probability(z, positive, negative, temperature):
    """P(finding) [N] from structure embeddings z [N, P] and the two prompt sets [M, P]."""
    z = z.float()
    logits = torch.stack([z @ pool(negative), z @ pool(positive)], dim=-1) / float(temperature)
    return F.softmax(logits, dim=-1)[:, 1]


def categorical_probabilities(z, grade_sets, temperature):
    """Softmax over grades [N, G]: one prompt set [M_g, P] per grade, in grade order."""
    centres = torch.stack([pool(g) for g in grade_sets])
    return F.softmax(z.float() @ centres.T / float(temperature), dim=-1)


def regression_estimate(z, prompt_embeddings, values, keep=0.2):
    """Value estimate [N]: the median of the `keep` fraction of values whose prompts are most similar to each row.

    prompt_embeddings [M, P] and values [M] come from `regression_prompts` (a value may appear once per template)."""
    if not 0.0 < keep <= 1.0:
        raise ValueError("keep must be in (0, 1].")
    values = torch.as_tensor(values, dtype=torch.float32)
    if prompt_embeddings.shape[0] != values.shape[0]:
        raise ValueError("One value per prompt embedding.")
    sims = z.float() @ prompt_embeddings.float().T                      # [N, M]
    top = max(1, int(round(keep * values.shape[0])))
    best = sims.topk(top, dim=-1).indices                              # [N, top]
    return values[best].median(dim=-1).values
