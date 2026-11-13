"""Convert a DeepCORO research bundle into the weights of the region probe (`pooling_mode: region`).

Run from the repository root:

    python -m tools.convert_deepcoro_bundle --bundle <bundle.pt> --config <config.yaml> --out <weights.pt>

The bundle is the checkpoint of DeepCORO's two-pass research trainer (`bundle_schema`
deepcoro.integrated_study.multilayer.v2): one state dict holding the encoder, the mask decoder
and the study heads. The output holds three entries, tensors only (read with `weights_only=True`):

- `encoder`: the ViT, for `model_kwargs.checkpoint` (`state_dict_key: encoder`, `encoder_prefix: ""`);
- `region_decoder`: for `region.decoder.checkpoint`;
- `probe`: the rest of the probe, for `region.init_checkpoint`.

The config (e.g. configs/eval/vitl/study_multitask_deepcoro_jepa.yaml) names the groups, readouts
and heads; the bundle's layout is fixed here, and the two must describe the same model:

- `head.<i>`: the one `group_mlp` readout. Group i's regression head takes output 0 (the bundle
  predicts the target divided by 100, rescaled here to the head's `target_mean`/`target_std`)
  and its binary head output 1;
- `lesion_head`: the `group_linear` readouts in the config's order, one head per group each;
- `lesion_pooler`: `region.study_pooler` (every `group_linear` readout then has a `study_head`);
- `physiology_head`: the one `groups_mlp` readout, its `heads` in the bundle's output order (two
  values squashed into [0, 1.05], i.e. `region.output_range`, then their two logits).

A bundle part the config has no place for, a config readout the bundle has no weights for, and
any difference in groups, decoder blocks or sizes are errors: nothing is left at its initial value.
The one exception is a bundle trained before the study pooler, converted with a config that has
`region.study_pooler`: its `probe` entry has no pooler, and a probe started from it
(`init_checkpoint`) trains a new one.
"""

import argparse
from pathlib import Path

import torch
import yaml

from src.models.region_probe import RegionStudyProbe, region_options
from src.models.study_probe import head_specs
from src.utils.checkpoint_loader import robust_checkpoint_loader

SCHEMA = "deepcoro.integrated_study.multilayer.v2"
BUNDLE_SCALE = 100.0  # the bundle's regression output 0 is the target divided by 100.
BUNDLE_RANGE = [0.0, 1.05]  # the bundle's `physiology_head` values are 1.05 * sigmoid.
RENAMED = {"q": "crop_query", "mask_w": "mask_weight", "type_emb": "type_embedding", "group_emb": "group_embedding"}
PREFIXES = {"att_v.": "attention.V.", "att_u.": "attention.U.", "att_w.": "attention.w.", "view_proj.": "view_proj.",
            "lesion_pooler.": "study_pooler."}


def _take(state, prefix):
    """Remove the entries of `state` under `prefix` and return them without it."""
    return {k[len(prefix):]: state.pop(k) for k in [k for k in state if k.startswith(prefix)]}


def _one(readouts, kind, part):
    """The name of the config's only readout of `kind`, which the bundle's `part` fills."""
    names = [name for name, readout in readouts.items() if readout["kind"] == kind]
    if len(names) != 1:
        raise ValueError(f"The bundle's `{part}` fills one {kind} readout; the config has {names}.")
    return names[0]


def convert(bundle, config):
    """The `encoder`, `region_decoder` and `probe` entries from a loaded research `bundle`, named
    and checked against the region probe that `config` (a loaded YAML config) describes."""
    classifier = config["experiment"]["classifier"]
    specs = head_specs(classifier)
    options = region_options(classifier, config["experiment"]["data"], config["model_kwargs"], specs)
    if not isinstance(bundle, dict) or bundle.get("bundle_schema") != SCHEMA or "state_dict" not in bundle:
        raise ValueError(f"Not a research bundle of the schema {SCHEMA}.")
    groups, readouts, task_groups = list(options["groups"]), options["readouts"], options["task_groups"]
    if list(bundle["groups"]) != groups:
        raise ValueError(f"The bundle's groups {list(bundle['groups'])} are not `region.groups` {groups}, in order.")
    decoder = {"blocks": [int(b) for b in bundle["block_indices_zero_based"]], "width": int(bundle["decoder_width"])}
    if decoder != options["decoder"]:
        raise ValueError(f"The bundle's decoder {decoder} is not `region.decoder` {options['decoder']}.")
    means, stds = classifier.get("target_mean") or {}, classifier.get("target_std") or {}
    normalization = {h: {"mean": float(means[h]), "std": float(stds[h])} for h in specs if h in means and h in stds}

    state = dict(bundle["state_dict"])
    weights = {"encoder": _take(state, "enc."), "region_decoder": _take(state, "mask_decoder.")}
    if not weights["encoder"] or not weights["region_decoder"]:
        raise ValueError("The bundle holds no encoder (`enc.*`) or no mask decoder (`mask_decoder.*`).")
    probe = {new: state.pop(old) for old, new in RENAMED.items() if old in state}
    for old, new in PREFIXES.items():
        probe.update({new + k: v for k, v in _take(state, old).items()})

    def fill(head, weight, bias):
        probe[f"heads.{head}.1.weight"], probe[f"heads.{head}.1.bias"] = weight.clone(), bias.clone()

    def group_heads(name, group):
        return [h for h in readouts[name]["heads"] if task_groups[h] == group]

    part = _take(state, "head.")
    if part:
        name = _one(readouts, "group_mlp", "head")
        for gi, group in enumerate(groups):
            for layer in ("0.weight", "0.bias", "1.weight", "1.bias"):
                probe[f"trunks.{name}.{group}.{layer}"] = part.pop(f"{gi}.{layer}")
            weight, bias = part.pop(f"{gi}.4.weight"), part.pop(f"{gi}.4.bias")
            tasks = sorted(specs[h]["task"] for h in group_heads(name, group))
            if tasks != ["binary", "regression"]:
                raise ValueError(f"The bundle's `head.{gi}` gives one regression and one binary head of the group "
                                 f"{group!r}; the readout {name!r} has {group_heads(name, group)}.")
            for h in group_heads(name, group):
                if specs[h]["task"] == "binary":
                    fill(h, weight[1:2], bias[1:2])
                elif h not in normalization:
                    raise ValueError(f"Set `target_mean.{h}` and `target_std.{h}`: the bundle's output is rescaled "
                                     "to them.")
                else:
                    mean, std = normalization[h]["mean"], normalization[h]["std"]
                    fill(h, weight[0:1] * (BUNDLE_SCALE / std), (bias[0:1] * BUNDLE_SCALE - mean) / std)
        if part:
            raise ValueError(f"The bundle's `head` has entries with no place in the config: {sorted(part)[:8]}.")

    part = _take(state, "lesion_head.")
    linear = [name for name, readout in readouts.items() if readout["kind"] == "group_linear"]
    if part:
        if len(linear) != part["weight"].shape[0]:
            raise ValueError(f"The bundle's `lesion_head` has {part['weight'].shape[0]} outputs; the config has the "
                             f"group_linear readouts {linear}.")
        for k, name in enumerate(linear):
            probe[f"trunks.{name}.weight"], probe[f"trunks.{name}.bias"] = (part["norm.weight"].clone(),
                                                                           part["norm.bias"].clone())
            for gi, group in enumerate(groups):
                heads = group_heads(name, group)
                if len(heads) != 1:
                    raise ValueError(f"The bundle's `lesion_head` gives one head per group; the readout {name!r} "
                                     f"has {heads} for {group!r}.")
                fill(heads[0], part["weight"][k, gi][None], part["bias"][k, gi][None])
    if any(k.startswith("study_pooler.") for k in probe) and any(
            readouts[name].get("study_head") is None for name in linear):
        raise ValueError("The bundle's `lesion_pooler` gives every group_linear readout a `study_head`; the config "
                         f"gives one to {[n for n in linear if readouts[n].get('study_head') is not None]} only.")

    part = _take(state, "physiology_head.")
    if part:
        name = _one(readouts, "groups_mlp", "physiology_head")
        physiology = bundle.get("physiology") or {}
        read = [groups[i] for i in physiology.get("group_indices", ())]
        if read != readouts[name]["groups"] or physiology.get("geometry_dim", 0):
            raise ValueError(f"The bundle's `physiology_head` reads the groups {read} (and "
                             f"{physiology.get('geometry_dim', 0)} more inputs); the readout {name!r} reads "
                             f"{readouts[name]['groups']}.")
        heads, tasks = readouts[name]["heads"], [specs[h]["task"] for h in readouts[name]["heads"]]
        if tasks != ["regression", "regression", "binary", "binary"]:
            raise ValueError(f"The bundle's `physiology_head` gives two values then their two logits; the readout "
                             f"{name!r} lists {dict(zip(heads, tasks))}.")
        ranges = [options["output_range"].get(h) for h in heads[:2]]
        if ranges != [BUNDLE_RANGE, BUNDLE_RANGE]:
            raise ValueError(f"The bundle's values lie in {BUNDLE_RANGE}: set `region.output_range` of {heads[:2]} "
                             f"to it, not {ranges}.")
        for layer in ("0.weight", "0.bias", "1.weight", "1.bias"):
            probe[f"trunks.{name}.{layer}"] = part[f"net.{layer}"]
        for i, h in enumerate(heads):
            fill(h, part["net.4.weight"][i:i + 1], part["net.4.bias"][i:i + 1])
    if state:
        raise ValueError(f"The bundle has parts this converter does not know: {sorted(state)[:8]}.")

    # The probe the config builds must take every weight, and be filled entirely by them.
    built = RegionStudyProbe(embed_dim=weights["encoder"]["norm.weight"].shape[0], heads=specs,
                             attention_hidden=int(classifier.get("attention_hidden", 256)),
                             dropout=float(classifier.get("dropout", 0.0)), normalization=normalization, **options)
    if built.load_weights(weights["region_decoder"], probe):
        print("The bundle has no `lesion_pooler`: the `probe` entry holds no study pooler.")
    return {**weights, "probe": probe}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True, help="The research bundle (read with weights_only=True)")
    parser.add_argument("--config", required=True, help="The probe's YAML config, with its `region` block")
    parser.add_argument("--out", required=True, help="Where to write the converted weights; never overwritten")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"{out} exists; converted weights are not overwritten.")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    bundle = robust_checkpoint_loader(args.bundle, map_location="cpu", max_retries=1, weights_only=True)
    weights = convert(bundle, config)
    torch.save(weights, out)
    print(f"Wrote {out}: " + ", ".join(f"{len(v)} {k} tensors" for k, v in weights.items()))


if __name__ == "__main__":
    main()
