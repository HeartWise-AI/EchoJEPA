# tests/utils/test_convert_deepcoro_bundle.py

"""`tools/convert_deepcoro_bundle.py` on a synthetic bundle in the research layout (16-dim tokens),
with the shipped DeepCORO-JEPA config: every weight lands where the research formulas read it."""

import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F
import yaml

from evals.study_multitask_frozen.eval import load_probe_checkpoint
from src.models.region_probe import RegionMaskDecoder, RegionStudyProbe, StudyTokenPooler, region_options
from src.models.study_probe import head_specs
from tools.convert_deepcoro_bundle import SCHEMA, convert, main

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/eval/vitl/study_multitask_deepcoro_jepa.yaml"
DIM, FINDINGS = 16, ("calcif", "thrombus", "cto")


def shipped_config():
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def research_bundle(config, lesions=True, pooler=True):
    """A random bundle with the research trainer's keys and shapes, for `DIM`-wide tokens."""
    region = config["experiment"]["classifier"]["region"]
    groups, d = list(region["groups"]), DIM
    generator = torch.Generator().manual_seed(0)

    def r(*shape):
        return torch.randn(*shape, generator=generator)

    def randomized(module):
        return {k: r(*v.shape) for k, v in module.state_dict().items()}

    state = {"enc.norm.weight": r(d), "enc.norm.bias": r(d), "enc.blocks.0.attn.qkv.weight": r(3 * d, d)}
    decoder = RegionMaskDecoder(d, region["num_classes"], region["decoder"]["width"], region["decoder"]["blocks"])
    state.update({f"mask_decoder.{k}": v for k, v in randomized(decoder).items()})
    state.update({"q": r(len(groups), d), "mask_w": r(len(groups)), "type_emb": r(2, d),
                  "group_emb": r(len(groups), d), "att_w.weight": r(1, 128), "att_w.bias": r(1),
                  "view_proj.0.weight": r(2 * d + 2), "view_proj.0.bias": r(2 * d + 2),
                  "view_proj.1.weight": r(d, 2 * d + 2), "view_proj.1.bias": r(d),
                  "view_proj.3.weight": r(d, d), "view_proj.3.bias": r(d)})
    for name in ("att_v", "att_u"):
        state.update({f"{name}.weight": r(128, d), f"{name}.bias": r(128)})
    for i in range(len(groups)):
        state.update({f"head.{i}.0.weight": r(d), f"head.{i}.0.bias": r(d), f"head.{i}.1.weight": r(256, d),
                      f"head.{i}.1.bias": r(256), f"head.{i}.4.weight": r(2, 256), f"head.{i}.4.bias": r(2)})
    if lesions:
        state.update({"lesion_head.weight": r(len(FINDINGS), len(groups), d),
                      "lesion_head.bias": r(len(FINDINGS), len(groups)),
                      "lesion_head.norm.weight": r(d), "lesion_head.norm.bias": r(d)})
    if pooler:
        module = StudyTokenPooler(d, len(groups), len(FINDINGS), width=256, queries=4, num_heads=8)
        state.update({f"lesion_pooler.{k}": v for k, v in randomized(module).items()})
    state.update({"physiology_head.net.0.weight": r(6 * d), "physiology_head.net.0.bias": r(6 * d),
                  "physiology_head.net.1.weight": r(256, 6 * d), "physiology_head.net.1.bias": r(256),
                  "physiology_head.net.4.weight": r(4, 256), "physiology_head.net.4.bias": r(4)})
    return {"bundle_schema": SCHEMA, "groups": tuple(groups), "decoder_width": region["decoder"]["width"],
            "block_indices_zero_based": list(region["decoder"]["blocks"]), "epoch": 1,
            "physiology": {"feature_dim": 6 * d, "geometry_dim": 0, "group_indices": (5, 6, 7, 8, 9, 10)},
            "state_dict": state}


def build_probe(config, weights):
    classifier = config["experiment"]["classifier"]
    specs = head_specs(classifier)
    normalization = {h: {"mean": classifier["target_mean"][h], "std": classifier["target_std"][h]}
                     for h in classifier["target_mean"]}
    probe = RegionStudyProbe(embed_dim=DIM, heads=specs, attention_hidden=classifier["attention_hidden"],
                             normalization=normalization,
                             **region_options(classifier, config["experiment"]["data"], config["model_kwargs"], specs))
    probe.load_weights(weights["region_decoder"], weights["probe"])
    return probe.eval()


def without_lesions(config, keep_pooler=False):
    """The shipped config without its lesion readouts (or only without their study heads)."""
    config = copy.deepcopy(config)
    classifier = config["experiment"]["classifier"]
    region = classifier["region"]
    dropped = [f"any_{f}" for f in FINDINGS]
    for finding in FINDINGS:
        if keep_pooler:
            del region["readouts"][finding]["study_head"]
        else:
            dropped += region["readouts"].pop(finding)["heads"]
    del region["study_pooler"]
    classifier["target_label"] = [h for h in classifier["target_label"] if h not in dropped]
    for block in ("head_task", "head_weights", "task_groups"):
        classifier[block] = {h: v for h, v in classifier[block].items() if h not in dropped}
    return config


class TestConvertDeepcoroBundle(unittest.TestCase):

    def test_every_weight_lands_where_the_research_formulas_read_it(self):
        config = shipped_config()
        bundle = research_bundle(config)
        state = bundle["state_dict"]
        weights = convert(bundle, config)
        self.assertEqual(sorted(weights), ["encoder", "probe", "region_decoder"])
        self.assertEqual(set(weights["encoder"]), {"norm.weight", "norm.bias", "blocks.0.attn.qkv.weight"})
        for k, v in weights["region_decoder"].items():
            self.assertTrue(torch.equal(v, state[f"mask_decoder.{k}"]), k)
        probe = build_probe(config, weights)
        for old, new in (("q", "crop_query"), ("mask_w", "mask_weight"), ("type_emb", "type_embedding"),
                         ("group_emb", "group_embedding"), ("att_v.weight", "attention.V.weight"),
                         ("att_u.bias", "attention.U.bias"), ("att_w.weight", "attention.w.weight"),
                         ("view_proj.3.weight", "view_proj.3.weight")):
            self.assertTrue(torch.equal(probe.state_dict()[new], state[old]), new)

        generator = torch.Generator().manual_seed(1)
        groups = list(probe.groups)
        features = torch.randn(len(groups), DIM, generator=generator)
        tokens = torch.randn(3, 8, DIM, generator=generator)
        with torch.no_grad():
            outputs = probe.read(features, tokens)

            def close(head, expected, scale=1.0):
                self.assertTrue(torch.allclose(outputs[head].reshape(()) * scale, expected, rtol=1e-4, atol=1e-4),
                                (head, outputs[head], expected))

            # Stenosis: head.<i> = LayerNorm, Linear, GELU, (Dropout,) Linear -> (percent / 100, logit).
            for i, group in enumerate(groups):
                hidden = F.gelu(F.linear(F.layer_norm(features[i], (DIM,), state[f"head.{i}.0.weight"],
                                                      state[f"head.{i}.0.bias"]),
                                         state[f"head.{i}.1.weight"], state[f"head.{i}.1.bias"]))
                prediction = F.linear(hidden, state[f"head.{i}.4.weight"], state[f"head.{i}.4.bias"])
                close(f"{group}_stenosis", prediction[0] * 100, scale=100.0)  # the head's std is 100, its mean 0.
                close("left_main_ge50" if group == "left_main" else f"{group}_ge70", prediction[1])
            # Lesions: (norm(features) * weight).sum(-1) + bias, plus the study pooler's offsets.
            pooler = StudyTokenPooler(DIM, len(groups), len(FINDINGS), width=256, queries=4, num_heads=8)
            pooler.load_state_dict({k[len("lesion_pooler."):]: v for k, v in state.items()
                                    if k.startswith("lesion_pooler.")})
            offsets, study = pooler(tokens)
            normed = F.layer_norm(features, (DIM,), state["lesion_head.norm.weight"], state["lesion_head.norm.bias"])
            logits = (normed[None] * state["lesion_head.weight"]).sum(-1) + state["lesion_head.bias"] + offsets
            for k, finding in enumerate(FINDINGS):
                close(f"any_{finding}", study[k])
                for i, group in enumerate(groups):
                    close(f"{group}_{finding}", logits[k, i])
            # Pressure: the six groups side by side -> (1.05 sigmoid, 1.05 sigmoid, logit, logit).
            hidden = F.gelu(F.linear(F.layer_norm(features[5:11].flatten(), (6 * DIM,),
                                                  state["physiology_head.net.0.weight"],
                                                  state["physiology_head.net.0.bias"]),
                                     state["physiology_head.net.1.weight"], state["physiology_head.net.1.bias"]))
            raw = F.linear(hidden, state["physiology_head.net.4.weight"], state["physiology_head.net.4.bias"])
            close("ffr", raw[0].sigmoid() * 1.05, scale=0.05)  # the head's std is 0.05, its mean 0.
            close("ifr", raw[1].sigmoid() * 1.05, scale=0.05)
            close("ffr_le80", raw[2])
            close("ifr_le89", raw[3])

    def test_the_regression_head_is_rescaled_to_the_config_normalization(self):
        config = shipped_config()
        classifier = config["experiment"]["classifier"]
        classifier["target_mean"]["mid_lad_stenosis"], classifier["target_std"]["mid_lad_stenosis"] = 30.0, 25.0
        bundle = research_bundle(config)
        probe = build_probe(config, convert(bundle, config))
        features, tokens = torch.randn(18, DIM), torch.randn(2, 8, DIM)
        with torch.no_grad():
            z = probe.read(features, tokens)["mid_lad_stenosis"].reshape(())
            reference = build_probe(shipped_config(), convert(bundle, shipped_config())).read(features, tokens)
        self.assertTrue(torch.allclose(z * 25.0 + 30.0, reference["mid_lad_stenosis"].reshape(()) * 100.0, atol=1e-3))
        del classifier["target_std"]["mid_lad_stenosis"], classifier["target_mean"]["mid_lad_stenosis"]
        with self.assertRaisesRegex(ValueError, "Set `target_mean.mid_lad_stenosis`"):
            convert(bundle, config)

    def test_a_bundle_and_a_config_that_differ_are_refused(self):
        config = shipped_config()
        bundle = research_bundle(config)

        def changed(**entries):
            return {**bundle, **entries}

        with self.assertRaisesRegex(ValueError, "Not a research bundle"):
            convert(changed(bundle_schema="another"), config)
        with self.assertRaisesRegex(ValueError, "Not a research bundle"):
            convert({"encoder": {}}, config)
        with self.assertRaisesRegex(ValueError, "are not `region.groups`"):
            convert(changed(groups=tuple(reversed(bundle["groups"]))), config)
        with self.assertRaisesRegex(ValueError, "is not `region.decoder`"):
            convert(changed(block_indices_zero_based=[5, 11]), config)
        with self.assertRaisesRegex(ValueError, "reads the groups"):
            convert(changed(physiology={"geometry_dim": 0, "group_indices": (0, 1, 2, 3, 4, 5)}), config)
        with self.assertRaisesRegex(ValueError, "does not know: \\['procedure_head.weight'\\]"):
            convert(changed(state_dict={**bundle["state_dict"], "procedure_head.weight": torch.zeros(1)}), config)
        with self.assertRaisesRegex(ValueError, "no encoder"):
            convert(changed(state_dict={k: v for k, v in bundle["state_dict"].items() if not k.startswith("enc.")}),
                    config)
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            convert(changed(state_dict={**bundle["state_dict"], "mask_w": torch.zeros(3)}), config)

        other = shipped_config()
        other["experiment"]["classifier"]["region"]["output_range"]["ffr"] = [0.0, 1.0]
        with self.assertRaisesRegex(ValueError, "set `region.output_range` of \\['ffr', 'ifr'\\]"):
            convert(bundle, other)
        other = shipped_config()
        heads = other["experiment"]["classifier"]["region"]["readouts"]["pressure"]["heads"]
        heads[0], heads[2] = heads[2], heads[0]
        with self.assertRaisesRegex(ValueError, "two values then their two logits"):
            convert(bundle, other)

    def test_only_a_new_study_pooler_is_left_at_its_initial_values(self):
        config = shipped_config()
        # The config has lesion readouts that an older bundle has no weights for.
        with self.assertRaisesRegex(ValueError, "do not fit this probe: missing \\['heads.D1_calcif.1.bias'"):
            convert(research_bundle(config, lesions=False, pooler=False), config)
        # A bundle from before the study pooler starts a probe that trains a new one.
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            start = convert(research_bundle(config, pooler=False), config)
        self.assertIn("holds no study pooler", printed.getvalue())
        self.assertFalse(any(k.startswith("study_pooler.") for k in start["probe"]))
        started, fresh = build_probe(config, start), build_probe(config, convert(research_bundle(config), config))
        self.assertEqual(started.study_pooler.segment.abs().sum().item(), 0.0)
        self.assertNotEqual(fresh.study_pooler.segment.abs().sum().item(), 0.0)
        # The bundle has parts the config has no place for.
        with self.assertRaisesRegex(ValueError, "`lesion_head` has 3 outputs; the config has the group_linear "
                                                "readouts \\[\\]"):
            convert(research_bundle(config, pooler=False), without_lesions(config))
        with self.assertRaisesRegex(ValueError, "`lesion_pooler` gives every group_linear readout a `study_head`"):
            convert(research_bundle(config), without_lesions(config, keep_pooler=True))
        # Configs that describe the older bundles convert.
        older = convert(research_bundle(config, lesions=False, pooler=False), without_lesions(config))
        self.assertFalse(any("calcif" in k or k.startswith("study_pooler.") for k in older["probe"]))
        middle = convert(research_bundle(config, pooler=False), without_lesions(config, keep_pooler=True))
        self.assertIn("heads.D1_calcif.1.bias", middle["probe"])
        self.assertFalse(any(k.startswith("study_pooler.") for k in middle["probe"]))

    def test_the_command_writes_weights_the_probe_reads_and_never_overwrites(self):
        config = shipped_config()
        with tempfile.TemporaryDirectory() as folder:
            bundle, out = str(Path(folder) / "bundle.pt"), str(Path(folder) / "weights.pt")
            torch.save(research_bundle(config), bundle)
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                main(["--bundle", bundle, "--config", str(CONFIG), "--out", out])
            self.assertIn("304 probe tensors", printed.getvalue())
            weights = load_probe_checkpoint(out)  # weights_only=True: tensors and plain containers.
            build_probe(config, weights)
            self.assertEqual(weights["encoder"]["norm.weight"].shape, (DIM,))
            with self.assertRaisesRegex(FileExistsError, "not overwritten"):
                main(["--bundle", bundle, "--config", str(CONFIG), "--out", out])


if __name__ == "__main__":
    unittest.main()
