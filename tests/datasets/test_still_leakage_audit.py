"""Synthetic leakage reproductions and explicit unresolved safety contracts.

Passing observation tests reproduce the current vulnerabilities. Strict xfails
state desired safety contracts that the current implementation fails; they are
audit blockers, not a safety pass. An XPASS forces review when behavior changes.
No training, accelerator use, model weights, or patient data are involved.
"""

import random

import numpy as np
import pandas as pd
import pytest
import torch

from src.datasets.radar_stills import StillSource, same_encoder
from src.datasets.still_preprocess import overlay_mask
from src.models.radar.routing import load_routing
from tools.radar_still_leakage_audit import (
    LABEL_BOX, ROWS, VALUE_BOX, caliper_fixture, collect_pixel_evidence,
    estimate_row, has_flat_footprint, label_value_fixture, prepared_gray, textured_pair,
)


@pytest.mark.parametrize("intensity", [255, 200], ids=["white", "grey"])
def test_achromatic_calipers_and_digits_survive_exactly(intensity):
    for row in ROWS:
        clean, marked = caliper_fixture(row, intensity)
        output, report = prepared_gray(marked)
        assert not overlay_mask(marked).any()
        assert report["masked_overlay_px"] == 0
        assert np.array_equal(output, marked[..., 0])
        assert np.array_equal(output[180:200, 155:220], marked[180:200, 155:220, 0])
        assert not np.array_equal(output, clean[..., 0])
        assert estimate_row(output) == row


@pytest.mark.xfail(strict=True, reason="BLOCKER: grey/white measurement geometry reaches encoder input")
@pytest.mark.parametrize("intensity", [255, 200], ids=["white", "grey"])
def test_safety_achromatic_marker_rule_cannot_recover_programmed_rows(intensity):
    recovered = [estimate_row(prepared_gray(caliper_fixture(row, intensity)[1])[0]) == row for row in ROWS]
    # Fixture-only design criterion, not a clinical or real-screen threshold.
    assert np.mean(recovered) <= 0.6


def test_label_only_box_leaves_adjacent_numeric_value_but_complete_boxes_remove_it():
    clean, marked = label_value_fixture()
    label_only, _ = prepared_gray(marked, text_boxes=[LABEL_BOX])
    complete, _ = prepared_gray(marked, text_boxes=[LABEL_BOX, VALUE_BOX])
    ink = marked[..., 0] != clean[..., 0]
    value_ink = ink & (np.indices(ink.shape)[1] >= VALUE_BOX[0])
    assert value_ink.sum() > 20  # nonempty synthetic digit fixture
    assert np.array_equal(label_only[value_ink], marked[..., 0][value_ink])
    assert np.array_equal(complete, clean[..., 0])  # positive removal control
    assert np.all(label_only[ink & ~value_ink] == 40)


@pytest.mark.xfail(strict=True, reason="BLOCKER: label-only boxes leave the numeric caption beyond pad=4")
def test_safety_label_only_caption_box_removes_measurement_digits():
    clean, marked = label_value_fixture()
    assert np.array_equal(prepared_gray(marked, text_boxes=[LABEL_BOX])[0], clean[..., 0])


def test_textured_box_fill_identifies_intervention_and_preserves_box_geometry():
    metrics, _ = collect_pixel_evidence()
    footprint = metrics["box_footprint"]
    assert metrics["label_value"]["value_ink_surviving_label_only_px"] == metrics["label_value"]["value_ink_px"]
    assert footprint["prespecified_rule_accuracy"] == 1.0
    assert footprint["paired_difference_box_bbox_exact"] == footprint["paired_spectra"]
    for index in range(16):
        before, after, (x0, y0, x1, y1), report = textured_pair(index)
        assert np.std(before[y0:y1, x0:x1]) > 20
        assert np.std(after[y0:y1, x0:x1]) == 0
        assert not has_flat_footprint(before)
        assert has_flat_footprint(after)
        outside = np.ones(before.shape, dtype=bool)
        outside[y0:y1, x0:x1] = False
        assert np.array_equal(before[outside], after[outside])
        assert report["masked_text_px"] == (x1 - x0) * (y1 - y0)


@pytest.mark.xfail(strict=True, reason="BLOCKER: median-filled rectangles mark otherwise identical textured spectra")
def test_safety_box_application_not_decodable_by_prespecified_footprint_rule():
    metrics, _ = collect_pixel_evidence()
    # Fixture-only design criterion, not a clinical or real-screen threshold.
    assert metrics["box_footprint"]["prespecified_rule_accuracy"] <= 0.6


def test_tint_matched_colored_caliper_bypasses_chroma_on_complete_fixture():
    for row in ROWS:
        clean, marked = caliper_fixture(row, tint_matched=True)
        assert overlay_mask(marked).any()  # positive chroma-detection control
        assert not overlay_mask(marked, tint=(0, 0, 60)).any()
        output, report = prepared_gray(marked, tint=(0, 0, 60))
        control, _ = prepared_gray(clean, tint=(0, 0, 60))
        expected = marked.astype(np.float32).mean(axis=-1).round().astype(np.uint8)
        assert report["masked_overlay_px"] == 0
        assert np.array_equal(output, expected)
        assert not np.array_equal(output, control)
        assert estimate_row(output) == row


@pytest.mark.xfail(strict=True, reason="BLOCKER: a colored caliper aligned with vendor tint is retained")
def test_safety_tint_matched_marker_rule_cannot_recover_programmed_rows():
    recovered = [estimate_row(prepared_gray(caliper_fixture(row, tint_matched=True)[1], tint=(0, 0, 60))[0]) == row
                 for row in ROWS]
    # Fixture-only design criterion, not a clinical or real-screen threshold.
    assert np.mean(recovered) <= 0.6


class SyntheticCache:
    """Minimal in-memory token cache; no encoder or cache files are involved."""

    meta = {"token_grid": [1, 1, 1]}
    shape = (1, 2)

    def __contains__(self, key):
        return key in {"a", "b", "c", "d"}

    def load(self, key):
        return torch.zeros(self.shape)


def conditioning_source(columns):
    target = np.array([1.5, 2.0, 2.5, 3.0])
    table = pd.DataFrame({"study_id": ["A", "B", "C", "D"], "still_id": list("abcd"),
                          "still_tag": ["AV"] * 4, "still_type": ["cw_AV"] * 4,
                          "y_AV_VMAX": target, "y_alias": target,
                          "raw_maxpg": 4 * target ** 2, "n_rows": target * 100})
    routing = load_routing("configs/radar/radar_routing_v1.yaml")
    return StillSource(table, routing, SyntheticCache(), [s.name for s in routing.trained], num_stills=1,
                       cond_columns=columns, target_columns=["y_AV_VMAX"])


@pytest.mark.parametrize("column", ["y_AV_VMAX", "y_alias"])
def test_conditioning_rejects_target_name_and_y_prefix(column):
    with pytest.raises(ValueError, match="never an input"):
        conditioning_source([column])


def test_conditioning_accepts_target_determining_alias_and_synthetic_row_proxy():
    source = conditioning_source(["raw_maxpg", "n_rows"])
    for study, target in zip("ABCD", (1.5, 2, 2.5, 3)):
        item = source.item(study, random.Random(0))
        maxpg, n_rows = item["still_cond"][0].tolist()
        assert (maxpg / 4) ** 0.5 == target
        assert n_rows / 100 == target
        assert item["still_y_mask"][0, 0]


@pytest.mark.xfail(strict=True, reason="BLOCKER: name-only guard accepts target-determining conditioning aliases")
@pytest.mark.parametrize("column", ["raw_maxpg", "n_rows"])
def test_safety_target_determining_conditioning_is_refused(column):
    with pytest.raises(ValueError):
        conditioning_source([column])


@pytest.mark.parametrize("preprocess", [None, "unmasked"])
def test_matching_encoder_accepts_missing_or_unmasked_preprocessing_provenance(preprocess):
    clip = {"embed_dim": 2, "provenance": {"encoder_sha256": "synthetic", "encoder": {"name": "fixture"}}}
    still = {"embed_dim": 2, "provenance": dict(clip["provenance"])}
    if preprocess is not None:
        still["provenance"]["preprocess"] = preprocess
    assert same_encoder(clip, still) is True


def assembled_training_input(tmp_path, preprocess):
    """Exercise training's read_stills -> same_encoder -> dataset/collate path.

    This uses actual tiny on-disk token caches, not an encoder or patient images.
    It does not run optimization. A preprocessing gate here is a proposed audit
    requirement; preprocessing validation is outside same_encoder's stated job.
    """
    from evals.radar_frozen.train import read_stills
    from src.datasets.radar_study_dataset import RadarStudyDataset, collate
    from src.datasets.radar_token_cache import TokenCache, TokenCacheWriter
    from src.models.radar.text_encoder import HashTokenizer

    provenance = {"encoder_sha256": "synthetic", "encoder": {"name": "fixture"}}
    for name, value in (("clips", 0), ("stills", 7)):
        record = dict(provenance)
        if name == "stills" and preprocess is not None:
            record["preprocess"] = preprocess
        writer = TokenCacheWriter(str(tmp_path / name), (1, 1, 1), 2, provenance=record)
        writer.add(name, torch.full((1, 2), value))
        writer.close()
    table = tmp_path / "stills.csv"
    pd.DataFrame({"study_id": ["A"], "still_id": ["stills"], "still_tag": ["AV"],
                  "still_type": ["cw_AV"], "split": ["train"]}).to_csv(table, index=False)
    routing = load_routing("configs/radar/radar_routing_v1.yaml")
    stills = read_stills({"table": str(table), "cache_dir": str(tmp_path / "stills"), "num_stills": 1},
                         routing, "train")
    cache = TokenCache(str(tmp_path / "clips"))
    assert same_encoder(cache.meta, stills.cache.meta) is True
    clips = pd.DataFrame({"study_id": ["A"], "clip_id": ["clips"], "view": ["A4C"],
                          "acquisition": ["bmode"]})
    captions = pd.DataFrame(columns=["study_id", "structure", "normality", "caption", "caption_id"])
    dataset = RadarStudyDataset(clips, captions, routing, cache, num_clips=1, stills=stills)
    return collate([dataset[0]], HashTokenizer(vocab_size=50, max_length=16))


@pytest.mark.parametrize("preprocess", [None, "unmasked"])
def test_training_input_assembly_accepts_unaudited_preprocessing(tmp_path, preprocess):
    batch = assembled_training_input(tmp_path, preprocess)
    assert batch["still_valid"].tolist() == [[True]]
    assert torch.equal(batch["still_tokens"], torch.full((1, 1, 1, 2), 7.0))


@pytest.mark.xfail(strict=True, reason="PROPOSED AUDIT GATE ABSENT: loader accepts missing/unmasked preprocessing evidence")
@pytest.mark.parametrize("preprocess", [None, "unmasked"])
def test_proposed_gate_refuses_unaudited_training_input(tmp_path, preprocess):
    with pytest.raises(ValueError):
        assembled_training_input(tmp_path, preprocess)


def test_pooler_cond_false_still_leaves_measurement_head_conditioning_active():
    """Ablation caveat: the documented pooler flag does not disable the heads."""
    from evals.radar_frozen.train import build_model
    from tests.models.radar_synthetic import synthetic_config

    config = synthetic_config({})
    config["loss"] = {"use_exact": True}
    config["data"]["stills"] = {"cond_columns": ["raw_maxpg", "n_rows"]}
    config["model"]["stills"] = {"token_grid": [1, 1, 1], "cond": False}
    config["heads"] = {"targets": ["y_AV_VMAX"], "hidden_dim": 8, "demographics": False}
    routing = load_routing("configs/radar/radar_routing_v1.yaml")
    with torch.random.fork_rng():
        torch.manual_seed(27)
        model, _ = build_model(config, routing)
        model.eval()
        assert model.pooler.still_cond is None
        assert model.heads.cond is not None
        tokens = torch.randn(1, 1, 1, config["model"]["embed_dim"])
        with torch.no_grad():
            zero = model.heads(tokens, torch.zeros(1, 1, 2))
            changed = model.heads(tokens, torch.tensor([[[16.0, 200.0]]]))
        assert not torch.allclose(zero, changed, atol=1e-5)
