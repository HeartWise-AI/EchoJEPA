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
    estimate_row, has_flat_footprint, label_value_fixture, prepared_gray,
)


@pytest.mark.parametrize("intensity", [255, 200], ids=["white", "grey"])
def test_achromatic_caliper_positions_survive_current_crop(intensity):
    for row in ROWS:
        _, marked = caliper_fixture(row, intensity)
        output, report = prepared_gray(marked)
        assert not overlay_mask(marked).any()
        assert report["masked_overlay_px"] == 0
        assert report["window"][1] - report["window"][0] == 56
        assert estimate_row(output) == row


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="BLOCKER: grey/white measurement geometry reaches encoder input")
@pytest.mark.parametrize("intensity", [255, 200], ids=["white", "grey"])
def test_safety_achromatic_marker_rule_cannot_recover_programmed_rows(intensity):
    recovered = [estimate_row(prepared_gray(caliper_fixture(row, intensity)[1])[0]) == row for row in ROWS]
    assert np.mean(recovered) <= 0.6  # fixture-only criterion, not a clinical threshold


@pytest.mark.parametrize("boxes", [[LABEL_BOX], [LABEL_BOX, VALUE_BOX]])
def test_panel_fixture_without_free_window_is_rejected(boxes):
    _, marked = label_value_fixture()
    with pytest.raises(ValueError, match="No window"):
        prepared_gray(marked, text_boxes=boxes)


def test_current_crop_excludes_panel_instead_of_median_filling():
    metrics, _ = collect_pixel_evidence()
    assert metrics["label_only"]["rejected_no_free_window"]
    assert metrics["label_and_value"]["rejected_no_free_window"]
    first, last = metrics["textured_panel"]["window"]
    assert last - first == 56
    assert last <= 100 - 12
    assert not metrics["textured_panel"]["flat_footprint"]


def test_tint_matched_caliper_positions_survive_current_crop():
    for row in ROWS:
        _, marked = caliper_fixture(row, tint_matched=True)
        assert overlay_mask(marked).any()
        assert not overlay_mask(marked, tint=(0, 0, 60)).any()
        output, report = prepared_gray(marked, tint=(0, 0, 60))
        assert report["masked_overlay_px"] == 0
        assert estimate_row(output) == row


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="BLOCKER: a colored caliper aligned with vendor tint is retained")
def test_safety_tint_matched_marker_rule_cannot_recover_programmed_rows():
    recovered = [estimate_row(prepared_gray(caliper_fixture(row, tint_matched=True)[1], tint=(0, 0, 60))[0]) == row
                 for row in ROWS]
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


@pytest.mark.parametrize("column", ["raw_maxpg", "n_rows"])
def test_target_determining_conditioning_aliases_are_refused(column):
    with pytest.raises(ValueError, match="not among the inputs known before a measurement"):
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
def test_training_input_assembly_rejects_unaudited_preprocessing(tmp_path, preprocess):
    with pytest.raises(ValueError, match="preprocessing recipe"):
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


@pytest.mark.parametrize("preprocess", [None, "unmasked", "accepted"])
def test_training_entrypoint_checks_cache_recipe_before_model_construction(tmp_path, preprocess):
    from unittest import mock

    from evals.radar_frozen import train as radar_train
    from src.datasets.radar_token_cache import TokenCacheWriter
    from src.datasets.still_preprocess import PREPROCESS_VERSION
    from tests.models.radar_synthetic import DIM, make_synthetic, synthetic_config

    paths = make_synthetic(str(tmp_path / "data"))
    provenance = {"source": "invented audit fixture"}  # deliberately not exempt synthetic/empty provenance
    if preprocess is not None:
        provenance["preprocess"] = PREPROCESS_VERSION if preprocess == "accepted" else preprocess
    cache_dir = str(tmp_path / "stills-cache")
    writer = TokenCacheWriter(cache_dir, (1, 1, 1), DIM, provenance=provenance)
    writer.add("still", torch.zeros(1, DIM))
    writer.close()
    table = tmp_path / "stills.csv"
    pd.DataFrame({"study_id": ["S00"], "still_id": ["still"], "still_tag": ["AV"],
                  "still_type": ["cw_AV"], "split": ["train"]}).to_csv(table, index=False)
    config = synthetic_config(paths)
    config["data"]["stills"] = {"table": str(table), "cache_dir": cache_dir, "num_stills": 1}
    config["model"]["stills"] = {"token_grid": [1, 1, 1], "cond": False}
    # Stop at model construction: the actual training entry point and data checks run,
    # but no model or optimizer is created. The accepted-recipe control proves reachability.
    with mock.patch.object(radar_train, "build_model", side_effect=RuntimeError("model construction reached")) as build:
        if preprocess == "accepted":
            with pytest.raises(RuntimeError, match="model construction reached"):
                radar_train.train(config, str(tmp_path / "run"), resume=False)
            build.assert_called_once()
        else:
            with pytest.raises(ValueError, match="preprocessing recipe"):
                radar_train.train(config, str(tmp_path / "run"), resume=False)
            build.assert_not_called()
