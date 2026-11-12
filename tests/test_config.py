"""Tests for the run presets of the config."""

import numpy as np
import pytest

from jepa.config import DATASETS, DEFAULT_RUN, PRESETS, JEPAConfig, get_config
from jepa.masking import sample_masks
from jepa.model import JEPA


@pytest.mark.parametrize("run", sorted(PRESETS))
def test_every_preset_builds_a_valid_model_width(run):
    cfg = get_config(run)
    assert cfg.embed_dim % cfg.num_heads == 0
    assert cfg.predictor_embed_dim % cfg.predictor_num_heads == 0
    # The sin-cos position code splits the width in four.
    assert cfg.embed_dim % 4 == 0 and cfg.predictor_embed_dim % 4 == 0
    # As in I-JEPA, the predictor is narrower than the encoder.
    assert cfg.predictor_embed_dim < cfg.embed_dim


def test_the_baseline_preset_is_the_first_run():
    cfg = get_config("cifar10_d128")
    assert (cfg.embed_dim, cfg.num_heads,
            cfg.predictor_embed_dim, cfg.predictor_num_heads) == (128, 4, 64, 4)


def test_the_default_run_is_the_wider_model():
    cfg = get_config()
    assert DEFAULT_RUN == "cifar10_d256"
    assert (cfg.embed_dim, cfg.num_heads,
            cfg.predictor_embed_dim, cfg.predictor_num_heads) == (256, 8, 128, 8)
    # Same width per head as the baseline, in the encoder and in the
    # predictor: only the number of heads changes.
    base = get_config("cifar10_d128")
    assert cfg.embed_dim // cfg.num_heads == base.embed_dim // base.num_heads
    assert (cfg.predictor_embed_dim // cfg.predictor_num_heads
            == base.predictor_embed_dim // base.predictor_num_heads)


def test_an_unknown_run_is_rejected():
    with pytest.raises(ValueError, match="unknown run"):
        get_config("cifar10_d512")


@pytest.mark.parametrize("run", sorted(PRESETS))
def test_every_preset_runs_through_the_model(run):
    cfg = get_config(run)
    rng = np.random.default_rng(0)
    images = rng.standard_normal(
        (2, cfg.image_size, cfg.image_size, cfg.num_channels)).astype("float32")
    context_indices, target_indices = sample_masks(cfg, rng)

    model = JEPA(cfg)
    model.init_target_encoder()
    predictions, targets = model(images, context_indices, target_indices)

    assert np.asarray(model.target_encoder(images)).shape == (
        2, cfg.num_patches, cfg.embed_dim)
    for p, t, indices in zip(predictions, targets, target_indices):
        assert p.shape == t.shape == (2, len(indices), cfg.embed_dim)


@pytest.mark.parametrize("dataset", ["cifar10", "stl10"])
def test_both_datasets_keep_the_8x8_grid(dataset):
    cfg = JEPAConfig(**DATASETS[dataset])
    assert (cfg.grid_size, cfg.num_patches) == (8, 64)


def test_stl10_at_96_keeps_the_native_size_on_a_12x12_grid():
    cfg = JEPAConfig(**DATASETS["stl10_96"])
    assert cfg.image_size == 96
    assert (cfg.grid_size, cfg.num_patches) == (12, 144)


def test_the_vit_small_preset():
    cfg = get_config("stl10_vits")
    assert (cfg.dataset, cfg.image_size, cfg.grid_size) == ("stl10", 96, 12)
    assert (cfg.embed_dim, cfg.depth, cfg.num_heads) == (384, 12, 6)
    assert cfg.embed_dim // cfg.num_heads == 64
    assert (cfg.predictor_embed_dim, cfg.predictor_depth, cfg.predictor_num_heads) == (192, 6, 6)
    assert cfg.predictor_embed_dim // cfg.predictor_num_heads == 32


def test_the_vit_small_has_about_24m_trainable_weights():
    cfg = get_config("stl10_vits")
    rng = np.random.default_rng(0)
    model = JEPA(cfg)
    model.init_target_encoder()
    model(np.zeros((1, 96, 96, 3), "float32"), *sample_masks(cfg, rng))

    encoder = sum(int(np.prod(w.shape)) for w in model.context_encoder.trainable_weights)
    predictor = sum(int(np.prod(w.shape)) for w in model.predictor.trainable_weights)
    assert 20e6 < encoder < 23e6
    assert 2e6 < predictor < 4e6
