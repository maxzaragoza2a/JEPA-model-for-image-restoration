"""Tests for the compiled training step and the resume state.

A tiny model keeps them fast. They do not read any dataset.
"""

import os

import numpy as np
import pytest
import tensorflow as tf

from jepa.config import JEPAConfig
from jepa.masking import sample_masks
from jepa.model import JEPA, jepa_loss
from jepa.train import (ResumeState, as_tensors, build_optimizer, make_ema_step,
                        make_train_step)


@pytest.fixture
def cfg():
    return JEPAConfig(image_size=16, patch_size=4, embed_dim=32, depth=2, num_heads=2,
                      predictor_embed_dim=16, predictor_depth=1, predictor_num_heads=2)


def images(cfg, batch=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(
        (batch, cfg.image_size, cfg.image_size, cfg.num_channels)).astype("float32")


def build(cfg, seed=0):
    """Model, optimizer, compiled step and EMA, as train() builds them."""
    model = JEPA(cfg)
    model.init_target_encoder()
    model(images(cfg), *sample_masks(cfg, np.random.default_rng(seed)))
    optimizer = build_optimizer(cfg, total_steps=100, warmup_steps=10)
    return model, optimizer, make_train_step(model, optimizer, cfg), make_ema_step(model)


def as_arrays(weights):
    return [np.array(w) for w in weights]


# ---- the compiled step -----------------------------------------------------

def test_one_trace_serves_every_mask_size(cfg):
    _, _, train_step, _ = build(cfg)
    rng = np.random.default_rng(1)
    sizes = set()
    for _ in range(8):
        context, targets = sample_masks(cfg, rng)
        sizes.add((len(context),) + tuple(len(t) for t in targets))
        train_step(images(cfg), *as_tensors(context, targets))

    assert len(sizes) > 1, "the test needs draws of different sizes"
    assert train_step.experimental_get_tracing_count() == 1


def test_the_batch_size_can_change_without_a_new_trace(cfg):
    _, _, train_step, _ = build(cfg)
    rng = np.random.default_rng(1)
    for batch in (4, 2, 3):
        train_step(images(cfg, batch), *as_tensors(*sample_masks(cfg, rng)))
    assert train_step.experimental_get_tracing_count() == 1


def test_the_compiled_step_returns_the_loss_before_the_update(cfg):
    model, _, train_step, _ = build(cfg)
    context, targets = sample_masks(cfg, np.random.default_rng(1))
    x = images(cfg)

    expected = float(jepa_loss(*model(x, context, targets, training=True)))
    loss = float(train_step(x, *as_tensors(context, targets)))

    assert loss == pytest.approx(expected, rel=1e-5)


def test_the_step_updates_the_context_encoder_and_the_predictor_only(cfg):
    model, optimizer, train_step, _ = build(cfg)
    before = {name: as_arrays(getattr(model, name).weights)
              for name in ("context_encoder", "target_encoder", "predictor")}

    # The warmup starts at a learning rate of 0: the first step moves nothing.
    rng = np.random.default_rng(1)
    for _ in range(2):
        train_step(images(cfg), *as_tensors(*sample_masks(cfg, rng)))

    def changed(name):
        return any(not np.array_equal(a, b)
                   for a, b in zip(before[name], as_arrays(getattr(model, name).weights)))

    assert changed("context_encoder") and changed("predictor")
    assert not changed("target_encoder")
    assert int(optimizer.iterations) == 2


def test_the_compiled_ema_matches_the_eager_one(cfg):
    model, _, train_step, ema_step = build(cfg)
    train_step(images(cfg), *as_tensors(*sample_masks(cfg, np.random.default_rng(1))))
    target = as_arrays(model.target_encoder.weights)
    context = as_arrays(model.context_encoder.weights)

    ema_step(tf.constant(0.9, tf.float32))

    for t, c, new in zip(target, context, as_arrays(model.target_encoder.weights)):
        np.testing.assert_allclose(new, 0.9 * t + 0.1 * c, rtol=1e-6, atol=1e-7)


# ---- resume ----------------------------------------------------------------

def run_steps(cfg, train_step, ema_step, rng, n):
    for _ in range(n):
        train_step(images(cfg), *as_tensors(*sample_masks(cfg, rng)))
        ema_step(tf.constant(0.99, tf.float32))


def test_nothing_to_resume_in_an_empty_folder(cfg, tmp_path):
    model, optimizer, _, _ = build(cfg)
    assert ResumeState(str(tmp_path), model, optimizer).load(np.random.default_rng(0)) is None


def test_a_resumed_run_continues_exactly(cfg, tmp_path):
    model, optimizer, train_step, ema_step = build(cfg)
    rng = np.random.default_rng(1)
    run_steps(cfg, train_step, ema_step, rng, 3)
    ResumeState(str(tmp_path), model, optimizer).save(
        epoch=1, step=3, total_steps=100, rng=rng, history=[{"epoch": 1, "loss": 0.5}])

    # The step the original run takes next.
    context, targets = sample_masks(cfg, rng)
    expected = float(train_step(images(cfg), *as_tensors(context, targets)))

    # Another model, another optimizer, another rng: all overwritten.
    model2, optimizer2, train_step2, _ = build(cfg, seed=7)
    rng2 = np.random.default_rng(99)
    state = ResumeState(str(tmp_path), model2, optimizer2).load(rng2)

    assert (state["epoch"], state["step"], state["total_steps"]) == (1, 3, 100)
    assert state["history"] == [{"epoch": 1, "loss": 0.5}]
    assert int(optimizer2.iterations) == 3
    context2, targets2 = sample_masks(cfg, rng2)
    assert np.array_equal(context, context2)
    loss = float(train_step2(images(cfg), *as_tensors(context2, targets2)))
    assert loss == pytest.approx(expected, rel=1e-6)


def test_only_the_last_epoch_is_kept(cfg, tmp_path):
    model, optimizer, train_step, ema_step = build(cfg)
    state = ResumeState(str(tmp_path), model, optimizer)
    rng = np.random.default_rng(1)
    for epoch in (1, 2, 3):
        run_steps(cfg, train_step, ema_step, rng, 1)
        state.save(epoch, epoch, 100, rng, [])

    assert sorted(os.listdir(tmp_path / "resume")) == [
        "model-3.weights.h5", "optimizer-3.npz", "state.json"]
