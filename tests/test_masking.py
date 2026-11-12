"""Tests for context / target mask sampling."""

import numpy as np
import pytest

from jepa.config import DATASETS, JEPAConfig
from jepa.masking import (
    block_to_indices,
    gather_tokens,
    sample_block,
    sample_masks,
)
from jepa.patch_embed import PatchEmbedding, extract_patches


# The 8x8 grid of CIFAR-10 and the 12x12 grid of STL-10 at 96x96.
@pytest.fixture(params=["cifar10", "stl10_96"])
def cfg(request):
    return JEPAConfig(**DATASETS[request.param])


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ---- block_to_indices ------------------------------------------------------

def test_block_to_indices_is_row_major():
    """2x2 block at (1, 2) of a 4x4 grid."""
    assert list(block_to_indices(1, 2, 2, 2, 4)) == [6, 7, 10, 11]


@pytest.mark.parametrize("top,left,height,width,grid", [
    (0, 0, 1, 1, 4),
    (0, 0, 4, 4, 4),
    (1, 2, 2, 2, 4),
    (3, 5, 4, 2, 8),
    (0, 7, 8, 1, 8),
])
def test_block_to_indices_covers_the_rectangle_exactly(top, left, height, width, grid):
    got = block_to_indices(top, left, height, width, grid)

    expected = [r * grid + c
                for r in range(top, top + height)
                for c in range(left, left + width)]

    assert got.shape == (height * width,)
    assert list(got) == expected
    assert list(got) == sorted(got)


# ---- sample_block ----------------------------------------------------------

@pytest.mark.parametrize("grid", [4, 8, 14])
def test_sample_block_always_fits_in_the_grid(rng, cfg, grid):
    for _ in range(2000):
        top, left, height, width = sample_block(
            grid, cfg.target_scale, cfg.target_aspect_ratio, rng
        )
        assert 1 <= height <= grid and 1 <= width <= grid
        assert 0 <= top and top + height <= grid
        assert 0 <= left and left + width <= grid


def test_sample_block_area_tracks_the_requested_scale(rng, cfg):
    """Loose bounds: rounding to whole patches moves the realised area."""
    low, high = cfg.target_scale
    fractions = np.array([
        np.prod(sample_block(cfg.grid_size, cfg.target_scale,
                             cfg.target_aspect_ratio, rng)[2:]) / cfg.num_patches
        for _ in range(2000)
    ])

    assert fractions.min() >= low * 0.75
    assert fractions.max() <= high * 1.35
    assert low <= fractions.mean() <= high


# ---- sample_masks ----------------------------------------------------------

@pytest.mark.parametrize("seed", range(20))
def test_no_target_index_survives_in_the_context(cfg, seed):
    context, targets = sample_masks(cfg, np.random.default_rng(seed))

    leaked = np.intersect1d(context, np.concatenate(targets))
    assert leaked.size == 0, f"targets {leaked} survived in the context"


@pytest.mark.parametrize("seed", range(20))
def test_masks_stay_inside_the_grid_and_targets_stay_separate(cfg, seed):
    context, targets = sample_masks(cfg, np.random.default_rng(seed))

    assert len(targets) == cfg.num_target_blocks
    for block in targets:
        assert block.size > 0
        assert block.min() >= 0 and block.max() < cfg.num_patches

    assert list(context) == sorted(set(context))
    if context.size:
        assert context.min() >= 0 and context.max() < cfg.num_patches


def test_masks_are_reproducible_from_the_seed(cfg):
    first_ctx, first_tgts = sample_masks(cfg, np.random.default_rng(cfg.seed))
    second_ctx, second_tgts = sample_masks(cfg, np.random.default_rng(cfg.seed))

    assert np.array_equal(first_ctx, second_ctx)
    for a, b in zip(first_tgts, second_tgts):
        assert np.array_equal(a, b)


# ---- gather_tokens ---------------------------------------------------------

def test_gather_tokens_selects_without_transforming(cfg, rng):
    tokens = rng.standard_normal((3, cfg.num_patches, cfg.embed_dim)).astype("float32")
    indices = np.array([0, 5, 17, 63])

    gathered = np.asarray(gather_tokens(tokens, indices))

    assert gathered.shape == (3, indices.size, cfg.embed_dim)
    assert np.array_equal(gathered, tokens[:, indices, :])


def test_gather_tokens_agrees_with_the_patch_ordering(cfg, rng):
    """Gathered patches match the raw pixel crop at the same grid cells."""
    p, g = cfg.patch_size, cfg.grid_size
    top, left, height, width = 1, 2, 2, 2
    images = rng.standard_normal(
        (1, cfg.image_size, cfg.image_size, cfg.num_channels)
    ).astype("float32")

    patches = np.asarray(extract_patches(images, p))
    indices = block_to_indices(top, left, height, width, g)

    gathered = np.asarray(gather_tokens(patches, indices))[0]

    for k in range(height * width):
        r, c = top + k // width, left + k % width
        crop = images[0, r * p:(r + 1) * p, c * p:(c + 1) * p, :].reshape(-1)
        assert np.array_equal(gathered[k], crop), (
            f"slot {k} of the block should be the patch at grid cell ({r}, {c}); "
            f"masking.py and extract_patches disagree on the grid order"
        )


def test_the_whole_pipeline_produces_the_expected_shapes(cfg):
    batch = 2
    images = np.zeros(
        (batch, cfg.image_size, cfg.image_size, cfg.num_channels), "float32"
    )

    tokens = PatchEmbedding(cfg.patch_size, cfg.embed_dim)(images)
    assert tuple(tokens.shape) == (batch, cfg.num_patches, cfg.embed_dim)

    context_idx, target_idx = sample_masks(cfg, np.random.default_rng(cfg.seed))

    context = gather_tokens(tokens, context_idx)
    assert tuple(context.shape) == (batch, context_idx.size, cfg.embed_dim)

    for block in target_idx:
        target = gather_tokens(tokens, block)
        assert tuple(target.shape) == (batch, block.size, cfg.embed_dim)
