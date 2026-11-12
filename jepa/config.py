"""Hyperparameters for the project."""

from dataclasses import dataclass


@dataclass
class JEPAConfig:
    # ---- Data -------------------------------------------------------------
    # The defaults are CIFAR-10. DATASETS below holds the fields to change
    # for another dataset.
    dataset: str = "cifar10"
    image_size: int = 32          # CIFAR-10 is 32x32
    num_channels: int = 3
    patch_size: int = 4           # 8x8 grid, 64 patches per image

    # Per-channel CIFAR-10 statistics.
    mean: tuple = (0.4914, 0.4822, 0.4465)
    std: tuple = (0.2470, 0.2435, 0.2616)

    # ---- Encoder (shared by context and target) ---------------------------
    embed_dim: int = 256
    depth: int = 4
    num_heads: int = 8            # 32 dimensions per head
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # ---- Predictor --------------------------------------------------------
    # Narrow and shallow, as in I-JEPA.
    predictor_embed_dim: int = 128
    predictor_depth: int = 2
    predictor_num_heads: int = 8  # 16 dimensions per head

    # ---- Masking ----------------------------------------------------------
    num_target_blocks: int = 4
    target_scale: tuple = (0.15, 0.20)        # fraction of the grid per target
    target_aspect_ratio: tuple = (0.75, 1.5)  # height / width
    context_scale: tuple = (0.85, 1.00)

    # ---- Training ---------------------------------------------------------
    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 3

    # EMA momentum of the target encoder, annealed to ema_end.
    ema_start: float = 0.996
    ema_end: float = 1.0

    seed: int = 42

    # ---- Derived shapes ---------------------------------------------------
    @property
    def grid_size(self) -> int:
        """Number of patches along one side."""
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size {self.image_size} is not divisible by "
                f"patch_size {self.patch_size}"
            )
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        """Total tokens per image."""
        return self.grid_size ** 2

    @property
    def patch_dim(self) -> int:
        """Flattened size of one raw patch."""
        return self.patch_size * self.patch_size * self.num_channels


# Data fields of each dataset.
DATASETS = {
    "cifar10": dict(dataset="cifar10", image_size=32, patch_size=4,
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    # STL-10 is 96x96, resized to 64x64. Patches of 8 keep the 8x8 grid of
    # CIFAR-10, so the masking and the attention cost do not change.
    # Statistics of the unlabelled split after the resize.
    "stl10": dict(dataset="stl10", image_size=64, patch_size=8,
                  mean=(0.4406, 0.4273, 0.3858), std=(0.2592, 0.2519, 0.2597)),
    # STL-10 at its native 96x96, no resize. Patches of 8 give a 12x12 grid
    # of 144 tokens: finer detail per image, attention ~5x more expensive.
    # The std is higher than at 64x64, where the antialiased resize smooths
    # the images.
    "stl10_96": dict(dataset="stl10", image_size=96, patch_size=8,
                     mean=(0.4406, 0.4273, 0.3858), std=(0.2687, 0.2613, 0.2685)),
}

# One entry per training run: the fields that differ from the defaults above.
# The run name also names its checkpoint folder (checkpoints/<run>/).
_D128 = dict(embed_dim=128, num_heads=4, predictor_embed_dim=64,
             predictor_num_heads=4)
# ViT-Small: 12 blocks of width 384, 6 heads of 64 dims (~21M weights).
# The predictor stays narrow, as in I-JEPA: width 192, 6 blocks, 6 heads of
# 32 dims (~3M). Meant for a GPU: batch 256, 100 epochs. The learning rate is
# lower than the default 1e-3 (I-JEPA uses 1e-3 with batches of 2048).
_VITS = dict(embed_dim=384, depth=12, num_heads=6,
             predictor_embed_dim=192, predictor_depth=6, predictor_num_heads=6,
             batch_size=256, epochs=100, learning_rate=5e-4, warmup_epochs=10)
PRESETS = {
    # First run, see result_analysis/RESULTATS.md.
    "cifar10_d128": dict(**DATASETS["cifar10"], **_D128),
    "cifar10_d256": dict(**DATASETS["cifar10"]),
    "stl10_d128": dict(**DATASETS["stl10"], **_D128),
    "stl10_d256": dict(**DATASETS["stl10"]),
    "stl10_vits": dict(**DATASETS["stl10_96"], **_VITS),
}
DEFAULT_RUN = "cifar10_d256"


def get_config(run: str = DEFAULT_RUN) -> JEPAConfig:
    """Config of a named run, from PRESETS."""
    if run not in PRESETS:
        raise ValueError(f"unknown run {run!r}, expected one of {sorted(PRESETS)}")
    return JEPAConfig(**PRESETS[run])
