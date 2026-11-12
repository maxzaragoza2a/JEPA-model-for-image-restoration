"""Patch embedding: image -> patches -> embeddings + positional encoding.

Shapes (embed_dim=256):
    CIFAR-10, 32x32 image, 4x4 patches: (B, 32, 32, 3) -> (B, 64, 48)  -> (B, 64, 256)
    STL-10,   64x64 image, 8x8 patches: (B, 64, 64, 3) -> (B, 64, 192) -> (B, 64, 256)
    STL-10,   96x96 image, 8x8 patches: (B, 96, 96, 3) -> (B, 144, 192) -> (B, 144, 384)
                                        (stl10_vits, embed_dim=384)
"""

import numpy as np
import keras
from keras import ops


def extract_patches(images, patch_size: int):
    """Cut a batch of images into a flat sequence of raw patch vectors.

        (B, H, W, C)
          -> reshape   (B, H//p, p, W//p, p, C)
          -> transpose (B, H//p, W//p, p, p, C)
          -> reshape   (B, num_patches, p*p*C)

    Args:
        images: float tensor, shape (B, H, W, C).
        patch_size: side of each square patch. Must divide H and W.

    Returns:
        Tensor of shape (B, num_patches, patch_size * patch_size * C),
        patches in row-major order.
    """

    batch_size = ops.shape(images)[0]
    h, w, c = images.shape[1], images.shape[2], images.shape[3]

    # h and w are None under a symbolic build.
    for name, size in (("height", h), ("width", w)):
        if size is not None and size % patch_size != 0:
            raise ValueError(
                f"image {name} {size} is not divisible by patch_size "
                f"{patch_size}: the patches would not tile the image"
            )

    gr, gc = h // patch_size, w // patch_size

    transpose_order = [0, 1, 3, 2, 4, 5]                              # B, gr, gc, py, px, C
    split_shape = [batch_size, gr, patch_size, gc, patch_size, c]
    final_shape = [batch_size, gr * gc, patch_size * patch_size * c]

    x = ops.reshape(images, split_shape)
    x = ops.transpose(x, transpose_order)
    return ops.reshape(x, final_shape)


def get_1d_sincos(d, pos):
    """Sine-cosine encoding of 1-D positions.

    Args:
        d: output width, must be even.
        pos: 1-D array of positions, shape (M,).

    Returns:
        np.ndarray, shape (M, d).
    """
    omega = np.arange(d // 2, dtype=np.float32)
    omega /= d / 2
    omega = 1. / (10000 ** omega)  # (d/2,)

    angles = pos[:, None] * omega  # (M, d/2)

    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """Fixed 2-D sine-cosine positional encodings.

    Half of the dimensions encode the row, half the column.

    Args:
        embed_dim: width of each vector, must be divisible by 4.
        grid_size: patches per side.

    Returns:
        np.ndarray, shape (grid_size * grid_size, embed_dim), row-major.
    """
    if embed_dim % 4 != 0:
        raise ValueError(
            f"embed_dim {embed_dim} is not divisible by 4: the budget splits "
            f"row/col and then sin/cos, so the table would come out "
            f"{4 * (embed_dim // 4)} wide instead of {embed_dim}"
        )

    coords = np.arange(grid_size, dtype=np.float32)
    rows = np.repeat(coords, grid_size)
    cols = np.tile(coords, grid_size)

    emb_row = get_1d_sincos(embed_dim // 2, rows)   # (grid_size**2, embed_dim//2)
    emb_col = get_1d_sincos(embed_dim // 2, cols)   # (grid_size**2, embed_dim//2)

    return np.concatenate([emb_row, emb_col], axis=1)


class PatchEmbedding(keras.layers.Layer):
    """(B, H, W, C) images -> (B, num_patches, embed_dim) tokens."""

    def __init__(self, patch_size: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.projection = None
        self.pos_embed = None

    def build(self, input_shape):
        """Create the projection and the fixed positional table."""
        _, h, w, c = input_shape
        self.grid_size = h // self.patch_size
        self.num_patches = self.grid_size * (w // self.patch_size)
        patch_dim = self.patch_size * self.patch_size * c

        self.projection = keras.layers.Dense(self.embed_dim, name="projection")
        self.projection.build((None, patch_dim))

        # Set through the initializer: an assign() is lost when a parent layer
        # builds this one by symbolic tracing.
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(self.num_patches, self.embed_dim),
            initializer=keras.initializers.Constant(
                get_2d_sincos_pos_embed(self.embed_dim, self.grid_size)
            ),
            trainable=False,
        )

        super().build(input_shape)

    def call(self, images):
        """(B, H, W, C) -> (B, num_patches, embed_dim)."""
        patches = extract_patches(images, self.patch_size)
        tokens = self.projection(patches)
        return tokens + self.pos_embed

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size, "embed_dim": self.embed_dim})
        return config
