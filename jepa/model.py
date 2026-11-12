"""JEPA: context encoder, target encoder and predictor assembled.

Shapes for one batch (M context tokens, T tokens in one target block, N
tokens per image), for embed_dim = 256:
    context encoder: (B, H, W, 3), context_indices -> (B, M, 256)
    target encoder:  (B, H, W, 3) -> (B, N, 256) -> gather -> (B, T, 256)
    predictor:       (B, M, 256), context_indices, target_indices -> (B, T, 256)
N = 64 (8x8 grid) on CIFAR-10 at 32x32 and STL-10 at 64x64, 144 (12x12 grid)
on STL-10 at 96x96 (stl10_vits, where embed_dim = 384).

Only the context encoder and the predictor receive gradients. The target
encoder is an exponential moving average of the context encoder.
"""

import keras
from keras import ops

from jepa.config import JEPAConfig
from jepa.masking import gather_tokens
from jepa.predictor import Predictor
from jepa.transformer import VisionTransformer


def jepa_loss(predictions, targets):
    """Mean squared error between predictions and targets, averaged over blocks.

    Args:
        predictions: list of (B, T_k, embed_dim) tensors, one per target block.
        targets: list of (B, T_k, embed_dim) tensors, same order.

    Returns:
        Scalar tensor.
    """
    loss = 0.0
    for pred, targ in zip(predictions, targets):
        loss += ops.mean(ops.square(pred - targ))
    loss /= len(predictions)
    return loss


class JEPA(keras.Model):
    """Images and one mask draw -> predictions and targets, block by block."""

    def __init__(self, config: JEPAConfig, **kwargs):
        """Create context_encoder, target_encoder and predictor from config."""
        super().__init__(**kwargs)
        self.config = config
        self.context_encoder = VisionTransformer(
            config.patch_size, config.embed_dim, config.depth, config.num_heads, config.mlp_ratio, config.dropout, name="context_encoder")
        self.target_encoder = VisionTransformer(
            config.patch_size, config.embed_dim, config.depth, config.num_heads, config.mlp_ratio, config.dropout, name="target_encoder")
        self.predictor = Predictor(
            config.grid_size, config.embed_dim, config.predictor_embed_dim, config.predictor_depth, config.predictor_num_heads, config.mlp_ratio, config.dropout, name="predictor")

    def init_target_encoder(self):
        """Build both encoders and copy the context weights into the target.

        To call once, before training.
        """
        cfg = self.config
        dummy = ops.zeros((1, cfg.image_size, cfg.image_size, cfg.num_channels))
        self.context_encoder(dummy)
        self.target_encoder(dummy)
        # Updated by EMA only, never by the optimizer. Frozen after the build,
        # since PatchEmbedding creates its projection inside build().
        self.target_encoder.trainable = False
        for target, context in zip(self.target_encoder.weights, self.context_encoder.weights):
            target.assign(context)

    def update_target_encoder(self, momentum: float):
        """EMA step: target <- momentum * target + (1 - momentum) * context.

        To call after each optimizer step.
        """
        for target, context in zip(self.target_encoder.weights, self.context_encoder.weights):
            target.assign(momentum * target + (1.0 - momentum) * context)

    def encode_targets(self, images, target_indices):
        """Target embeddings of each block, from the full image.

        Args:
            images: (B, H, W, C), standardised.
            target_indices: list of 1-D int arrays, one per target block.

        Returns:
            List of (B, T_k, embed_dim) tensors, with no gradient.
        """
        tokens = self.target_encoder(images, training=False)
        return [ops.stop_gradient(gather_tokens(tokens, indices))
                for indices in target_indices]

    def predict_targets(self, images, context_indices, target_indices, training=None):
        """Predicted embeddings of each block, from the context only.

        Args:
            images: (B, H, W, C), standardised.
            context_indices: 1-D int array, the M visible grid positions.
            target_indices: list of 1-D int arrays, one per target block.

        Returns:
            List of (B, T_k, embed_dim) tensors.
        """
        context = self.context_encoder(images, indices=context_indices, training=training)
        return [self.predictor(context, context_indices, indices, training=training)
                for indices in target_indices]

    def call(self, images, context_indices, target_indices, training=None):
        """Run both branches.

        Returns:
            (predictions, targets), two lists of (B, T_k, embed_dim) tensors.
        """
        predictions = self.predict_targets(images, context_indices, target_indices, training=training)
        targets = self.encode_targets(images, target_indices)
        return predictions, targets
