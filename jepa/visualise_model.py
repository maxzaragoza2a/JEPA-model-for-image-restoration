"""Show the structure of the JEPA: every layer, its output shape and its weights.

Usage:

    python -m jepa.visualise_model                        # tree of DEFAULT_RUN
    python -m jepa.visualise_model --run stl10_d128
    python -m jepa.visualise_model --checkpoint checkpoints/cifar10_d128/jepa.weights.h5
    python -m jepa.visualise_model --netron               # also write .tflite files

Two outputs:
    1. A tree printed in the terminal: every layer down to the Dense ones,
       with the shape of its output and its number of weights.
    2. With --netron, model_graph/<run>/{encoder,predictor}.tflite. Open them
       in Netron (https://netron.app, or `pip install netron` then
       `netron model_graph/<run>/encoder.tflite`) for an interactive graph:
       click a node to see its weights. Netron also exports the graph as
       PNG or SVG.

The shapes come from one forward pass on a random batch with one mask draw,
so the number of context tokens M changes with --seed.
"""

import argparse
import os

import numpy as np
import keras
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from jepa.config import DEFAULT_RUN, PRESETS, JEPAConfig, get_config
from jepa.masking import sample_masks
from jepa.model import JEPA


def build_model(config: JEPAConfig, rng, checkpoint: str = None):
    """Build every weight of the JEPA, then load checkpoint if given.

    Returns:
        (model, images, context_indices, target_indices), the batch and the
        mask draw used for the build.
    """
    model = JEPA(config)
    model.init_target_encoder()
    images = rng.standard_normal(
        (1, config.image_size, config.image_size, config.num_channels)).astype("float32")
    context_indices, target_indices = sample_masks(config, rng)
    # The predictor is built by its first call.
    model(images, context_indices, target_indices[:1])
    if checkpoint is not None:
        model.load_weights(checkpoint)
    return model, images, context_indices, target_indices


def sublayers(layer):
    """Direct children of a layer, in creation order.

    InputLayer is skipped: a Sequential (the MLP) adds one that does nothing.
    """
    return [child for child in layer._flatten_layers(include_self=False, recursive=False)
            if not isinstance(child, keras.layers.InputLayer)]


def record_output_shapes(root, forward):
    """Run forward() and return {id(layer): output shape} for every sublayer.

    Each sublayer's call is wrapped for the duration of forward() only. A
    layer called several times keeps the shape of its last call.
    """
    shapes = {}
    patched = []

    def wrap(layer):
        original = layer.call

        def call(*args, **kwargs):
            out = original(*args, **kwargs)
            shapes[id(layer)] = tuple(out.shape) if hasattr(out, "shape") else None
            return out

        layer.call = call
        patched.append(layer)

    for layer in root._flatten_layers(include_self=False, recursive=True):
        wrap(layer)
    try:
        forward()
    finally:
        for layer in patched:
            del layer.call  # back to the class method
    return shapes


def weight_count(layer, trainable=None):
    """Number of scalars in the layer's weights (all, or only (non-)trainable)."""
    if trainable is None:
        weights = layer.weights
    elif trainable:
        weights = layer.trainable_weights
    else:
        weights = layer.non_trainable_weights
    return int(sum(np.prod(w.shape) for w in weights))


def format_shape(shape):
    if shape is None:
        return "-"
    return "(" + ", ".join("B" if i == 0 else str(d) for i, d in enumerate(shape)) + ")"


def tree_lines(layer, shapes, max_depth, prefix="", depth=0):
    """Lines of the tree below layer, one per sublayer."""
    lines = []
    children = sublayers(layer)
    for i, child in enumerate(children):
        last = i == len(children) - 1
        branch = "└─ " if last else "├─ "
        label = f"{prefix}{branch}{child.name} ({type(child).__name__})"
        lines.append((label, format_shape(shapes.get(id(child))), weight_count(child)))
        if depth + 1 < max_depth:
            lines += tree_lines(child, shapes, max_depth,
                                prefix + ("   " if last else "│  "), depth + 1)
    return lines


def print_tree(model, shapes, max_depth):
    lines = [(f"{model.name} ({type(model).__name__})", "", weight_count(model))]
    lines += tree_lines(model, shapes, max_depth)
    width = max(len(label) for label, _, _ in lines) + 2
    shape_width = max(len("output shape"), *(len(s) for _, s, _ in lines)) + 2
    header = f"{'layer':<{width}}{'output shape':<{shape_width}}{'weights':>12}"
    print(header)
    print("─" * len(header))
    for label, shape, count in lines:
        print(f"{label:<{width}}{shape:<{shape_width}}{count:>12,}")
    print("─" * len(header))
    for part in sublayers(model):
        print(f"{part.name:<20} trainable {weight_count(part, True):>10,}"
              f"   frozen {weight_count(part, False):>10,}")
    print("B is the batch size. frozen = sine-cosine positional tables, and the"
          " whole target encoder (updated by EMA, not by the optimizer).")


def export_tflite(model, config, context_indices, target_indices, out_dir):
    """Write encoder.tflite and predictor.tflite, graphs that Netron can open.

    The context encoder sees the full image. The predictor is traced for the
    mask draw of the build (M context tokens, one target block of T tokens).
    """
    os.makedirs(out_dir, exist_ok=True)
    image_spec = tf.TensorSpec(
        (1, config.image_size, config.image_size, config.num_channels), tf.float32, name="images")
    context_spec = tf.TensorSpec(
        (1, len(context_indices), config.embed_dim), tf.float32, name="context_tokens")

    encoder = model.context_encoder
    predictor = model.predictor
    targets = target_indices[0]

    @tf.function(input_signature=[image_spec])
    def encode(images):
        return encoder(images, training=False)

    @tf.function(input_signature=[context_spec])
    def predict(context_tokens):
        return predictor(context_tokens, context_indices, targets, training=False)

    paths = []
    for name, fn in (("encoder", encode), ("predictor", predict)):
        # Weights turned into constants, so that Netron shows their values.
        frozen = convert_variables_to_constants_v2(fn.get_concrete_function())
        converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
        path = os.path.join(out_dir, f"{name}.tflite")
        with open(path, "wb") as f:
            f.write(converter.convert())
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default=DEFAULT_RUN, choices=sorted(PRESETS),
                        help=f"preset of jepa/config.py (default: {DEFAULT_RUN})")
    parser.add_argument("--checkpoint", default=None,
                        help="weights to load, e.g. checkpoints/<run>/jepa.weights.h5"
                             " (default: random weights, same structure)")
    parser.add_argument("--depth", type=int, default=4,
                        help="levels of the tree to print (default: 4, down to the Dense layers)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the mask draw")
    parser.add_argument("--netron", action="store_true",
                        help="also write .tflite files to open in Netron")
    parser.add_argument("--out", default=None,
                        help="folder of the .tflite files (default: model_graph/<run>)")
    args = parser.parse_args()

    config = get_config(args.run)
    rng = np.random.default_rng(args.seed)
    model, images, context_indices, target_indices = build_model(config, rng, args.checkpoint)

    shapes = record_output_shapes(
        model, lambda: model(images, context_indices, target_indices[:1]))
    print(f"run {args.run}: {len(context_indices)} context tokens, "
          f"{len(target_indices[0])} tokens in the target block\n")
    print_tree(model, shapes, args.depth)

    if args.netron:
        out_dir = args.out or os.path.join("model_graph", args.run)
        paths = export_tflite(model, config, context_indices, target_indices, out_dir)
        print("\nOpen in Netron (https://netron.app):")
        for path in paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
