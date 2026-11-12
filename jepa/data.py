"""Input pipeline, for CIFAR-10 and STL-10 (config.dataset).

build_pretrain_dataset yields images only (self-supervised pretraining).
build_probe_dataset yields (image, label) pairs for linear-probe evaluation.

CIFAR-10 is 32x32 and loaded in memory by Keras. STL-10 is 96x96, streamed
by tensorflow_datasets and resized to config.image_size when it differs
(64 for the stl10 presets, kept at 96 for stl10_vits). Both reach augment
and standardise as uint8 images of config.image_size.

tensorflow_datasets reads STL-10 from $TFDS_DATA_DIR when it is set
(default ~/tensorflow_datasets), and downloads it there if missing.
"""

# tensorflow_datasets 4.9.10 calls urllib.parse after a bare `import urllib`
# when it downloads a dataset: import it first.
import urllib.parse  # noqa: F401

import numpy as np
import tensorflow as tf
import keras

from jepa.config import JEPAConfig

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]
CLASSES = {"cifar10": CIFAR10_CLASSES, "stl10": STL10_CLASSES}

# split of build_pretrain_dataset / build_probe_dataset -> tfds split.
# STL-10 pretrains on its 100k unlabelled images, made for this.
STL10_PRETRAIN_SPLITS = {"train": "unlabelled", "test": "test"}
STL10_PROBE_SPLITS = {"train": "train", "test": "test"}
STL10_SIZE = 96

AUTOTUNE = tf.data.AUTOTUNE


def load_cifar10():
    """Return raw uint8 CIFAR-10 arrays, labels flattened to (N,)."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    return (x_train, y_train.squeeze(-1)), (x_test, y_test.squeeze(-1))


def resize(image, config: JEPAConfig):
    """uint8 (H, W, C) -> uint8 (image_size, image_size, C), antialiased."""
    image = tf.image.resize(image, (config.image_size, config.image_size),
                            method="bilinear", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)


def load_stl10(config: JEPAConfig, split: str):
    """(image, label) dataset of one tfds split, in file order, resized
    to config.image_size if needed.

    The unlabelled split has label -1.
    """
    import tensorflow_datasets as tfds

    ds = tfds.load("stl10", split=split, as_supervised=True, shuffle_files=False)
    if config.image_size == STL10_SIZE:
        return ds
    return ds.map(lambda img, lab: (resize(img, config), lab),
                  num_parallel_calls=AUTOTUNE, deterministic=True)


def load_split(config: JEPAConfig, split: str, pretrain: bool):
    """(image, label) dataset of uint8 images of config.image_size.

    Args:
        split: "train" or "test".
        pretrain: for STL-10, "train" is then the unlabelled split.
    """
    if config.dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = load_cifar10()
        images, labels = (x_train, y_train) if split == "train" else (x_test, y_test)
        return tf.data.Dataset.from_tensor_slices((images, labels))
    if config.dataset == "stl10":
        splits = STL10_PRETRAIN_SPLITS if pretrain else STL10_PROBE_SPLITS
        return load_stl10(config, splits[split])
    raise ValueError(f"unknown dataset {config.dataset!r}, expected one of {sorted(CLASSES)}")


def standardise(images, config: JEPAConfig):
    """uint8 [0, 255] -> float32, zero mean and unit variance per channel."""
    images = tf.cast(images, tf.float32) / 255.0
    mean = tf.constant(config.mean, dtype=tf.float32)
    std = tf.constant(config.std, dtype=tf.float32)
    return (images - mean) / std


def augment(image, config: JEPAConfig):
    """Random crop with reflect padding and random horizontal flip.

    The padding is an eighth of the side: 4px at 32x32, 8px at 64x64,
    12px at 96x96.
    """
    pad = config.image_size // 8
    image = tf.pad(image, [[pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
    image = tf.image.random_crop(
        image, size=[config.image_size, config.image_size, config.num_channels]
    )
    image = tf.image.random_flip_left_right(image)
    return image


def build_pretrain_dataset(config: JEPAConfig, split: str = "train"):
    """Augmented, standardised images without labels."""
    ds = load_split(config, split, pretrain=True).map(lambda img, _: img)
    ds = ds.shuffle(10_000, seed=config.seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda img: augment(img, config), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda img: standardise(img, config), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(config.batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    return ds


def build_probe_dataset(config: JEPAConfig, split: str = "train", shuffle: bool = None):
    """(image, label) pairs. Only shuffled and augmented when shuffle is set."""
    ds = load_split(config, split, pretrain=False)
    if shuffle is None:
        shuffle = split == "train"

    if shuffle:
        ds = ds.shuffle(10_000, seed=config.seed, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda img, lab: (augment(img, config), lab), num_parallel_calls=AUTOTUNE
        )
    ds = ds.map(
        lambda img, lab: (standardise(img, config), lab), num_parallel_calls=AUTOTUNE
    )
    ds = ds.batch(config.batch_size).prefetch(AUTOTUNE)
    return ds
