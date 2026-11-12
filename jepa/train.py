"""Pre-train the JEPA on CIFAR-10 or STL-10 (config.dataset).

One training step, for one batch:
    sample_masks          -> context_indices, target_indices (shared by the batch)
    model(...)            -> predictions, targets
    jepa_loss             -> scalar
    optimizer             -> context encoder and predictor updated
    update_target_encoder -> target encoder pulled towards the context encoder

The step and the EMA are compiled with tf.function. The mask sizes change at
every batch, so the indices are passed as tensors of unknown length: one
trace serves the whole run.

Usage:
    python -m jepa.train                       # full run of DEFAULT_RUN
    python -m jepa.train --run cifar10_d128    # another preset of jepa/config.py
    python -m jepa.train --max-steps 20        # smoke test
    python -m jepa.train --run stl10_vits --resume   # go on after a stop

The weights go to checkpoints/<run>/jepa.weights.h5 after every epoch, read
by the analysis scripts. checkpoints/<run>/resume/ holds what --resume needs
on top of them: the optimizer state, the step and the state of the mask
generator. To keep the log that result_analysis/plot_loss.py reads:
    python -m jepa.train --run cifar10_d256 > logs/cifar10_d256.log 2>&1
"""

import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf
import keras

from jepa.config import DEFAULT_RUN, PRESETS, JEPAConfig, get_config
from jepa.data import build_pretrain_dataset
from jepa.masking import sample_masks
from jepa.model import JEPA, jepa_loss
from jepa.utils import (CosineWithWarmup, count_params, ema_momentum,
                        format_params, set_seed)

# Steps between two loss lines in the log (read by plot_loss.py).
LOG_EVERY = 50


def build_optimizer(config: JEPAConfig, total_steps: int, warmup_steps: int):
    """AdamW with linear warmup then cosine decay of the learning rate.

    Args:
        config: learning_rate and weight_decay are read from it.
        total_steps: number of optimizer steps of the whole run.
        warmup_steps: number of steps of the linear warmup.

    Returns:
        keras.optimizers.AdamW.
    """
    schedule = CosineWithWarmup(config.learning_rate, total_steps, warmup_steps)
    optimizer = keras.optimizers.AdamW(learning_rate=schedule, weight_decay=config.weight_decay)
    optimizer.exclude_from_weight_decay(var_names=["bias", "gamma", "beta", "mask_token"])
    return optimizer


def make_train_step(model: JEPA, optimizer, config: JEPAConfig):
    """Compiled optimizer step on one batch. The EMA is not done here.

    The returned function takes:
        images: (B, H, W, C), standardised.
        context_indices: 1-D int32 tensor, the M visible grid positions.
        target_indices: tuple of config.num_target_blocks 1-D int32 tensors.
    and returns the scalar loss.

    The input signature leaves the batch size and every index length open,
    so a new mask draw does not trigger a new trace.
    """
    indices = tf.TensorSpec([None], tf.int32)
    signature = [
        tf.TensorSpec([None, config.image_size, config.image_size, config.num_channels],
                      tf.float32),
        indices,
        tuple(indices for _ in range(config.num_target_blocks)),
    ]
    # The optimizer creates its slots now rather than inside the trace.
    optimizer.build(model.trainable_weights)

    @tf.function(input_signature=signature)
    def train_step(images, context_indices, target_indices):
        with tf.GradientTape() as tape:
            predictions, targets = model(images, context_indices, list(target_indices),
                                         training=True)
            loss = jepa_loss(predictions, targets)
        weights = model.trainable_weights
        grads = tape.gradient(loss, weights)
        optimizer.apply_gradients(zip(grads, weights))
        return loss

    return train_step


def make_ema_step(model: JEPA):
    """Compiled model.update_target_encoder, taking a float32 scalar."""
    return tf.function(model.update_target_encoder,
                       input_signature=[tf.TensorSpec([], tf.float32)])


def as_tensors(context_indices, target_indices):
    """Mask draw of sample_masks -> the int32 tensors train_step expects."""
    return (tf.constant(context_indices, tf.int32),
            tuple(tf.constant(t, tf.int32) for t in target_indices))


def train_one_epoch(train_step, ema_step, dataset, rng, step: int,
                    total_steps: int, config: JEPAConfig, max_steps=None):
    """Run over the dataset once: mask, step, EMA, for every batch.

    Args:
        train_step: from make_train_step.
        ema_step: from make_ema_step.
        dataset: from build_pretrain_dataset, yields (B, H, W, C) batches.
        rng: np.random.Generator used to draw the masks.
        step: global step at the start of the epoch.
        total_steps: global step count of the whole run, for the EMA schedule.
        config: masking and EMA hyperparameters.
        max_steps: stop once the global step reaches it, or None.

    Returns:
        (mean_loss, step): mean loss over the batches seen, and the global
        step at the end of the epoch.
    """
    # Kept as tensors: reading one waits for the GPU, so it is only done
    # every LOG_EVERY steps.
    losses = []
    for images in dataset:
        if max_steps is not None and step >= max_steps:
            break

        # One draw per batch, shared by every image of the batch.
        context_indices, target_indices = sample_masks(config, rng)
        losses.append(train_step(images, *as_tensors(context_indices, target_indices)))

        # After the optimizer step: the target follows the updated context encoder.
        momentum = ema_momentum(step, total_steps, config.ema_start, config.ema_end)
        ema_step(tf.constant(momentum, tf.float32))

        step += 1
        if step % LOG_EVERY == 0:
            recent = float(tf.reduce_mean(tf.stack(losses[-LOG_EVERY:])))
            print(f"  step {step}/{total_steps}  loss {recent:.4f}", end="\r", flush=True)

    mean_loss = float(tf.reduce_mean(tf.stack(losses))) if losses else float("nan")
    return mean_loss, step


def checkpoint_path(run: str, root: str = ".") -> str:
    """checkpoints/<run>/jepa.weights.h5, where train() saves that run."""
    return os.path.join(root, "checkpoints", run, "jepa.weights.h5")


def save_checkpoint(model: JEPA, out_dir: str, name: str = "jepa"):
    """Save every weight of the model (both encoders and the predictor).

    Args:
        model: the JEPA.
        out_dir: folder, created if missing (ignored by git: checkpoints/).
        name: file name without extension.

    Returns:
        Path of the saved file.
    """
    # The whole JEPA: VisionTransformer is a Layer and has no save_weights.
    # To reload: JEPA(config), init_target_encoder(), one call on a batch to
    # build the predictor, then load_weights(path).
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.weights.h5")
    model.save_weights(path)
    return path


class ResumeState:
    """Everything --resume needs, in out_dir/resume/.

    Per saved epoch: model-<epoch>.weights.h5 (every weight, both encoders
    and the predictor) and optimizer-<epoch>.npz (AdamW moments, and the
    iteration count that drives the learning rate). state.json names that
    epoch and holds the step, the run length, the history and the state of
    the mask generator. It is written last, so a stop during a save leaves
    the previous epoch usable.

    tf.train.Checkpoint is not used: it does not follow the blocks that the
    layers keep in Python lists, and would leave them out.

    A run resumes at the start of the epoch after the last one saved. The
    shuffle order of the dataset is not restored.
    """

    def __init__(self, out_dir: str, model: JEPA, optimizer):
        self.dir = os.path.join(out_dir, "resume")
        self.json_path = os.path.join(self.dir, "state.json")
        self.model = model
        self.optimizer = optimizer

    def _paths(self, epoch: int):
        return (os.path.join(self.dir, f"model-{epoch}.weights.h5"),
                os.path.join(self.dir, f"optimizer-{epoch}.npz"))

    def save(self, epoch: int, step: int, total_steps: int, rng, history: list):
        os.makedirs(self.dir, exist_ok=True)
        model_path, optimizer_path = self._paths(epoch)
        self.model.save_weights(model_path)
        store = {}
        self.optimizer.save_own_variables(store)
        np.savez(optimizer_path, **store)
        previous = self._saved_epoch()

        state = {
            "epoch": epoch,
            "step": step,
            "total_steps": total_steps,
            "rng": rng.bit_generator.state,
            "history": history,
        }
        tmp = self.json_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1)
        os.replace(tmp, self.json_path)

        # The new state is on disk: the files of the previous epoch can go.
        if previous is not None and previous != epoch:
            for path in self._paths(previous):
                if os.path.exists(path):
                    os.remove(path)

    def _saved_epoch(self):
        if not os.path.exists(self.json_path):
            return None
        with open(self.json_path, encoding="utf-8") as f:
            return json.load(f)["epoch"]

    def load(self, rng):
        """Restore the weights, the optimizer and rng in place.

        The model and the optimizer must be built (make_train_step builds
        the optimizer).

        Returns:
            The saved state (dict), or None when there is nothing to resume.
        """
        if not os.path.exists(self.json_path):
            return None
        with open(self.json_path, encoding="utf-8") as f:
            state = json.load(f)
        model_path, optimizer_path = self._paths(state["epoch"])
        self.model.load_weights(model_path)
        with np.load(optimizer_path) as store:
            self.optimizer.load_own_variables({key: store[key] for key in store.files})
        rng.bit_generator.state = state["rng"]
        return state


def train(config: JEPAConfig, epochs=None, max_steps=None,
          out_dir: str = "checkpoints", resume: bool = False):
    """Full pre-training run.

    Args:
        config: every hyperparameter.
        epochs: overrides config.epochs when given.
        max_steps: stop after this many optimizer steps, for a smoke test.
        out_dir: where checkpoints are written.
        resume: go on from out_dir/resume/ if a previous run saved there.

    Returns:
        (model, history), history being a list of dicts, one per epoch:
        {"epoch", "loss", "lr", "momentum", "seconds"}.
    """
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    dataset = build_pretrain_dataset(config)
    steps_per_epoch = int(dataset.cardinality())

    epochs = epochs or config.epochs
    total_steps = epochs * steps_per_epoch
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    # Capped at a tenth of the run so that a short run still reaches the
    # cosine decay. A full run (3 of 30 epochs) is not affected.
    warmup_steps = min(config.warmup_epochs * steps_per_epoch, total_steps // 10)

    # The optimizer comes last, once every trainable weight exists:
    # init_target_encoder builds the encoders, one call builds the predictor.
    model = JEPA(config)
    model.init_target_encoder()
    model(next(iter(dataset)), *sample_masks(config, rng))
    optimizer = build_optimizer(config, total_steps, warmup_steps)
    train_step = make_train_step(model, optimizer, config)
    ema_step = make_ema_step(model)

    trainable, non_trainable = count_params(model)
    print(f"parameters: {format_params(trainable)} trainable, "
          f"{format_params(non_trainable)} frozen (target encoder, positions)")
    print(f"{epochs} epochs x {steps_per_epoch} steps, {total_steps} steps in all, "
          f"{warmup_steps} of warmup")

    state = ResumeState(out_dir, model, optimizer)
    history = []
    step = 0
    first_epoch = 1
    if resume:
        saved = state.load(rng)
        if saved is None:
            print("nothing to resume, starting from scratch")
        elif saved["total_steps"] != total_steps:
            raise ValueError(
                f"the saved run has {saved['total_steps']} steps in all, this one "
                f"{total_steps}: keep the same --epochs and --max-steps to resume")
        else:
            history, step, first_epoch = saved["history"], saved["step"], saved["epoch"] + 1
            print(f"resumed after epoch {saved['epoch']}, step {step}")

    for epoch in range(first_epoch, epochs + 1):
        if max_steps is not None and step >= max_steps:
            break
        t0 = time.time()
        loss, step = train_one_epoch(train_step, ema_step, dataset, rng, step,
                                     total_steps, config, max_steps)
        record = {
            "epoch": epoch,
            "loss": loss,
            "lr": float(optimizer.learning_rate),
            "momentum": ema_momentum(step, total_steps, config.ema_start, config.ema_end),
            "seconds": time.time() - t0,
        }
        history.append(record)
        print(f"epoch {epoch:3d}/{epochs}  loss {record['loss']:.4f}  "
              f"lr {record['lr']:.2e}  ema {record['momentum']:.4f}  "
              f"{record['seconds']:.0f}s", flush=True)
        save_checkpoint(model, out_dir)
        state.save(epoch, step, total_steps, rng, history)

    return model, history


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default=DEFAULT_RUN, choices=sorted(PRESETS),
                        help=f"preset of jepa/config.py (default: {DEFAULT_RUN})")
    parser.add_argument("--epochs", type=int, default=None,
                        help="number of epochs (default: config.epochs)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="stop after this many steps, for a smoke test")
    parser.add_argument("--out", default=None,
                        help="where to save the weights (default: checkpoints/<run>)")
    parser.add_argument("--resume", action="store_true",
                        help="go on from the last epoch saved in <out>/resume/, if any")
    args = parser.parse_args()

    out_dir = args.out or os.path.dirname(checkpoint_path(args.run))
    train(get_config(args.run), epochs=args.epochs, max_steps=args.max_steps,
          out_dir=out_dir, resume=args.resume)


if __name__ == "__main__":
    main()
