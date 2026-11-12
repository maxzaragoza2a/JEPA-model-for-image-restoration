"""Vérifie si l'encodeur cible s'est effondré (collapse), sur des images de test.

Pour chaque encodeur (entraîné, et aléatoire comme référence), sur N images :
    - std par dimension des embeddings L2-normalisés, moyennée sur les dims.
      Collapse -> ~0. Embeddings bien étalés -> ~1/sqrt(d) (0.088 pour d=128,
      0.0625 pour d=256).
    - nombre de dimensions mortes (std < 1e-3).
    - rang effectif (exp de l'entropie des valeurs singulières normalisées).
      Collapse dimensionnel -> petit devant d.
Mesuré sur deux niveaux : l'embedding moyen de chaque image (moyenne des 64
tokens, 144 pour stl10_vits), et tous les tokens pris séparément.

Usage (depuis la racine du projet) :
    python result_analysis/check_collapse.py                    # DEFAULT_RUN
    python result_analysis/check_collapse.py --run cifar10_d128
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jepa.config import DEFAULT_RUN, PRESETS, get_config  # noqa: E402
from jepa.data import build_probe_dataset  # noqa: E402
from jepa.masking import sample_masks  # noqa: E402
from jepa.model import JEPA  # noqa: E402
from jepa.train import checkpoint_path  # noqa: E402

NUM_BATCHES = 8  # 8 x 128 = 1024 images de test


def build_model(config, images, run=None):
    """Même séquence de build que dans train.py, puis chargement des poids
    du run (aucun si run vaut None : encodeur aléatoire)."""
    model = JEPA(config)
    model.init_target_encoder()
    model(images, *sample_masks(config, np.random.default_rng(0)))
    if run is not None:
        model.load_weights(checkpoint_path(run, ROOT))
    return model


def stats(x):
    """x: (n, d) -> dict de métriques de collapse."""
    x = x - x.mean(axis=0, keepdims=True)
    raw_std = x.std(axis=0)
    z = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    s = np.linalg.svd(z, compute_uv=False)
    p = s / s.sum()
    return {
        "std brute moy.": raw_std.mean(),
        "std normalisée": z.std(axis=0).mean(),
        "dims mortes": int((raw_std < 1e-3).sum()),
        "rang effectif": float(np.exp(-(p * np.log(p + 1e-12)).sum())),
    }


def embed(encoder, batches):
    """-> (embedding moyen par image (n, d), tous les tokens (n * N, d))."""
    tokens = np.concatenate([np.asarray(encoder(b, training=False)) for b in batches])
    return tokens.mean(axis=1), tokens.reshape(-1, tokens.shape[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default=DEFAULT_RUN, choices=sorted(PRESETS),
                        help="run à analyser (preset de jepa/config.py)")
    args = parser.parse_args()

    config = get_config(args.run)
    batches = [img for img, _ in build_probe_dataset(config, split="test").take(NUM_BATCHES)]

    encoders = {
        "aléatoire (réf.)": build_model(config, batches[0]).target_encoder,
        "entraîné": build_model(config, batches[0], args.run).target_encoder,
    }
    embeddings = {name: embed(enc, batches) for name, enc in encoders.items()}

    print(f"\nrun {args.run}, {len(batches) * config.batch_size} images de test, "
          f"d={config.embed_dim} "
          f"(std normalisée idéale ~{1 / np.sqrt(config.embed_dim):.3f}, "
          f"rang effectif max {config.embed_dim})")
    for i, level in enumerate(("par image (moyenne des tokens)", "par token")):
        print(f"\n--- {level} ---")
        print(f"{'encodeur':<18}" + "".join(f"{k:>17}" for k in stats(np.eye(2))))
        for name, emb in embeddings.items():
            m = stats(emb[i])
            print(f"{name:<18}" + "".join(
                f"{v:>17}" if isinstance(v, int) else f"{v:>17.4f}" for v in m.values()))


if __name__ == "__main__":
    main()
