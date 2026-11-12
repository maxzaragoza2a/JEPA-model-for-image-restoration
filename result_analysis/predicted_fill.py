"""Visualise les prédictions du modèle sur des images de test.

I-JEPA prédit des embeddings, pas des pixels : il n'y a pas de décodeur. Pour
voir une prédiction, chaque patch masqué est remplacé par le patch d'une AUTRE
image de test dont l'embedding (encodeur cible) est le plus proche, en distance
L2, de l'embedding prédit (retrieval par plus proche voisin).

Lignes : image originale, ce que voit le modèle (contexte), prédiction (contexte
+ patchs retrouvés dans les blocs cibles). Un seul masque pour toutes les images,
comme à l'entraînement.

Usage (depuis la racine du projet) :
    python result_analysis/predicted_fill.py
    python result_analysis/predicted_fill.py --seed 3 --images 3
    python result_analysis/predicted_fill.py --run cifar10_d128
    python result_analysis/predicted_fill.py --target-scale 0.07 0.10   # blocs 2x plus petits
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
from jepa.visualise import (BLOCK_COLOURS, block_bounds,  # noqa: E402
                            context_view, denormalise, patch_region)

# 2048 images de test, soit 131k patchs candidats en grille 8x8 et 295k en
# grille 12x12 (stl10_vits), quelle que soit la taille de batch du run.
BANK_IMAGES = 2048


def load_model(config, images, rng, run):
    """Même séquence de build que dans train.py, puis chargement des poids du run."""
    model = JEPA(config)
    model.init_target_encoder()
    model(images, *sample_masks(config, rng))
    model.load_weights(checkpoint_path(run, ROOT))
    return model


class PatchBank:
    """Embedding cible et pixels de chaque patch de N images de test du dataset du run."""

    def __init__(self, model, config, num_images=BANK_IMAGES):
        test = build_probe_dataset(config, split="test").unbatch().take(num_images).batch(128)
        batches = [img for img, _ in test]
        self.batches = batches
        self.images = np.concatenate(batches)  # standardisées
        self.pixels = denormalise(self.images, config)
        emb = np.concatenate([np.asarray(model.target_encoder(b, training=False)) for b in batches])
        self.emb = emb.reshape(len(self.images) * config.num_patches, -1)
        self.sq = (self.emb ** 2).sum(axis=1)
        self.owner = np.repeat(np.arange(len(self.images)), config.num_patches)
        self.config = config

    def __len__(self):
        return len(self.emb)

    def distances(self, query):
        """Distance L2 (à une constante près) de query (D,) à chaque patch."""
        return self.sq - 2 * self.emb @ query

    def patch(self, flat_index):
        """Pixels du patch numéro flat_index de la banque."""
        image, token = divmod(int(flat_index), self.config.num_patches)
        rows, cols = patch_region(token, self.config)
        return self.pixels[image][rows, cols]


def predicted_tokens(model, images, context_indices, target_indices):
    """-> {token: (B, D)}, la prédiction de chaque token masqué.

    Les blocs cibles peuvent se chevaucher : on moyenne les prédictions.
    """
    preds = model.predict_targets(images, context_indices, target_indices, training=False)
    sums, counts = {}, {}
    for block, pred in zip(target_indices, preds):
        pred = np.asarray(pred)
        for j, token in enumerate(block):
            sums[int(token)] = sums.get(int(token), 0) + pred[:, j]
            counts[int(token)] = counts.get(int(token), 0) + 1
    return {t: sums[t] / counts[t] for t in sums}


def fill_images(model, bank, images, context_indices, target_indices, exclude=None):
    """Comble les blocs cibles de chaque image par retrieval dans la banque.

    Args:
        images: (B, H, W, C), standardisées.
        exclude: pour chaque image, son indice dans la banque (ses propres
            patchs sont alors interdits), ou None si elle n'y est pas.

    Returns:
        (fills, hits, total) : images comblées dans [0, 1], et, si exclude est
        donné, combien de vrais patchs sont le plus proche voisin de leur
        prédiction, banque complète.
    """
    config = bank.config
    preds = predicted_tokens(model, images, context_indices, target_indices)
    originals = denormalise(images, config)
    fills, hits, total = [], 0, 0
    for i in range(len(images)):
        fill = context_view(originals[i], context_indices, config)
        for token, pred in preds.items():
            dist = bank.distances(pred[i])
            if exclude is not None:
                hits += int(np.argmin(dist) == exclude[i] * config.num_patches + token)
                total += 1
                dist[bank.owner == exclude[i]] = np.inf
            rows, cols = patch_region(token, config)
            fill[rows, cols] = bank.patch(np.argmin(dist))
        fills.append(fill)
    return fills, hits, total


def plot_fills(originals, fills, context_indices, target_indices, config, title, out,
               names=None):
    """Grille 3 lignes : original, contexte, prédiction. Sauvée dans out."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = len(originals)
    fig, axes = plt.subplots(3, n, figsize=(2.6 * n, 8.2), squeeze=False)
    labels = ["image originale", f"vu par le modèle ({context_indices.size} patchs)",
              "prédiction (retrieval)"]
    for col in range(n):
        panels = [originals[col], context_view(originals[col], context_indices, config), fills[col]]
        for row, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(labels[row], fontsize=9)
        if names:
            axes[0, col].set_title(names[col], fontsize=8)
        for k, block in enumerate(target_indices):
            left, top, width, height = block_bounds(block, config)
            axes[2, col].add_patch(Rectangle(
                (left - 0.5, top - 0.5), width, height, fill=False, linewidth=1.4,
                edgecolor=BLOCK_COLOURS[k % len(BLOCK_COLOURS)]))
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved to {out}")


def mask_summary(context_indices, target_indices, config):
    hidden = np.unique(np.concatenate(target_indices)).size
    return f"{hidden} patchs masqués en {len(target_indices)} blocs (sur {config.num_patches})"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default=DEFAULT_RUN, choices=sorted(PRESETS),
                        help="run à charger (preset de jepa/config.py)")
    parser.add_argument("--seed", type=int, default=0, help="tirage du masque et des images")
    parser.add_argument("--images", type=int, default=3)
    parser.add_argument("--out", default=None,
                        help="image produite (défaut : predicted_samples_<run>.png)")
    parser.add_argument("--target-scale", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="aire de chaque bloc cible en fraction de la grille "
                             "(défaut : celle de l'entraînement, 0.15 0.20)")
    args = parser.parse_args()
    out = args.out or str(Path(__file__).parent / f"predicted_samples_{args.run}.png")

    config = get_config(args.run)
    if args.target_scale:
        config.target_scale = tuple(args.target_scale)
    rng = np.random.default_rng(args.seed)
    first_batch = next(iter(build_probe_dataset(config, split="test")))[0]
    model = load_model(config, first_batch, rng, args.run)
    bank = PatchBank(model, config)

    context_indices, target_indices = sample_masks(config, rng)
    chosen = rng.choice(len(bank.images), size=args.images, replace=False)
    images = bank.images[chosen]

    fills, hits, total = fill_images(model, bank, images, context_indices, target_indices,
                                     exclude=chosen)
    title = (f"I-JEPA ({args.run}), test set  --  seed {args.seed}, "
             f"{mask_summary(context_indices, target_indices, config)}\n"
             f"chaque patch masqué = patch d'une autre image dont l'embedding est le plus proche "
             f"de la prédiction\nvrai patch retrouvé en 1er parmi {len(bank) // 1000}k : "
             f"{100 * hits / total:.0f} % ({hits}/{total})")
    plot_fills(denormalise(images, config), fills, context_indices, target_indices, config,
               title, out)
    print(f"vrai patch en top-1 : {hits}/{total} ({100 * hits / total:.1f} %)")


if __name__ == "__main__":
    main()
