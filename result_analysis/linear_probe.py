"""Linear probe : mesure la qualité des représentations par la classification.

L'encodeur cible est figé. Chaque image devient un vecteur, la moyenne de ses
tokens (64 ou 144 selon le run), puis une régression logistique apprend les
classes à partir de ces vecteurs. Plus la précision sur le test est haute, plus
l'information de classe est lisible linéairement dans les représentations.

Données : les splits annotés du dataset du run (CIFAR-10 : 50k train / 10k
test ; STL-10 : 5k train / 8k test), sans augmentation. La régularisation C est
choisie sur 10 % du train mis de côté, puis le classifieur est réentraîné sur
tout le train.

Un encodeur aléatoire de même architecture sert de référence : l'écart entre
les deux est ce que le pré-entraînement a appris.

Usage (depuis la racine du projet) :
    python result_analysis/linear_probe.py --run cifar10_d128
    python result_analysis/linear_probe.py --run cifar10_d128 stl10_vits
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jepa.config import PRESETS, get_config  # noqa: E402
from jepa.data import build_probe_dataset  # noqa: E402
from jepa.masking import sample_masks  # noqa: E402
from jepa.model import JEPA  # noqa: E402
from jepa.train import checkpoint_path  # noqa: E402

C_GRID = (0.01, 0.1, 1.0, 10.0)
VALIDATION_FRACTION = 0.1


def build_model(config, images, run=None):
    """Même séquence de build que dans train.py, puis chargement des poids
    du run (aucun si run vaut None : encodeur aléatoire)."""
    model = JEPA(config)
    model.init_target_encoder()
    model(images, *sample_masks(config, np.random.default_rng(0)))
    if run is not None:
        path = checkpoint_path(run, ROOT)
        if not Path(path).exists():
            sys.exit(f"pas de checkpoint pour {run} : {path}")
        model.load_weights(path)
    return model


def features(encoder, dataset):
    """-> (moyenne des tokens par image (n, d), labels (n,))."""
    feats, labels = [], []
    for images, labs in dataset:
        tokens = encoder(images, training=False)
        feats.append(np.asarray(tokens).mean(axis=1))
        labels.append(np.asarray(labs))
    return np.concatenate(feats), np.concatenate(labels)


def fit(x, y, c):
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(C=c, max_iter=2000).fit(scaler.transform(x), y)
    return scaler, clf


def accuracy(scaler, clf, x, y):
    return float((clf.predict(scaler.transform(x)) == y).mean())


def probe(x_train, y_train, x_test, y_test, seed=0):
    """Choix de C sur une validation, réentraînement sur tout le train.

    Returns:
        (précision test, précision train, C retenu).
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x_train))
    n_val = int(len(x_train) * VALIDATION_FRACTION)
    val, tr = order[:n_val], order[n_val:]
    scores = {c: accuracy(*fit(x_train[tr], y_train[tr], c), x_train[val], y_train[val])
              for c in C_GRID}
    best_c = max(scores, key=scores.get)
    scaler, clf = fit(x_train, y_train, best_c)
    return (accuracy(scaler, clf, x_test, y_test),
            accuracy(scaler, clf, x_train, y_train), best_c)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", nargs="+", required=True, choices=sorted(PRESETS),
                        help="run(s) à évaluer (presets de jepa/config.py)")
    parser.add_argument("--no-random", action="store_true",
                        help="ne pas évaluer l'encodeur aléatoire de référence")
    args = parser.parse_args()

    rows = []
    for run in args.run:
        config = get_config(run)
        train_ds = build_probe_dataset(config, split="train", shuffle=False)
        test_ds = build_probe_dataset(config, split="test", shuffle=False)
        first = next(iter(train_ds))[0]

        encoders = {} if args.no_random else {"aléatoire": None}
        encoders["entraîné"] = run
        for name, weights in encoders.items():
            t0 = time.time()
            encoder = build_model(config, first, weights).target_encoder
            x_train, y_train = features(encoder, train_ds)
            x_test, y_test = features(encoder, test_ds)
            test_acc, train_acc, c = probe(x_train, y_train, x_test, y_test, config.seed)
            rows.append((run, name, test_acc, train_acc, c, len(x_train), len(x_test)))
            print(f"{run} / {name} : {100 * test_acc:.1f} % en test "
                  f"({time.time() - t0:.0f} s)", flush=True)

    print(f"\n{'run':<14}{'encodeur':<11}{'test':>8}{'train':>8}{'C':>7}"
          f"{'n train':>9}{'n test':>8}")
    for run, name, test_acc, train_acc, c, n_train, n_test in rows:
        print(f"{run:<14}{name:<11}{100 * test_acc:>7.1f}%{100 * train_acc:>7.1f}%"
              f"{c:>7g}{n_train:>9}{n_test:>8}")


if __name__ == "__main__":
    main()
