"""Train the canonical XGBoost region classifier on one (source, mode) combo.

Adapts the 5-fold, PID-grouped training loop in
``packages/ibleatools/examples/training_region_predictor_gradient_boosting.py``
(the repo's actual production region-classifier pipeline) to this project's
aggregated tables, restricted to the LF-only feature set (no ``raw_ap``/
``waveforms`` -- these sources never compute AP/spike features) and Cosmos-level
labels.

Run once per (--lfp-source, --snippet-mode) combo, after ``aggregate.py``:
    python train_classifier.py --lfp-source default --snippet-mode all
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics
from xgboost import XGBClassifier

import iblutil.numerical
import ephysatlas.anatomy
import ephysatlas.data
import ephysatlas.features
import ephysatlas.fixtures
import ephysatlas.regionclassifier

import extract

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("train_classifier")

TRAIN_LABEL = "Cosmos_id"
FEATURE_SET = ["raw_lf", "raw_lf_csd"]
N_FOLDS = 5
RANDOM_SEED = 12345
# Not a canonical dated feature-table snapshot (this project builds its own
# aggregate, not a downloaded VINTAGE) -- a fixed project tag instead, required
# by ephysatlas.regionclassifier.save_model's folder-naming convention.
VINTAGE = "2026-08_lfp-decoders"


def train_fold(df_features: pd.DataFrame, x_list: list[str], test_idx: np.ndarray, device: str):
    """Fit one fold's XGBClassifier; return (probas, classifier, accuracy, confusion, classes)."""
    train_idx = ~test_idx
    x_train = df_features.loc[train_idx, x_list].values.astype(float)
    x_test = df_features.loc[test_idx, x_list].values.astype(float)
    y_train = df_features.loc[train_idx, TRAIN_LABEL].values.astype(float)
    y_test = df_features.loc[test_idx, TRAIN_LABEL].values.astype(float)
    classes = np.unique(df_features.loc[train_idx, TRAIN_LABEL])

    _, iy_train = iblutil.numerical.ismember(y_train, classes)

    classifier = XGBClassifier(device=device, verbosity=1)
    classifier.fit(x_train, iy_train)

    y_pred = classes[classifier.predict(x_test)]
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    confusion = sklearn.metrics.confusion_matrix(y_test, y_pred)
    return classifier.predict_proba(x_test), classifier, accuracy, confusion, classes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lfp-source", choices=extract.SOURCES, required=True)
    parser.add_argument("--snippet-mode", choices=extract.SNIPPET_MODES, default="all")
    parser.add_argument("--output-root", type=Path, default=extract.OUTPUT_ROOT)
    parser.add_argument(
        "--device", choices=["cpu", "gpu"], default="cpu",
        help="XGBoost device; default cpu (ask before escalating to gpu)",
    )
    parser.add_argument(
        "--exclude-misaligned", action="store_true",
        help="drop ephysatlas.fixtures.misaligned_pids, matching the canonical training script",
    )
    args = parser.parse_args()

    combo_dir = args.output_root.joinpath(args.lfp_source, args.snippet_mode)
    path_models = combo_dir.joinpath("models")
    path_models.mkdir(parents=True, exist_ok=True)

    brain_atlas = ephysatlas.anatomy.ClassifierAtlas()
    # strict=False: ModelRawFeatures is the full AP+LF schema (requires e.g. rms_ap);
    # this project's sources are LF-only and never compute AP/waveform features.
    df_features = ephysatlas.data.read_features_from_disk(
        combo_dir, brain_atlas=brain_atlas, mappings=["Cosmos"], strict=False
    )
    if args.exclude_misaligned:
        df_features = df_features[
            ~df_features.index.get_level_values(0).isin(ephysatlas.fixtures.misaligned_pids)
        ]

    x_list = ephysatlas.features.voltage_features_set(FEATURE_SET)
    x_list.append("outside")
    # rms_lf_no_car (and any other Optional schema column) is never computed for
    # these sources: LF is already CAR'd upstream by lfpack, so no pre-CAR channel exists.
    x_list = [c for c in x_list if c in df_features.columns]
    rids = np.unique(df_features.loc[:, TRAIN_LABEL])

    all_pids = np.array(df_features.index.get_level_values(0).unique())
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(all_pids)
    ifold = np.floor(np.arange(len(all_pids)) / len(all_pids) * N_FOLDS)

    df_predictions = pd.DataFrame(
        index=df_features.index, columns=list(rids) + ["prediction", "fold"], dtype=float
    )
    fold_accuracies = []
    for i in range(N_FOLDS):
        test_pids = all_pids[ifold == i]
        train_pids = all_pids[ifold != i]
        test_idx = np.isin(df_features.index.get_level_values(0), test_pids)

        probas, classifier, accuracy, confusion, classes = train_fold(
            df_features, x_list, test_idx, args.device
        )
        np.testing.assert_array_equal(classes, rids)
        LOGGER.info("fold %d: accuracy=%.4f (%d test PIDs)", i, accuracy, len(test_pids))
        fold_accuracies.append(accuracy)

        df_predictions.loc[test_idx, rids] = probas
        df_predictions.loc[test_idx, "fold"] = i
        df_predictions.loc[test_idx, "prediction"] = rids[np.argmax(probas, axis=1)]

        meta = dict(
            VINTAGE=VINTAGE,
            RANDOM_SEED=RANDOM_SEED,
            LFP_SOURCE=args.lfp_source,
            SNIPPET_MODE=args.snippet_mode,
            REGION_MAP="Cosmos",
            FEATURES=x_list,
            CLASSES=[int(c) for c in rids],
            ACCURACY=accuracy,
            TRAINING=dict(
                training_size=len(train_pids),
                testing_size=len(test_pids),
                hash_training=iblutil.numerical.hash_uuids(train_pids),
                hash_testing=iblutil.numerical.hash_uuids(test_pids),
            ),
        )
        ephysatlas.regionclassifier.save_model(
            path_models, classifier, meta, subfolder=f"FOLD{i:02d}", identifier="tmp"
        )

    overall_accuracy = sklearn.metrics.accuracy_score(
        df_features[TRAIN_LABEL].values, df_predictions["prediction"].values.astype(int)
    )
    confusion = sklearn.metrics.confusion_matrix(
        df_features[TRAIN_LABEL].values, df_predictions["prediction"].values.astype(int)
    )
    df_predictions.to_parquet(path_models.joinpath("predictions.pqt"))
    np.save(path_models.joinpath("confusion_matrix.npy"), confusion)

    result = {
        "lfp_source": args.lfp_source,
        "snippet_mode": args.snippet_mode,
        "accuracy": float(overall_accuracy),
        "fold_accuracies": [float(a) for a in fold_accuracies],
        "n_channels": int(len(df_features)),
        "n_pids": int(len(all_pids)),
        "classes": [int(c) for c in rids],
    }
    combo_dir.joinpath("accuracy.json").write_text(json.dumps(result, indent=2))
    LOGGER.info("lfp_source=%s snippet_mode=%s: overall accuracy=%.4f", args.lfp_source, args.snippet_mode, overall_accuracy)
    LOGGER.info("Wrote %s", combo_dir.joinpath("accuracy.json"))


if __name__ == "__main__":
    main()
