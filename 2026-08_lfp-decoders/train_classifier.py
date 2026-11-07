"""Train the canonical XGBoost region classifier on one (source, mode) combo.

Adapts the 5-fold, PID-grouped training loop in
``packages/ibleatools/examples/training_region_predictor_gradient_boosting.py``
(the repo's actual production region-classifier pipeline) to this project's
aggregated tables and Cosmos-level labels.

Two axes are compared, both against the SAME held-out test PIDs:

- ``--feature-set``: ``lf`` (default, ``raw_lf``/``raw_lf_csd`` only -- these
  sources never compute AP/waveform features) vs ``full``
  (``ephysatlas.features.voltage_features_set("all")``; groups with no matching
  column in this project's data, e.g. ``raw_ap``/``waveforms``, are silently
  dropped).
- misaligned-insertion policy: ``all_insertions`` (every PID) vs
  ``excluded_misaligned`` (drops ``ephysatlas.fixtures.misaligned_pids`` from
  training). Both are trained automatically in every run. The 5 CV folds are
  drawn from the non-misaligned PIDs only, so the test set is identical -- and
  never includes a misaligned PID -- across both policies; only the training
  pool differs.

Run once per (--lfp-source, --snippet-mode, --feature-set) combo, after
``aggregate.py``:
    python train_classifier.py --lfp-source default --snippet-mode all --feature-set lf
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import ephysatlas.anatomy
import ephysatlas.data
import ephysatlas.features
import ephysatlas.fixtures
import iblutil.numerical
import numpy as np
import pandas as pd
import sklearn.metrics
from xgboost import XGBClassifier

import extract

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("train_classifier")

TRAIN_LABEL = "Cosmos_id"
N_FOLDS = 5
RANDOM_SEED = 12345
# Not a canonical dated feature-table snapshot (this project builds its own
# aggregate, not a downloaded VINTAGE) -- a fixed project tag instead, recorded
# in each fold's meta.json for provenance.
VINTAGE = "2026-08_lfp-decoders"
FEATURE_GROUPS = {
    "lf": ["raw_lf", "raw_lf_csd"],
    "full": "all",
}
MISALIGNED_VARIANTS = ["all_insertions", "excluded_misaligned"]


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


def compute_test_folds(aligned_pids: np.ndarray, n_folds: int, seed: int):
    """Shuffle `aligned_pids` once and assign each to one of `n_folds` test folds.

    Shared across both misaligned-insertion variants so they are evaluated on
    exactly the same held-out PIDs -- misaligned PIDs are never part of
    `aligned_pids`, so they can never land in a test fold either.
    """
    shuffled = np.array(aligned_pids, copy=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)
    ifold = np.floor(np.arange(len(shuffled)) / len(shuffled) * n_folds)
    return shuffled, ifold


def train_variant(
    df_features: pd.DataFrame,
    x_list: list[str],
    shuffled_aligned_pids: np.ndarray,
    ifold: np.ndarray,
    variant: str,
    device: str,
    path_models: Path,
    combo_meta: dict,
) -> dict:
    """Run the N_FOLDS PID-grouped CV for one misaligned-insertion `variant` and
    write its fold models + predictions under path_models/<variant>/.
    """
    df_variant = df_features
    if variant == "excluded_misaligned":
        df_variant = df_features[
            df_features.index.get_level_values(0).isin(shuffled_aligned_pids)
        ]

    variant_dir = path_models.joinpath(variant)
    variant_dir.mkdir(parents=True, exist_ok=True)

    rids = np.unique(df_variant.loc[:, TRAIN_LABEL])
    df_predictions = pd.DataFrame(
        index=df_variant.index, columns=list(rids) + ["prediction", "fold"], dtype=float
    )
    fold_accuracies = []
    for i in range(N_FOLDS):
        test_pids = shuffled_aligned_pids[ifold == i]
        test_idx = np.isin(df_variant.index.get_level_values(0), test_pids)
        train_pids = np.unique(df_variant.index.get_level_values(0)[~test_idx])

        probas, classifier, accuracy, confusion, classes = train_fold(
            df_variant, x_list, test_idx, device
        )
        np.testing.assert_array_equal(classes, rids)
        LOGGER.info(
            "variant=%s fold %d: accuracy=%.4f (%d test PIDs, %d train PIDs)",
            variant, i, accuracy, len(test_pids), len(train_pids),
        )
        fold_accuracies.append(accuracy)

        df_predictions.loc[test_idx, rids] = probas
        df_predictions.loc[test_idx, "fold"] = i
        df_predictions.loc[test_idx, "prediction"] = rids[np.argmax(probas, axis=1)]

        fold_dir = variant_dir.joinpath(f"FOLD{i:02d}")
        fold_dir.mkdir(exist_ok=True)
        classifier.save_model(fold_dir.joinpath("model.ubj"))
        meta = dict(
            combo_meta,
            MISALIGNED_VARIANT=variant,
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
        fold_dir.joinpath("meta.json").write_text(json.dumps(meta, indent=2))

    overall_accuracy = sklearn.metrics.accuracy_score(
        df_variant[TRAIN_LABEL].values, df_predictions["prediction"].values.astype(int)
    )
    confusion = sklearn.metrics.confusion_matrix(
        df_variant[TRAIN_LABEL].values, df_predictions["prediction"].values.astype(int)
    )
    df_predictions.to_parquet(variant_dir.joinpath("predictions.pqt"))
    np.save(variant_dir.joinpath("confusion_matrix.npy"), confusion)

    return {
        "accuracy": float(overall_accuracy),
        "fold_accuracies": [float(a) for a in fold_accuracies],
        "n_pids": len(np.unique(df_variant.index.get_level_values(0))),
        "n_channels": len(df_variant),
        "classes": [int(c) for c in rids],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lfp-source", choices=extract.SOURCES, required=True)
    parser.add_argument("--snippet-mode", choices=extract.SNIPPET_MODES, default="all")
    parser.add_argument("--output-root", type=Path, default=extract.OUTPUT_ROOT)
    parser.add_argument(
        "--feature-set", choices=list(FEATURE_GROUPS), default="lf",
        help="'lf' (default): raw_lf + raw_lf_csd only, matching this project's LF-only "
        "data. 'full': voltage_features_set('all') -- groups with no matching column in "
        "this project's data (e.g. raw_ap/waveforms) are silently dropped.",
    )
    parser.add_argument(
        "--device", choices=["cpu", "gpu"], default="cpu",
        help="XGBoost device; default cpu (ask before escalating to gpu)",
    )
    args = parser.parse_args()

    combo_dir = args.output_root.joinpath(args.lfp_source, args.snippet_mode)
    path_models = combo_dir.joinpath("models", args.feature_set)
    path_models.mkdir(parents=True, exist_ok=True)

    brain_atlas = ephysatlas.anatomy.ClassifierAtlas()
    # strict=False: ModelRawFeatures is the full AP+LF schema (requires e.g. rms_ap);
    # this project's sources are LF-only and never compute AP/waveform features.
    df_features = ephysatlas.data.read_features_from_disk(
        combo_dir, brain_atlas=brain_atlas, mappings=["Cosmos"], strict=False
    )

    x_list = ephysatlas.features.voltage_features_set(FEATURE_GROUPS[args.feature_set])
    x_list.append("outside")
    # rms_lf_no_car (and any other Optional schema column, or a whole group like
    # raw_ap/waveforms under --feature-set full) is never computed for these sources:
    # LF is already CAR'd upstream by lfpack, so no pre-CAR channel exists.
    x_list = [c for c in x_list if c in df_features.columns]

    all_pids = np.unique(df_features.index.get_level_values(0))
    misaligned_present = np.intersect1d(all_pids, ephysatlas.fixtures.misaligned_pids)
    aligned_pids = np.setdiff1d(all_pids, misaligned_present)
    LOGGER.info(
        "feature_set=%s: %d features; %d/%d PIDs are misaligned (excluded from the "
        "excluded_misaligned variant and from every test fold)",
        args.feature_set, len(x_list), len(misaligned_present), len(all_pids),
    )
    shuffled_aligned_pids, ifold = compute_test_folds(aligned_pids, N_FOLDS, RANDOM_SEED)

    combo_meta = dict(
        VINTAGE=VINTAGE,
        RANDOM_SEED=RANDOM_SEED,
        LFP_SOURCE=args.lfp_source,
        SNIPPET_MODE=args.snippet_mode,
        FEATURE_SET=args.feature_set,
        REGION_MAP="Cosmos",
    )
    variant_results = {}
    for variant in MISALIGNED_VARIANTS:
        LOGGER.info("=== variant=%s ===", variant)
        variant_results[variant] = train_variant(
            df_features, x_list, shuffled_aligned_pids, ifold, variant, args.device,
            path_models, combo_meta,
        )

    result = {
        "lfp_source": args.lfp_source,
        "snippet_mode": args.snippet_mode,
        "feature_set": args.feature_set,
        "features": x_list,
        "n_pids": variant_results["all_insertions"]["n_pids"],
        "n_channels": variant_results["all_insertions"]["n_channels"],
        "n_misaligned_pids": len(misaligned_present),
        "classes": variant_results["all_insertions"]["classes"],
        # top-level accuracy/fold_accuracies kept as the "all_insertions" numbers for
        # backward compatibility with report.py, which was already comparing that policy.
        "accuracy": variant_results["all_insertions"]["accuracy"],
        "fold_accuracies": variant_results["all_insertions"]["fold_accuracies"],
        "excluded_misaligned_accuracy": variant_results["excluded_misaligned"]["accuracy"],
        "excluded_misaligned_fold_accuracies": variant_results["excluded_misaligned"]["fold_accuracies"],
    }
    # Only "lf" (the original default) reuses the "accuracy.json" name report.py globs for
    # its LF-compression/saturation comparison; "full" writes alongside it under its own name.
    accuracy_filename = (
        "accuracy.json" if args.feature_set == "lf" else f"accuracy_{args.feature_set}.json"
    )
    combo_dir.joinpath(accuracy_filename).write_text(json.dumps(result, indent=2))
    LOGGER.info(
        "lfp_source=%s snippet_mode=%s feature_set=%s: all_insertions=%.4f "
        "excluded_misaligned=%.4f",
        args.lfp_source, args.snippet_mode, args.feature_set,
        result["accuracy"], result["excluded_misaligned_accuracy"],
    )
    LOGGER.info("Wrote %s", combo_dir.joinpath(accuracy_filename))


if __name__ == "__main__":
    main()
