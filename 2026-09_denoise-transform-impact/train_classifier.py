"""Train the canonical XGBoost Cosmos-region classifier on the ea_active/2026_W39 agg_full
features, once per variant, to check whether denoising and the waveform-shape remap
(``ephysatlas.aggregation.apply_waveform_remap``) change region-decoding accuracy.

Adapts the 5-fold, PID-grouped CV loop in
``sdsc-slurms/2026-08_lfp-decoders/train_classifier.py`` to this project's full
(raw_ap + raw_lf + raw_lf_csd + waveforms + micro-manipulator) feature set, instead of
that project's LF-only subset.

Variants:
    raw               -- raw_ephys_features.pqt (no denoising)
    denoised          -- raw_ephys_features_denoised.pqt (denoised, original ModelSpikeFeatures
                         waveform columns)
    denoised_remapped -- raw_ephys_features_denoised_remapped.pqt (denoised, waveform columns
                         remapped onto the sparser ModelSpikeShapeFeatures set)

Run:
    python train_classifier.py --variant denoised --variant denoised_remapped
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

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("train_classifier")

DATA_DIR = Path(
    "/Users/olivier/Documents/datadisk/ephys-atlas-decoding/features/ea_active/2026_W39/agg_full"
)
TRAIN_LABEL = "Cosmos_id"
N_FOLDS = 5
RANDOM_SEED = 12345
VARIANTS = ["raw", "denoised", "denoised_remapped"]
# Columns that are schema members but not genuine training features: channel_labels is the
# QC flag "outside" is already derived from, and rms_lf_no_car isn't computed for this project.
EXCLUDE_COLUMNS = ["channel_labels", "rms_lf_no_car"]


def _read_remapped_features(
    path_features: Path, brain_atlas: "ephysatlas.anatomy.ClassifierAtlas", mappings: list[str]
) -> pd.DataFrame:
    """Load ``raw_ephys_features_denoised_remapped.pqt``, mirroring the merge/labelling steps of
    ``ephysatlas.data.read_features_from_disk`` (which hardcodes ``raw_ephys_features_denoised.pqt``
    and validates against ``ModelRawFeatures``, a schema this variant no longer matches after
    ``ephysatlas.aggregation.apply_waveform_remap`` swaps in ``ModelSpikeShapeFeatures``).
    """
    df_features = pd.read_parquet(
        path_features.joinpath("raw_ephys_features_denoised_remapped.pqt")
    )
    df_channels = pd.read_parquet(path_features.joinpath("channels.pqt"))
    duplicate_cols = set(df_features.columns).intersection(set(df_channels.columns))
    df_channels = df_channels.drop(columns=duplicate_cols)
    df_features = df_features.merge(df_channels, how="inner", left_index=True, right_index=True)
    df_features["outside"] = df_features["channel_labels"] == 3

    aids = brain_atlas.get_labels(df_features.loc[:, ["x", "y", "z"]].values, mode="clip")
    df_features["Allen_id"] = aids
    for mapping in mappings:
        df_features[f"{mapping}_id"] = brain_atlas.regions.remap(aids, "Allen", mapping)

    return ephysatlas.data.outlier_treatment(df_features, columns=["alpha_mean", "alpha_std"])


def mask_low_confidence_waveforms(
    df_features: pd.DataFrame, waveform_cols: list[str], spike_count_threshold: float
) -> int:
    """Set waveform-derived columns to NaN on channels whose own (raw, pre-denoise, pre-log2)
    spike rate falls below `spike_count_threshold` spikes/s -- so few spikes contributed to
    those channels' waveform estimates that XGBoost should treat them as missing rather than
    trust the noisy point estimate. Mutates `df_features` in place; returns the channel count masked.

    Args:
        spike_count_threshold: spikes/s; channels strictly below this are masked. 0 masks nothing
            (no raw spike_count is negative).
    """
    df_raw_spike_count = pd.read_parquet(
        DATA_DIR.joinpath("raw_ephys_features.pqt"), columns=["spike_count"]
    )
    raw_spike_count = df_raw_spike_count["spike_count"].reindex(df_features.index)
    low_confidence = raw_spike_count < spike_count_threshold
    df_features.loc[low_confidence, waveform_cols] = np.nan
    return int(low_confidence.sum())


def load_variant(
    variant: str,
    brain_atlas: "ephysatlas.anatomy.ClassifierAtlas",
    spike_count_threshold: float | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load one variant's features dataframe and its list of training columns."""
    if variant == "denoised_remapped":
        df_features = _read_remapped_features(DATA_DIR, brain_atlas, mappings=["Cosmos"])
        waveform_cols = list(ephysatlas.features.ModelSpikeShapeFeatures.to_schema().columns.keys())
        x_list = ephysatlas.features.voltage_features_set(
            ["raw_ap", "raw_lf", "raw_lf_csd", "micro-manipulator"]
        ) + waveform_cols
        if spike_count_threshold:
            n_masked = mask_low_confidence_waveforms(
                df_features, [c for c in waveform_cols if c in df_features.columns],
                spike_count_threshold,
            )
            LOGGER.info(
                "spike_count_threshold=%s: masked waveform features on %d/%d channels",
                spike_count_threshold, n_masked, len(df_features),
            )
    else:
        df_features = ephysatlas.data.read_features_from_disk(
            DATA_DIR,
            brain_atlas=brain_atlas,
            mappings=["Cosmos"],
            # only the denoised variant is guaranteed free of the NaNs raw (pre-denoise)
            # alpha_mean/alpha_std can have on channels with no detected spikes
            strict=(variant == "denoised"),
            load_denoised=(variant == "denoised"),
        )
        x_list = ephysatlas.features.voltage_features_set(
            ["raw_ap", "raw_lf", "raw_lf_csd", "waveforms", "micro-manipulator"]
        )
    x_list = [c for c in x_list if c not in EXCLUDE_COLUMNS]
    x_list.append("outside")
    x_list = [c for c in x_list if c in df_features.columns]
    return df_features, x_list


def train_fold(df_features: pd.DataFrame, x_list: list[str], test_idx: np.ndarray, device: str):
    """Fit one fold's XGBClassifier; return (probas, classifier, accuracy, classes)."""
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
    return classifier.predict_proba(x_test), classifier, accuracy, classes


def train_variant(
    variant: str,
    brain_atlas,
    device: str,
    output_root: Path,
    drop_features: list[str] | None = None,
    run_name: str | None = None,
    spike_count_threshold: float | None = None,
) -> dict:
    """Run the 5-fold, PID-grouped CV for one variant and write its outputs to disk.

    Args:
        drop_features: extra feature columns to remove from the variant's default
            training set (e.g. to ablate the micro-manipulator target coordinates).
        run_name: output subfolder name; defaults to `variant` when not given, so an
            ablation run needs its own name to avoid overwriting the plain variant's output.
        spike_count_threshold: see `mask_low_confidence_waveforms`; only honoured for
            variant='denoised_remapped'.
    """
    run_name = run_name or variant
    LOGGER.info(
        "=== run=%s (variant=%s, drop=%s, spike_count_threshold=%s) ===",
        run_name, variant, drop_features, spike_count_threshold,
    )
    df_features, x_list = load_variant(
        variant, brain_atlas, spike_count_threshold=spike_count_threshold
    )
    if drop_features:
        x_list = [c for c in x_list if c not in drop_features]
    rids = np.unique(df_features.loc[:, TRAIN_LABEL])

    all_pids = np.array(df_features.index.get_level_values(0).unique())
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(all_pids)
    ifold = np.floor(np.arange(len(all_pids)) / len(all_pids) * N_FOLDS)

    path_models = output_root.joinpath(run_name)
    path_models.mkdir(parents=True, exist_ok=True)

    df_predictions = pd.DataFrame(
        index=df_features.index, columns=list(rids) + ["prediction", "fold"], dtype=float
    )
    fold_accuracies = []
    for i in range(N_FOLDS):
        test_pids = all_pids[ifold == i]
        test_idx = np.isin(df_features.index.get_level_values(0), test_pids)

        probas, classifier, accuracy, classes = train_fold(df_features, x_list, test_idx, device)
        np.testing.assert_array_equal(classes, rids)
        LOGGER.info("fold %d: accuracy=%.4f (%d test PIDs)", i, accuracy, len(test_pids))
        fold_accuracies.append(accuracy)

        df_predictions.loc[test_idx, rids] = probas
        df_predictions.loc[test_idx, "fold"] = i
        df_predictions.loc[test_idx, "prediction"] = rids[np.argmax(probas, axis=1)]

        fold_dir = path_models.joinpath(f"FOLD{i:02d}")
        fold_dir.mkdir(exist_ok=True)
        classifier.save_model(fold_dir.joinpath("model.ubj"))

    overall_accuracy = sklearn.metrics.accuracy_score(
        df_features[TRAIN_LABEL].values, df_predictions["prediction"].values.astype(int)
    )
    confusion = sklearn.metrics.confusion_matrix(
        df_features[TRAIN_LABEL].values, df_predictions["prediction"].values.astype(int)
    )
    df_predictions.to_parquet(path_models.joinpath("predictions.pqt"))
    np.save(path_models.joinpath("confusion_matrix.npy"), confusion)

    result = {
        "run_name": run_name,
        "variant": variant,
        "drop_features": drop_features or [],
        "spike_count_threshold": spike_count_threshold,
        "train_label": TRAIN_LABEL,
        "features": x_list,
        "accuracy": float(overall_accuracy),
        "fold_accuracies": [float(a) for a in fold_accuracies],
        "n_channels": int(len(df_features)),
        "n_pids": int(len(all_pids)),
        "classes": [int(c) for c in rids],
    }
    path_models.joinpath("accuracy.json").write_text(json.dumps(result, indent=2))
    LOGGER.info("run=%s: overall accuracy=%.4f", run_name, overall_accuracy)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=VARIANTS, action="append", dest="variants",
        help="repeatable; defaults to denoised and denoised_remapped",
    )
    parser.add_argument(
        "--drop", action="append", dest="drop_features", default=None,
        help="repeatable; feature column to remove from every requested variant's training set",
    )
    parser.add_argument(
        "--run-name", default=None,
        help="output subfolder name for this run (required to disambiguate when --drop is used "
        "with a single --variant); defaults to the variant name",
    )
    parser.add_argument(
        "--spike-count-threshold", type=float, action="append", dest="spike_count_thresholds",
        default=None,
        help="repeatable; spikes/s below which denoised_remapped waveform features are set to "
        "NaN. One run per value, each auto-named (ignores --run-name); only valid with "
        "--variant denoised_remapped (and only that variant, since masking isn't implemented "
        "for the other variants).",
    )
    parser.add_argument(
        "--device", choices=["cpu", "gpu"], default="cpu",
        help="XGBoost device; default cpu (ask before escalating to gpu)",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DATA_DIR.joinpath("models_denoise_impact"),
    )
    args = parser.parse_args()
    variants = args.variants or ["denoised", "denoised_remapped"]

    brain_atlas = ephysatlas.anatomy.ClassifierAtlas()
    if args.spike_count_thresholds:
        assert variants == ["denoised_remapped"], (
            "--spike-count-threshold only supports --variant denoised_remapped"
        )
        results = [
            train_variant(
                "denoised_remapped", brain_atlas, args.device, args.output_root,
                drop_features=args.drop_features,
                run_name=f"denoised_remapped_spikethr_{threshold}",
                spike_count_threshold=threshold,
            )
            for threshold in args.spike_count_thresholds
        ]
    else:
        results = [
            train_variant(
                v, brain_atlas, args.device, args.output_root,
                drop_features=args.drop_features, run_name=args.run_name,
            )
            for v in variants
        ]

    LOGGER.info("%-20s %10s %s", "run_name", "accuracy", "fold_accuracies")
    for result in results:
        LOGGER.info(
            "%-20s %10.4f %s",
            result["run_name"],
            result["accuracy"],
            ["%.4f" % a for a in result["fold_accuracies"]],
        )
    comparison_path = args.output_root.joinpath("comparison.json")
    comparison = json.loads(comparison_path.read_text()) if comparison_path.exists() else {}
    comparison.update({r["run_name"]: r["accuracy"] for r in results})
    comparison_path.write_text(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
