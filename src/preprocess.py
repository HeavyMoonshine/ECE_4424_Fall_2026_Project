import os
import glob
import numpy as np
import pandas as pd
import lightkurve as lk


RAW_DIR = "data/raw"
METADATA_FILE = "data/metadata/koi_targets.csv"
OUTPUT_DIR = "data/processed"

NUM_BINS = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_target_lightcurves(target_folder):
    csv_files = sorted(
        glob.glob(
            os.path.join(target_folder, "lightcurve_*.csv")
        )
    )

    lightcurves = []

    for file_path in csv_files:
        df = pd.read_csv(file_path)

        if "time" not in df.columns or "flux" not in df.columns:
            continue

        time = df["time"].to_numpy()
        flux = df["flux"].to_numpy()

        lc = lk.LightCurve(
            time=time,
            flux=flux
        )

        lightcurves.append(lc)

    return lightcurves


def clean_and_stitch(lightcurves):
    cleaned = []

    for lc in lightcurves:
        lc_clean = (
            lc.remove_nans()
              .flatten(window_length=101)
              .remove_outliers(
                  sigma_upper=3,
                  sigma_lower=np.inf
              )
        )

        cleaned.append(lc_clean)

    if len(cleaned) == 0:
        return None

    stitched = lk.LightCurveCollection(
        cleaned
    ).stitch()

    return stitched


def fold_lightcurve(
    lightcurve,
    period,
    epoch
):
    folded = lightcurve.fold(
        period=period,
        epoch_time=epoch
    )

    return folded


def convert_to_fixed_length(
    folded_lc,
    num_bins=NUM_BINS
):
    phase = np.asarray(
        folded_lc.time.value,
        dtype=float
    )

    flux = np.asarray(
        folded_lc.flux.value,
        dtype=float
    )

    valid = (
        np.isfinite(phase)
        & np.isfinite(flux)
    )

    phase = phase[valid]
    flux = flux[valid]

    if len(phase) < 10:
        return None

    sort_idx = np.argsort(phase)

    phase = phase[sort_idx]
    flux = flux[sort_idx]

    min_phase = phase.min()
    max_phase = phase.max()

    bin_edges = np.linspace(
        min_phase,
        max_phase,
        num_bins + 1
    )

    bin_centers = (
        bin_edges[:-1]
        + bin_edges[1:]
    ) / 2

    binned_flux = np.full(
        num_bins,
        np.nan
    )

    bin_indices = np.digitize(
        phase,
        bin_edges
    ) - 1

    for i in range(num_bins):
        values = flux[
            bin_indices == i
        ]

        if len(values) > 0:
            binned_flux[i] = np.nanmedian(
                values
            )

    valid_bins = np.isfinite(
        binned_flux
    )

    if valid_bins.sum() < 2:
        return None

    binned_flux = np.interp(
        bin_centers,
        bin_centers[valid_bins],
        binned_flux[valid_bins]
    )

    median = np.median(
        binned_flux
    )

    std = np.std(
        binned_flux
    )

    if std == 0:
        return None

    normalized_flux = (
        binned_flux - median
    ) / std

    return normalized_flux.astype(
        np.float32
    )


def process_target(
    kepoi_name,
    period,
    epoch
):
    safe_name = kepoi_name.replace(
        ".",
        "_"
    )

    target_folder = os.path.join(
        RAW_DIR,
        safe_name
    )

    if not os.path.isdir(
        target_folder
    ):
        return None

    lightcurves = load_target_lightcurves(
        target_folder
    )

    if len(lightcurves) == 0:
        return None

    stitched = clean_and_stitch(
        lightcurves
    )

    if stitched is None:
        return None

    folded = fold_lightcurve(
        stitched,
        period,
        epoch
    )

    fixed = convert_to_fixed_length(
        folded
    )

    return fixed