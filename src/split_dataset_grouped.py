"""Create reproducible, star-disjoint Kepler candidate data splits.

Run from the repo with:
    python src/split_dataset_grouped.py

This script does not retrain a model or overwrite existing split files.
It expects X.npy, y.npy, and processed_metadata.csv from build_dataset.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

SEED = 42
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
SPLITS = ("train", "val", "cal", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    x_path = DATA_DIR / "X.npy"
    y_path = DATA_DIR / "y.npy"
    metadata_path = DATA_DIR / "processed_metadata.csv"

    X = np.load(x_path, allow_pickle=False)
    y = np.load(y_path, allow_pickle=False)
    meta = pd.read_csv(metadata_path)

    required = {"kepid", "kepoi_name", "label"}
    if not required.issubset(meta.columns):
        raise ValueError(f"Missing metadata columns: {required - set(meta.columns)}")
    if X.ndim != 2 or X.shape[1] != 500:
        raise ValueError(f"Expected X shape (n, 500), got {X.shape}")
    if y.ndim != 1 or len(X) != len(y) or len(y) != len(meta):
        raise ValueError("X, y, and metadata must have the same row count")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinity")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y must contain only 0 and 1")
    if meta[["kepid", "kepoi_name", "label"]].isna().any().any():
        raise ValueError("Missing star ID, KOI ID, or label")
    if meta["kepoi_name"].duplicated().any():
        raise ValueError("Repeated KOI IDs found")
    if not np.array_equal(y, meta["label"].to_numpy()):
        raise ValueError("Labels in y.npy do not match metadata row order")

    # This method requires a single class per star; verified for this dataset.
    # Fail loudly rather than silently assign a misleading group label.
    n_labels_per_star = meta.groupby("kepid")["label"].nunique()
    if (n_labels_per_star != 1).any():
        raise ValueError("A star has mixed candidate labels; use a different grouped splitting strategy")

    stars = meta[["kepid", "label"]].drop_duplicates(subset="kepid")
    train_stars, temp_stars = train_test_split(
        stars, test_size=0.40, random_state=SEED, stratify=stars["label"]
    )
    val_stars, remaining_stars = train_test_split(
        temp_stars, test_size=0.625, random_state=SEED,
        stratify=temp_stars["label"]
    )
    cal_stars, test_stars = train_test_split(
        remaining_stars, test_size=0.60, random_state=SEED,
        stratify=remaining_stars["label"]
    )
    groups = {
        "train": train_stars,
        "val": val_stars,
        "cal": cal_stars,
        "test": test_stars,
    }

    indices = {}
    assignments = np.full(len(meta), "", dtype=object)
    seen_stars = set()
    for split, group_frame in groups.items():
        ids = set(group_frame["kepid"])
        if seen_stars.intersection(ids):
            raise AssertionError("A star appears in multiple splits")
        seen_stars.update(ids)
        indices[split] = np.flatnonzero(meta["kepid"].isin(ids).to_numpy())
        assignments[indices[split]] = split

    if len(seen_stars) != len(stars) or np.any(assignments == ""):
        raise AssertionError("Some stars or examples were not assigned")
    if sum(len(x) for x in indices.values()) != len(meta):
        raise AssertionError("Some examples were assigned more than once")

    output_files = [
        DATA_DIR / f"{prefix}_{split}.npy"
        for split in SPLITS for prefix in ("X", "y")
    ] + [
        DATA_DIR / "grouped_split_manifest.csv",
        DATA_DIR / "grouped_split_summary.json",
    ]
    existing = [str(path) for path in output_files if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing split artifacts:\n"
            + "\n".join(existing)
            + "\nBack up or move them deliberately before rerunning."
        )

    # Preserve original data order within each partition.
    for split in SPLITS:
        idx = indices[split]
        np.save(DATA_DIR / f"X_{split}.npy", X[idx])
        np.save(DATA_DIR / f"y_{split}.npy", y[idx])

    manifest = meta[["kepid", "kepoi_name", "label"]].copy()
    manifest.insert(0, "original_row", np.arange(len(meta)))
    manifest["split"] = assignments
    manifest.to_csv(DATA_DIR / "grouped_split_manifest.csv", index=False)

    summary = {
        "method": "stratified unique-star three-stage split",
        "unit_of_separation": "kepid",
        "seed": SEED,
        "intended_fraction_by_star": {
            "train": 0.60, "val": 0.15, "cal": 0.10, "test": 0.15
        },
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "sha256_inputs": {
            "X.npy": sha256_file(x_path),
            "y.npy": sha256_file(y_path),
            "processed_metadata.csv": sha256_file(metadata_path),
        },
        "splits": {},
    }
    for split in SPLITS:
        idx = indices[split]
        labels, counts = np.unique(y[idx], return_counts=True)
        summary["splits"][split] = {
            "examples": int(len(idx)),
            "distinct_stars": int(meta.iloc[idx]["kepid"].nunique()),
            "class_counts": {str(k): int(v) for k, v in zip(labels, counts)},
        }

    (DATA_DIR / "grouped_split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("STAR-GROUPED SPLIT COMPLETE")
    print("Total examples:", len(meta))
    print("Unique stars:", len(stars))
    for split, record in summary["splits"].items():
        print(
            f"{split:5s}: {record['examples']:3d} examples, "
            f"{record['distinct_stars']:3d} stars, "
            f"labels {record['class_counts']}"
        )
    print("No star IDs overlap between partitions.")
    print("Saved: grouped_split_manifest.csv, grouped_split_summary.json")


if __name__ == "__main__":
    main()
