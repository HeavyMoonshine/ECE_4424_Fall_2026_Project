import os
import numpy as np
import pandas as pd

from preprocess import process_target


METADATA_FILE = "data/metadata/koi_targets.csv"
DOWNLOAD_LOG = "data/metadata/download_log.csv"
OUTPUT_DIR = "data/processed"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


metadata = pd.read_csv(
    METADATA_FILE
)

download_log = pd.read_csv(
    DOWNLOAD_LOG
)

successful_ids = set(
    download_log[
        download_log["status"] == "success"
    ]["kepoi_name"]
)

metadata = metadata[
    metadata["kepoi_name"].isin(
        successful_ids
    )
].copy()


X = []
y = []
processed_rows = []


for i, row in metadata.iterrows():

    kepoi_name = row["kepoi_name"]
    period = row["koi_period"]
    epoch = row["koi_time0bk"]
    label = int(row["label"])

    print(
        f"Processing {kepoi_name}..."
    )

    try:
        fixed_curve = process_target(
            kepoi_name,
            period,
            epoch
        )

        if fixed_curve is None:
            print("Skipped.")
            continue

        X.append(
            fixed_curve
        )

        y.append(
            label
        )

        processed_rows.append({
            "kepid": row["kepid"],
            "kepoi_name": kepoi_name,
            "label": label,
            "period": period,
            "epoch": epoch
        })

        print("Success.")

    except Exception as e:
        print(
            f"Error: {e}"
        )


X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y,
    dtype=np.int64
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "X.npy"
    ),
    X
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y.npy"
    ),
    y
)

processed_df = pd.DataFrame(
    processed_rows
)

processed_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "processed_metadata.csv"
    ),
    index=False
)


print()
print("Done.")
print(
    f"X shape: {X.shape}"
)
print(
    f"y shape: {y.shape}"
)

if len(y) > 0:
    unique, counts = np.unique(
        y,
        return_counts=True
    )

    print(
        "Class counts:",
        dict(
            zip(
                unique,
                counts
            )
        )
    )