import os
import pandas as pd
import lightkurve as lk


METADATA_FILE = "data/metadata/koi_targets.csv"
OUTPUT_DIR = "data/raw"

MAX_TARGETS = None

os.makedirs(OUTPUT_DIR, exist_ok=True)


targets = pd.read_csv(METADATA_FILE)

if MAX_TARGETS is not None:
    targets = targets.head(MAX_TARGETS)


download_log = []


for index, row in targets.iterrows():

    kepid = int(row["kepid"])
    kepoi_name = row["kepoi_name"]
    label = int(row["label"])

    search_name = f"KIC {kepid}"

    print()
    print(
        f"[{index + 1}/{len(targets)}] "
        f"Searching {kepoi_name} ({search_name})..."
    )

    try:
        search_result = lk.search_lightcurve(
            search_name,
            mission="Kepler"
        )

        if len(search_result) == 0:
            print("No light curves found.")

            download_log.append({
                "kepid": kepid,
                "kepoi_name": kepoi_name,
                "label": label,
                "status": "not_found",
                "num_lightcurves": 0
            })

            continue

        print(
            f"Found {len(search_result)} light curves. "
            "Downloading..."
        )

        lc_collection = search_result.download_all()

        if lc_collection is None:
            print("Download failed.")

            download_log.append({
                "kepid": kepid,
                "kepoi_name": kepoi_name,
                "label": label,
                "status": "download_failed",
                "num_lightcurves": 0
            })

            continue


        safe_name = kepoi_name.replace(".", "_")

        target_dir = os.path.join(
            OUTPUT_DIR,
            safe_name
        )

        os.makedirs(
            target_dir,
            exist_ok=True
        )


        for i, lc in enumerate(lc_collection):

            file_path = os.path.join(
                target_dir,
                f"lightcurve_{i}.csv"
            )

            df = lc.to_pandas().reset_index()

            df.to_csv(
                file_path,
                index=False
            )


        print(
            f"Saved {len(lc_collection)} light curves."
        )


        download_log.append({
            "kepid": kepid,
            "kepoi_name": kepoi_name,
            "label": label,
            "status": "success",
            "num_lightcurves": len(lc_collection)
        })


    except Exception as e:

        print(f"Error: {e}")

        download_log.append({
            "kepid": kepid,
            "kepoi_name": kepoi_name,
            "label": label,
            "status": "error",
            "num_lightcurves": 0
        })


log_df = pd.DataFrame(download_log)

log_path = "data/metadata/download_log.csv"

log_df.to_csv(
    log_path,
    index=False
)


print()
print("Download complete.")
print()
print(log_df)