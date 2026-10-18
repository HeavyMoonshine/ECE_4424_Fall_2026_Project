import os
import pandas as pd


INPUT_FILE = "data/metadata/cumulative.csv"
OUTPUT_FILE = "data/metadata/koi_targets.csv"

NUM_PER_CLASS = 100
RANDOM_SEED = 42


print("Loading KOI catalog...")

header_row = None

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if "kepid" in line and "kepoi_name" in line:
            header_row = i
            break

if header_row is None:
    raise ValueError("Could not find the KOI table header in cumulative.csv")

print(f"Found table header on line {header_row + 1}")

df = pd.read_csv(
    INPUT_FILE,
    skiprows=header_row
)

print(f"Loaded {len(df)} rows.")
print("Columns:")
print(df.columns.tolist())


required_columns = [
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_disposition",
    "koi_period",
    "koi_time0bk"
]

df = df[required_columns].copy()

df = df.dropna(
    subset=[
        "kepid",
        "kepoi_name",
        "koi_disposition",
        "koi_period",
        "koi_time0bk"
    ]
)

confirmed = df[
    df["koi_disposition"] == "CONFIRMED"
].copy()

false_positive = df[
    df["koi_disposition"] == "FALSE POSITIVE"
].copy()

print()
print(f"Confirmed available: {len(confirmed)}")
print(f"False positives available: {len(false_positive)}")


confirmed = confirmed.sample(
    n=min(NUM_PER_CLASS, len(confirmed)),
    random_state=RANDOM_SEED
)

false_positive = false_positive.sample(
    n=min(NUM_PER_CLASS, len(false_positive)),
    random_state=RANDOM_SEED
)


confirmed["label"] = 1
false_positive["label"] = 0


targets = pd.concat(
    [confirmed, false_positive],
    ignore_index=True
)

targets = targets.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)


targets.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print(f"Saved {len(targets)} targets to:")
print(OUTPUT_FILE)

print()
print("Class counts:")
print(targets["label"].value_counts())

print()
print(targets.head())