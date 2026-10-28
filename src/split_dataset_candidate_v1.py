import os
import numpy as np
from sklearn.model_selection import train_test_split


INPUT_DIR = "data/processed"
OUTPUT_DIR = "results/candidate_split_v1/dataset_snapshot"

RANDOM_SEED = 42


X = np.load(
    os.path.join(INPUT_DIR, "X.npy")
)

y = np.load(
    os.path.join(INPUT_DIR, "y.npy")
)


# 60% training, 40% remaining
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=RANDOM_SEED,
    stratify=y
)


# Split remaining 40% into:
# 15% validation and 25% remaining
X_val, X_remaining, y_val, y_remaining = train_test_split(
    X_temp,
    y_temp,
    test_size=0.625,
    random_state=RANDOM_SEED,
    stratify=y_temp
)


# Split remaining 25% into:
# 10% calibration and 15% test
X_cal, X_test, y_cal, y_test = train_test_split(
    X_remaining,
    y_remaining,
    test_size=0.60,
    random_state=RANDOM_SEED,
    stratify=y_remaining
)


np.save(
    os.path.join(OUTPUT_DIR, "X_train.npy"),
    X_train
)

np.save(
    os.path.join(OUTPUT_DIR, "y_train.npy"),
    y_train
)

np.save(
    os.path.join(OUTPUT_DIR, "X_val.npy"),
    X_val
)

np.save(
    os.path.join(OUTPUT_DIR, "y_val.npy"),
    y_val
)

np.save(
    os.path.join(OUTPUT_DIR, "X_cal.npy"),
    X_cal
)

np.save(
    os.path.join(OUTPUT_DIR, "y_cal.npy"),
    y_cal
)

np.save(
    os.path.join(OUTPUT_DIR, "X_test.npy"),
    X_test
)

np.save(
    os.path.join(OUTPUT_DIR, "y_test.npy"),
    y_test
)


print("Dataset split complete.")
print()

print(f"Training:    {X_train.shape}")
print(f"Validation:  {X_val.shape}")
print(f"Calibration: {X_cal.shape}")
print(f"Test:        {X_test.shape}")

print()

print(
    "Training classes:",
    dict(zip(*np.unique(y_train, return_counts=True)))
)

print(
    "Validation classes:",
    dict(zip(*np.unique(y_val, return_counts=True)))
)

print(
    "Calibration classes:",
    dict(zip(*np.unique(y_cal, return_counts=True)))
)

print(
    "Test classes:",
    dict(zip(*np.unique(y_test, return_counts=True)))
)
