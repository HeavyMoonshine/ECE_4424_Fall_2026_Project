import json
import os
import numpy as np
import pandas as pd
import torch

from baseline.model import BaselineCNN
from conformal.prediction import calibrate, prediction_sets, evaluate

DATA_DIR = "data/processed"
MODEL_PATH = "results/grouped_v1/baseline/baseline_cnn.pt"
OUTPUT_DIR = "results/grouped_v1/conformal"
ALPHAS = [0.05, 0.1, 0.2]


# Load one saved split and check its shape, labels, and missing values.
def load_data(split, sequence_length):
    X = np.load(f"{DATA_DIR}/X_{split}.npy", allow_pickle=False)
    y = np.load(f"{DATA_DIR}/y_{split}.npy", allow_pickle=False)

    if X.ndim != 2 or X.shape[1] != sequence_length or len(X) == 0:
        raise ValueError(f"X_{split} must have shape (n, {sequence_length}).")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"X_{split} contains missing or infinite values.")
    if y.shape != (len(X),) or not np.all(np.isin(y, [0, 1])):
        raise ValueError(f"y_{split} must contain one binary label per example.")
    return X, y.astype(int)


# Get positive-class probabilities from the frozen CNN in batches.
def predict(model, X):
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(X), 256):
            batch = torch.tensor(X[start:start + 256], dtype=torch.float32)
            batch = batch.unsqueeze(1)
            logits = model(batch)
            probabilities.extend(torch.sigmoid(logits).numpy())
    return np.asarray(probabilities)


# Calibrate on the calibration split, evaluate on test data, and save results.
def main():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = BaselineCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sequence_length = checkpoint["sequence_length"]
    baseline_threshold = checkpoint.get("threshold", 0.5)
    X_cal, y_cal = load_data("cal", sequence_length)
    X_test, y_test = load_data("test", sequence_length)
    cal_probabilities = predict(model, X_cal)
    test_probabilities = predict(model, X_test)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_metrics = []

    for alpha in ALPHAS:
        threshold, scores, rank = calibrate(cal_probabilities, y_cal, alpha)
        sets = prediction_sets(test_probabilities, threshold)
        metrics, decisions, covered = evaluate(
            test_probabilities, y_test, sets, baseline_threshold
        )

        # JSON cannot represent infinity, so record it with a separate flag.
        saved_threshold = threshold
        if np.isinf(threshold):
            saved_threshold = None

        metrics["alpha"] = alpha
        metrics["target_coverage"] = 1 - alpha
        metrics["threshold"] = saved_threshold
        metrics["threshold_is_infinite"] = bool(np.isinf(threshold))
        metrics["quantile_rank"] = rank
        metrics["n_calibration"] = len(y_cal)
        metrics["n_test"] = len(y_test)
        metrics["baseline_threshold"] = baseline_threshold
        all_metrics.append(metrics)

        predictions = pd.DataFrame({
            "sample_index": np.arange(len(y_test)),
            "y_true": y_test,
            "p_positive": test_probabilities,
            "baseline_prediction": (test_probabilities >= baseline_threshold).astype(int),
            "include_0": sets[:, 0],
            "include_1": sets[:, 1],
            "set_size": sets.sum(axis=1),
            "decision": decisions,
            "abstain": decisions == -1,
            "covered": covered,
        })
        predictions.to_csv(f"{OUTPUT_DIR}/predictions_alpha_{alpha}.csv", index=False)

    calibration = pd.DataFrame({
        "sample_index": np.arange(len(y_cal)),
        "y_true": y_cal,
        "p_positive": cal_probabilities,
        "score": scores,
    })
    calibration.to_csv(f"{OUTPUT_DIR}/calibration_scores.csv", index=False)

    with open(f"{OUTPUT_DIR}/metrics.json", "w") as file:
        json.dump(all_metrics, file, indent=2, allow_nan=False)
    print(json.dumps(all_metrics, indent=2, allow_nan=False))
    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

