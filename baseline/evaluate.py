import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)

import os
import json

from model import BaselineCNN

DATA_DIR = "data/processed"
OUTPUT_DIR = "results/baseline"
os.makedirs(OUTPUT_DIR, exist_ok=True)

THRESHOLD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------- load test data for evaluation -------

X_test = np.load(f"{DATA_DIR}/X_test.npy")
y_test = np.load(f"{DATA_DIR}/y_test.npy").astype(int)

# add data validation here if needed

checkpoint = torch.load(
    f"{OUTPUT_DIR}/baseline_cnn.pt",
    map_location=device,
    weights_only=True,
)

model = BaselineCNN().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ------- Generate 1 probability per test example -------

X_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

probabilities = []

with torch.no_grad():
    for start in range(0, len(X_tensor), 256):
        batch = X_tensor[start:start + 256].to(device)
        logits = model(batch)
        probs = torch.sigmoid(logits)
        probabilities.extend(probs.cpu().numpy())

probabilities = np.asarray(probabilities)
predictions = (probabilities >= THRESHOLD).astype(int)
correct = (predictions == y_test)

# ------- CONFUSION MATRIX -------
# structure: [[TN, FP], [FN, TP]]

tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0,1]).ravel()

# ------- METRICS -------

# standard metrics
metrics = {
    "threshold": THRESHOLD,
    "n_test": int(len(y_test)),
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(precision_score(y_test, predictions, zero_division=0)),
    "recall": float(recall_score(y_test, predictions, zero_division=0)),
    "f1": float(f1_score(y_test, predictions, zero_division=0)),
    "false_positive_rate": float(fp / (fp + tn) if (fp + tn) else 0.0),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
}

# Area Under the Receiver Operating Characteristic Curve (ROC AUC)
# measures how well the model distinguished between classes 
if len(np.unique(y_test)) == 2:
    metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
    metrics["pr_auc_average_precision"] = float(average_precision_score(y_test, probabilities))


# Save one row for every held-out example
results = pd.DataFrame({
    "sample_index": np.arange(len(y_test)),
    "y_true": y_test,
    "predicted_probability": probabilities,
    "y_pred": predictions,
    "correct": correct,
})

# ------- OUTPUT -------

# results csv
results.to_csv(
    f"{OUTPUT_DIR}/test_predictions.csv",
    index=False,
)

with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# confusion matrix
pd.DataFrame(
    confusion_matrix(y_test, predictions, labels=[0,1]),
    index=["actual_0", "actual_1"],
    columns=["predicted_0", "predicted_1"],
).to_csv(f"{OUTPUT_DIR}/confusion_matrix.csv")

# classification report
with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
    f.write(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["non_transit", "transit"],
        zero_division=0,
    ))

print(json.dumps(metrics, indent=2))
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["non_transit", "transit"],
    zero_division=0,
))
