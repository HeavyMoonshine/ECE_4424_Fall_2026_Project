import math
import numpy as np


# Check that probabilities and labels are valid before calibration or evaluation.
def check_inputs(probabilities, labels):
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels)

    if probabilities.ndim != 1 or len(probabilities) == 0:
        raise ValueError("Probabilities must be a nonempty list or array.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must be finite.")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be between 0 and 1.")
    if labels.shape != probabilities.shape or not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Each probability needs a label of 0 or 1.")

    return probabilities, labels.astype(int)


# Use calibration labels to calculate scores and the finite-sample threshold.
def calibrate(probabilities, labels, alpha=0.1):
    probabilities, labels = check_inputs(probabilities, labels)
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")

    scores = np.zeros(len(labels))
    for i in range(len(labels)):
        if labels[i] == 1:
            scores[i] = 1 - probabilities[i]
        else:
            scores[i] = probabilities[i]

    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if rank > len(scores):
        threshold = float("inf")
    else:
        sorted_scores = np.sort(scores)
        threshold = float(sorted_scores[rank - 1])

    return threshold, scores, rank


# Include each label whose score is at most the calibration threshold.
def prediction_sets(probabilities, threshold):
    probabilities, _ = check_inputs(probabilities, np.zeros(len(probabilities)))
    if np.isnan(threshold) or threshold < 0:
        raise ValueError("Invalid threshold.")

    sets = np.zeros((len(probabilities), 2), dtype=bool)
    for i in range(len(probabilities)):
        sets[i, 0] = probabilities[i] <= threshold
        sets[i, 1] = 1 - probabilities[i] <= threshold
    return sets


# Return None when a metric has no examples in its denominator.
def divide(numerator, denominator):
    if denominator == 0:
        return None
    return float(numerator / denominator)


# Compare baseline predictions with conformal decisions; -1 means abstain.
def evaluate(probabilities, labels, sets, baseline_threshold=0.5):
    probabilities, labels = check_inputs(probabilities, labels)
    sets = np.asarray(sets, dtype=bool)
    if sets.shape != (len(labels), 2):
        raise ValueError("Each example needs two label-inclusion flags.")

    decisions = np.full(len(labels), -1)
    covered = np.zeros(len(labels), dtype=bool)
    sizes = sets.sum(axis=1)

    for i in range(len(labels)):
        covered[i] = sets[i, labels[i]]
        if sizes[i] == 1:
            if sets[i, 1]:
                decisions[i] = 1
            else:
                decisions[i] = 0

    baseline = probabilities >= baseline_threshold
    negative = labels == 0
    positive = labels == 1
    decided = decisions != -1

    tp = int(np.sum(baseline & positive))
    fp = int(np.sum(baseline & negative))
    tn = int(np.sum(~baseline & negative))
    fn = int(np.sum(~baseline & positive))
    conformal_fp = int(np.sum((decisions == 1) & negative))
    correct_decisions = int(np.sum((decisions == labels) & decided))

    baseline_metrics = {
        "accuracy": float(np.mean(baseline == labels)),
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "false_positive_rate": divide(fp, fp + tn),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    conformal_metrics = {
        "coverage": float(np.mean(covered)),
        "abstention_rate": float(np.mean(~decided)),
        "mean_set_size": float(np.mean(sizes)),
        "singleton_count": int(np.sum(decided)),
        "empty_count": int(np.sum(sizes == 0)),
        "ambiguous_count": int(np.sum(sizes == 2)),
        "singleton_accuracy": divide(correct_decisions, np.sum(decided)),
        "false_positive_count": conformal_fp,
        "false_positive_rate_all_negatives": divide(conformal_fp, np.sum(negative)),
        "false_positive_rate_decided_negatives": divide(conformal_fp, np.sum(decided & negative)),
    }
    metrics = {"baseline": baseline_metrics, "conformal": conformal_metrics}
    return metrics, decisions, covered
