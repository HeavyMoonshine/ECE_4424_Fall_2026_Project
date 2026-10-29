"""Reproducible synthetic input-noise stress test for reproduced original CNN.

Run from an active virtual environment, with this file in the repository root:
    python -u .\noise_stress_original.py

Uses Connor's ORIGINAL frozen baseline CNN and original calibration/test arrays.
Requires original split arrays under results/original_v1/dataset_snapshot/.
Does not modify any existing datasets, checkpoints, or evaluation results.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from baseline.model import BaselineCNN
from conformal.prediction import calibrate, evaluate, prediction_sets

ROOT = Path(__file__).resolve().parent
ALPHAS = (0.05, 0.10, 0.20)
CONDITIONS = (("clean", 0.0), ("low", 0.10), ("medium", 0.25), ("high", 0.50))
DEFAULT_SEED = 20261028


def sha256_file(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_partition(name, expected_length):
    d = ROOT / "results" / "original_v1" / "dataset_snapshot"
    x = np.load(d / f"X_{name}.npy", allow_pickle=False)
    y = np.load(d / f"y_{name}.npy", allow_pickle=False)
    if x.ndim != 2 or x.shape[1] != expected_length or x.shape[0] == 0:
        raise ValueError(f"Invalid X_{name} shape: {x.shape}")
    if y.shape != (len(x),) or not np.all(np.isin(y, [0, 1])):
        raise ValueError(f"Invalid y_{name} shape or binary labels")
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite inputs in X_{name}")
    return x.astype(np.float32, copy=False), y.astype(int)


def infer_probabilities(model, x):
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            x_batch = torch.from_numpy(np.ascontiguousarray(x[start:start + 256])).unsqueeze(1)
            logits = model(x_batch)
            predictions.append(torch.sigmoid(logits).cpu().numpy())
    p = np.concatenate(predictions).astype(float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Model produced invalid probabilities")
    return p


def get_rows_and_predictions(p, y, condition, sigma, trial, alpha_thresholds, baseline_threshold):
    p = np.asarray(p)
    baseline_pred = (p >= baseline_threshold).astype(int)
    negative = y == 0
    pos_mask = y == 1
    tn = int(np.sum((baseline_pred == 0) & negative))
    fp = int(np.sum((baseline_pred == 1) & negative))
    fn = int(np.sum((baseline_pred == 0) & pos_mask))
    tp = int(np.sum((baseline_pred == 1) & pos_mask))
    correct = int(np.sum(baseline_pred == y))
    n = len(y)
    fpr = fp / int(negative.sum())
    accuracy = correct / n
    core = {"condition": condition, "sigma": sigma, "trial": trial, "n_test": n}
    metrics = [{
        **core, "method": "baseline", "alpha": None,
        "accuracy": accuracy, "false_positive_rate_all_negatives": fpr,
        "false_positive_rate_decided_negatives": fpr,
        "coverage": None, "abstention_rate": 0.0,
        "singleton_accuracy": None, "decided_accuracy": accuracy,
        "n_decided": n, "n_abstained": 0, "n_correct_decided": correct,
        "n_incorrect_decided": n - correct, "false_positive_count": fp,
        "n_empty": None, "n_ambiguous": None, "mean_set_size": None,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }]
    predictions = pd.DataFrame({
        "condition": condition, "sigma": sigma, "trial": trial,
        "sample_index": np.arange(n), "y_true": y,
        "p_positive": p, "baseline_prediction": baseline_pred,
    })

    for alpha in ALPHAS:
        threshold = alpha_thresholds[alpha]
        sets = prediction_sets(p, threshold)
        all_metrics, decisions, covered = evaluate(p, y, sets, baseline_threshold)
        cp = all_metrics["conformal"]
        decided = decisions != -1
        n_correct = int(np.sum(decided & (decisions == y)))
        n_decided = int(decided.sum())
        metrics.append({
            **core, "method": f"cp_{alpha:g}", "alpha": alpha,
            "accuracy": None,
            "false_positive_rate_all_negatives": cp["false_positive_rate_all_negatives"],
            "false_positive_rate_decided_negatives": cp["false_positive_rate_decided_negatives"],
            "coverage": cp["coverage"], "abstention_rate": cp["abstention_rate"],
            "singleton_accuracy": cp["singleton_accuracy"],
            "decided_accuracy": cp["singleton_accuracy"],
            "n_decided": n_decided, "n_abstained": n - n_decided,
            "n_correct_decided": n_correct,
            "n_incorrect_decided": n_decided - n_correct,
            "false_positive_count": cp["false_positive_count"],
            "n_empty": cp["empty_count"], "n_ambiguous": cp["ambiguous_count"],
            "mean_set_size": cp["mean_set_size"],
            "tn": None, "fp": None, "fn": None, "tp": None,
        })
        label = f"cp_{alpha:g}"
        predictions[f"{label}_include_0"] = sets[:, 0]
        predictions[f"{label}_include_1"] = sets[:, 1]
        predictions[f"{label}_set_size"] = sets.sum(axis=1)
        predictions[f"{label}_decision"] = decisions
        predictions[f"{label}_covered"] = covered
    return metrics, predictions


def verify_reference(clean_p, y_test, cal_p, y_cal, thresholds, clean_rows):
    """Strictly verify the just-reproduced original_v1 run, and compare historically.

    The historical repo metrics were generated with slightly different probabilities,
    so report those discrepancies rather than requiring exact historical equality.
    """
    root = ROOT / "results"
    reproduced = root / "original_v1"
    b = pd.read_csv(reproduced / "baseline/test_predictions.csv")
    b_json = json.loads((reproduced / "baseline/metrics.json").read_text())
    cp_json = json.loads((reproduced / "conformal/metrics.json").read_text())
    cal = pd.read_csv(reproduced / "conformal/calibration_scores.csv")

    if len(b) != len(y_test) or len(cal) != len(y_cal):
        raise AssertionError("Reproduced evaluation dimensions differ from snapshot")
    np.testing.assert_array_equal(b["y_true"].to_numpy(), y_test)
    np.testing.assert_array_equal(cal["y_true"].to_numpy(), y_cal)
    np.testing.assert_allclose(b["predicted_probability"].to_numpy(), clean_p, rtol=0, atol=1e-6)
    np.testing.assert_allclose(cal["p_positive"].to_numpy(), cal_p, rtol=0, atol=1e-6)
    cal_scores = np.where(y_cal == 1, 1.0 - cal_p, cal_p)
    np.testing.assert_allclose(cal["score"].to_numpy(), cal_scores, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(b["y_pred"].to_numpy(dtype=int), (clean_p >= 0.5).astype(int))

    base = next(row for row in clean_rows if row["method"] == "baseline")
    for ours, theirs in (("accuracy", "accuracy"),
                         ("false_positive_rate_all_negatives", "false_positive_rate"),
                         ("tn", "tn"), ("fp", "fp"), ("fn", "fn"), ("tp", "tp")):
        np.testing.assert_allclose(base[ours], b_json[theirs], rtol=0, atol=1e-9)

    if set(float(r["alpha"]) for r in cp_json) != set(ALPHAS):
        raise AssertionError("Reproduced conformal alpha values differ")
    for saved in cp_json:
        a = float(saved["alpha"])
        q = thresholds[a]
        recorded_q = float("inf") if saved["threshold_is_infinite"] else float(saved["threshold"])
        np.testing.assert_allclose(q, recorded_q, rtol=0, atol=1e-6)
        row = next(row for row in clean_rows if row["method"] == f"cp_{a:g}")
        cp = saved["conformal"]
        for ours, theirs in (
            ("coverage", "coverage"), ("abstention_rate", "abstention_rate"),
            ("singleton_accuracy", "singleton_accuracy"),
            ("n_decided", "singleton_count"),
            ("n_empty", "empty_count"), ("n_ambiguous", "ambiguous_count"),
            ("false_positive_count", "false_positive_count"),
            ("false_positive_rate_all_negatives", "false_positive_rate_all_negatives"),
        ):
            np.testing.assert_allclose(row[ours], cp[theirs], rtol=0, atol=1e-9)
        saved_predictions = pd.read_csv(
            reproduced / "conformal" / f"predictions_alpha_{a}.csv"
        )
        np.testing.assert_array_equal(saved_predictions["y_true"].to_numpy(), y_test)
        np.testing.assert_allclose(saved_predictions["p_positive"].to_numpy(), clean_p,
                                   rtol=0, atol=1e-6)
        sets = prediction_sets(clean_p, q)
        _, decisions, covered = evaluate(clean_p, y_test, sets, 0.5)
        np.testing.assert_array_equal(saved_predictions["include_0"].to_numpy(dtype=bool), sets[:, 0])
        np.testing.assert_array_equal(saved_predictions["include_1"].to_numpy(dtype=bool), sets[:, 1])
        np.testing.assert_array_equal(saved_predictions["decision"].to_numpy(dtype=int), decisions)
        np.testing.assert_array_equal(saved_predictions["covered"].to_numpy(dtype=bool), covered)

    print("PASS: Clean original_v1 baseline and all reproduced conformal results match.")
    print("Historical comparison (informational; saved historical outputs can differ):")
    historical = json.loads((root / "conformal/metrics.json").read_text())
    comparison = {}
    for saved in historical:
        a = float(saved["alpha"])
        rep = next(item for item in cp_json if float(item["alpha"]) == a)
        item = {
            "historical_threshold": saved["threshold"],
            "reproduced_threshold": rep["threshold"],
            "historical_singleton_count": int(saved["conformal"]["singleton_count"]),
            "reproduced_singleton_count": int(rep["conformal"]["singleton_count"]),
            "historical_coverage": float(saved["conformal"]["coverage"]),
            "reproduced_coverage": float(rep["conformal"]["coverage"]),
        }
        comparison[str(a)] = item
        print(f"  alpha={a:g}: threshold {item['historical_threshold']:.6f} -> "
              f"{item['reproduced_threshold']:.6f}; automatic decisions "
              f"{item['historical_singleton_count']} -> {item['reproduced_singleton_count']}")
    return comparison

def summarize(runs):
    selected_metrics = [
        "accuracy", "false_positive_rate_all_negatives",
        "false_positive_rate_decided_negatives", "coverage", "abstention_rate",
        "singleton_accuracy", "decided_accuracy", "n_decided", "n_abstained",
        "false_positive_count", "n_empty", "n_ambiguous", "mean_set_size",
    ]
    rows = []
    for (cond, sig, method), frame in runs.groupby(
        ["condition", "sigma", "method"], sort=False, dropna=False
    ):
        d = {"condition": cond, "sigma": float(sig), "method": method,
             "n_realizations": len(frame), "n_test_per_realization": int(frame["n_test"].iloc[0])}
        for col in selected_metrics:
            arr = frame[col].to_numpy(dtype=float)
            valid = arr[np.isfinite(arr)]
            d[f"{col}_mean"] = float(valid.mean()) if len(valid) else np.nan
            d[f"{col}_std"] = float(valid.std(ddof=0)) if len(valid) else np.nan
            d[f"{col}_min"] = float(valid.min()) if len(valid) else np.nan
            d[f"{col}_max"] = float(valid.max()) if len(valid) else np.nan
        rows.append(d)
    return pd.DataFrame(rows)


def plot_metric(summary, field, title, filename, output):
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for method in ("baseline", "cp_0.2", "cp_0.1", "cp_0.05"):
        rows = summary[summary["method"] == method].sort_values("sigma")
        x = rows["sigma"].to_numpy()
        y = rows[f"{field}_mean"].to_numpy(dtype=float)
        err = rows[f"{field}_std"].to_numpy(dtype=float)
        mask = np.isfinite(y)
        if mask.any():
            ax.errorbar(x[mask], y[mask], yerr=err[mask], marker="o", capsize=3,
                        label=method.replace("cp_", "Conformal α="))
    ax.set(title=title, xlabel="Added Gaussian noise standard deviation (normalized units)",
           ylabel="Rate")
    ax.set_xticks([s for _, s in CONDITIONS])
    ax.set_ylim(-0.04, 1.04)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / filename, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=10,
                        help="Independent Gaussian realizations per noisy level (default 10)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", default="results/original_v1/stress_noise_reproduced_v1",
                        help="New, nonexistent output directory, relative to repo root")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {output}")

    checkpoint_path = ROOT / "results/baseline/baseline_cnn.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = BaselineCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    sequence_length = int(checkpoint["sequence_length"])
    baseline_threshold = float(checkpoint.get("threshold", 0.5))
    x_cal, y_cal = load_partition("cal", sequence_length)
    x_test, y_test = load_partition("test", sequence_length)
    if len(y_test) != 30 or len(y_cal) != 20:
        raise ValueError("The original experiment expects 30 test and 20 calibration candidates")
    if not ((y_test == 0).any() and (y_test == 1).any()):
        raise ValueError("Test partition must contain both classes")

    p_cal = infer_probabilities(model, x_cal)
    thresholds = {a: calibrate(p_cal, y_cal, a)[0] for a in ALPHAS}
    p_clean = infer_probabilities(model, x_test)
    clean_rows, clean_prediction_rows = get_rows_and_predictions(
        p_clean, y_test, "clean", 0.0, 0, thresholds, baseline_threshold
    )
    historical_comparison = verify_reference(p_clean, y_test, p_cal, y_cal, thresholds, clean_rows)
    all_rows = list(clean_rows)
    all_predictions = [clean_prediction_rows]
    example_noisy_arrays = {}  # Save the first realization for inspectable low/medium/high inputs.

    for trial in range(args.trials):
        # One standard-normal array per realization, scaled at each level. This pairs
        # the disturbances across low/medium/high conditions, reducing comparison noise.
        z = np.random.default_rng(args.seed + trial).standard_normal(x_test.shape).astype(np.float32)
        for condition, sigma in CONDITIONS[1:]:
            noisy = x_test + np.float32(sigma) * z
            if trial == 0:
                example_noisy_arrays[condition] = noisy.copy()
            probabilities = infer_probabilities(model, noisy)
            rows, pred = get_rows_and_predictions(
                probabilities, y_test, condition, sigma, trial,
                thresholds, baseline_threshold,
            )
            all_rows.extend(rows)
            all_predictions.append(pred)
        print(f"Completed noise realization {trial + 1}/{args.trials}", flush=True)

    runs = pd.DataFrame(all_rows)
    prediction_df = pd.concat(all_predictions, ignore_index=True)
    summary = summarize(runs)
    inputs = [
        checkpoint_path,
        *(ROOT / "results/original_v1/dataset_snapshot" / f"{x}_{split}.npy"
          for split in ("cal", "test") for x in ("X", "y")),
        ROOT / "results/baseline/test_predictions.csv",
        ROOT / "results/conformal/metrics.json",
    ]
    inputs.extend([
        ROOT / "results/conformal/calibration_scores.csv",
        ROOT / "results/original_v1/baseline/metrics.json",
        ROOT / "results/original_v1/baseline/test_predictions.csv",
        ROOT / "results/original_v1/conformal/metrics.json",
        ROOT / "results/original_v1/conformal/calibration_scores.csv",
    ])
    manifest = {
        "description": "Gaussian perturbations added AFTER Kepler preprocessing and per-curve normalization",
        "interpretation": "Synthetic input-noise stress; NOT physically calibrated Kepler noise or stellar flares",
        "experiment": "original_v1_reproduced", "n_calibration": len(y_cal), "n_test": len(y_test),
        "historical_comparison": historical_comparison,
        "test_labels": {"0": int(sum(y_test == 0)), "1": int(sum(y_test == 1))},
        "noise_levels": dict(CONDITIONS), "trials_per_noise_level": args.trials,
        "seed_rule": "numpy default_rng(seed + trial), same base noise z scaled across noise levels",
        "seed": args.seed, "baseline_threshold": baseline_threshold,
        "alphas": list(ALPHAS), "conformal_score_thresholds": {str(a): float(q) for a, q in thresholds.items()},
        "calibration": "calibration predictions from clean, unperturbed X_cal; fixed for all test conditions",
        "selection": "all sigma levels predefined; do not tune to test outcomes",
        "repeated_trials": "Same 30 test candidates are reused; repeated noise realizations are not independent test cases",
        "fpr_all_negatives_denominator": f"all {int(sum(y_test == 0))} negatives, including abstained negatives",
        "fpr_decided_negatives_denominator": "only negatively labeled candidates receiving a singleton decision",
        "file_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in inputs},
    }
    output.mkdir(parents=True, exist_ok=False)
    for condition, noisy in example_noisy_arrays.items():
        np.save(output / f"X_test_{condition}_trial0.npy", noisy)
    runs.to_csv(output / "per_run_metrics.csv", index=False)
    prediction_df.to_csv(output / "per_example_predictions.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plot_metric(summary, "decided_accuracy", "Accuracy among automatically decided cases", "decided_accuracy.png", output)
    plot_metric(summary, "false_positive_rate_all_negatives", "False positives / all negative test candidates", "false_positive_rate.png", output)
    plot_metric(summary, "coverage", "Conformal prediction-set coverage", "coverage.png", output)
    plot_metric(summary, "abstention_rate", "Abstention rate", "abstention_rate.png", output)
    print("\nRESULTS (mean across realizations; clean is evaluated once)")
    cols = ["condition", "method", "n_realizations", "accuracy_mean",
            "false_positive_rate_all_negatives_mean", "coverage_mean",
            "abstention_rate_mean", "singleton_accuracy_mean", "n_decided_mean"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved new experiment outputs to: {output}")
    print("No existing original, candidate_split_v1, or grouped_v1 results were modified.")


if __name__ == "__main__":
    main()
