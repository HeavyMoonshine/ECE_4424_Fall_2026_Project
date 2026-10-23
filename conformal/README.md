# Conformal prediction

Run from the repository root:

```bash
python -m pip install -r conformal/requirements.txt
python -m conformal.evaluate
```

- `prediction.py`: calibration threshold, prediction sets, and evaluation metrics.
- `evaluate.py`: loads the saved CNN and processed data, then saves results.
- Edit the paths and `ALPHAS` constants at the top of `evaluate.py` if needed.

Required: `results/baseline/baseline_cnn.pt` and `X_cal.npy`, `y_cal.npy`,
`X_test.npy`, `y_test.npy` in `data/processed`. Use the original splits belonging
to this checkpoint. Features have shape `(n, 500)` and labels have shape `(n,)`.
The arrays are ignored by Git and must be supplied locally.

Results go to `results/conformal`: calibration scores, predictions for each alpha,
and `metrics.json`. Label 1 means confirmed planet; label 0 means catalog false
positive. A single included label is a decision; zero or two labels means abstain
(`decision = -1`). Undefined metrics are saved as null.

Calibration uses score `1 - P(true label)` and the
`ceil((n + 1) * (1 - alpha))`-th smallest score. If the rank exceeds the number of
calibration examples, both labels are included. Targets are 95%, 90%, and 80%.

Coverage measures whether the set contains the true label. Singleton accuracy
uses only decided examples. False-positive rates are reported using both all
negative examples and only decided negative examples; compare them alongside
abstention rate.

Coverage guarantees require exchangeability and do not automatically hold under
OOD shift. Existing candidate-level splits may share stars between sets. This
code preserves the existing splits; a star-grouped experiment needs retraining
and recalibration. Small calibration/test sets also limit conclusions.

Run the calculation tests with:

```bash
python -m unittest discover -s tests -v
```
