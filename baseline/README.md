# Baseline CNN

## Running the Baseline

1. Run all data pipeline code, and ensure you have all process data in `data/processed` directory:
    - `X_train.npy`
    - `X_test.npy`
    - `X_val.npy`
    - `X_cal.npy`
    - `y_train.npy`
    - `y_test.npy`
    - `y_val.npy`
    - `y_cal.npy`
2. *(Recommended)* Create a virtual environment and activate it
3. From the root directory, run 
```bash
python baseline/train.py
```
4. From the root directory, run
```bash
python baseline/evaluate.py
```
5. If desired, create visual plots. From the root directory, run
```bash
python baseline/plot_training_loss.py
python baseline/plot_confusion_matrix.py
```

## File breakdown

### `train.py`

- Houses training loop
- Runs 250 epochs with batch size 32 (configurable through constants at top of file)
- Disables gradient calculation/usage
- Writes model to `results/baseline/baseline_cnn.pt`
- Writes results to `results/baseline/training_history.csv`

### `model.py`

- Houses the actual nueral network and layers
- Uses 3 layers:  
    1. 1 -> 16 features
    2. 16 -> 32 features
    3. 32 -> 64 features
- Each layer goes through ReLU and padding

### `evaluate.py`

- Computes:  
    - accuracy
    - precision
    - recall
    - f1 score
    - confusion matrix
    - ROC-AUC
- Writes prediction results to `results/baseline/test_predictions.csv`
- Writes classification results to `results/baseline/classification_report.txt`
- Writes metrics results to `results/baseline/metrics.json`
- Writes confusion matrix to `results/baseline/confusion_matrix.csv`

### `plot_training_loss.py`

- Plots training loss and validation loss to visualize where the model starts to overfit
- Saves plot to `results/baseline/loss_curves.png`

### `plot_confusion_matrix.py`

- Plots confusion matrix to visualize misclassifications
- Saves plot to `results/baseline/confuion_matrix.png`