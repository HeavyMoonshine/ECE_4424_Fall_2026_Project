import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd

import os
import random
import copy

from model import BaselineCNN

# ------- set seeds for reproducability -------

SEED = 42
BATCH_SIZE = 32
EPOCHS = 250
LEARNING_RATE = 1e-3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# ------- paths to datasets/outputs -------

DATA_DIR = "data/processed"
OUTPUT_DIR = "results/grouped_v1/baseline"
os.makedirs(OUTPUT_DIR, exist_ok=True)

X_TRAIN_PATH = f"{DATA_DIR}/X_train.npy"
Y_TRAIN_PATH = f"{DATA_DIR}/y_train.npy"

X_VAL_PATH = f"{DATA_DIR}/X_val.npy"
Y_VAL_PATH = f"{DATA_DIR}/y_val.npy"

X_CAL_PATH = f"{DATA_DIR}/X_cal.npy"
Y_CAL_PATH = f"{DATA_DIR}/y_cal.npy"

X_TEST_PATH = f"{DATA_DIR}/X_test.npy"
Y_TEST_PATH = f"{DATA_DIR}/y_test.npy"

# ------- extract data into np arrays -------

X_train = np.load(X_TRAIN_PATH)
y_train = np.load(Y_TRAIN_PATH)

X_val = np.load(X_VAL_PATH)
y_val = np.load(Y_VAL_PATH)

# add data validation here if needed

# ------- Convert np arrays to torch tensors/loaders -------

def make_dataset(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)

def make_loader(X, y, shuffle):
    dataset = make_dataset(X, y)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, True)
val_loader = make_loader(X_val, y_val, False)

# ------- Create the model (defined in model.py) -------

model = BaselineCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ------- TRAIN AND VALIDATE -------

# run a training epoch
def run_epoch(loader, training):
    model.train(training)
    total_loss = 0.0
    total_examples = 0

    with torch.set_grad_enabled(training):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if training:
                optimizer.zero_grad()       # only compute gradients in validation

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if training:
                loss.backward()
                optimizer.step()

            n = len(y_batch)
            total_loss += loss.item() * n
            total_examples += n

    return total_loss / total_examples

best_val_loss = float("inf")
best_state = None
history = []

for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(train_loader, training=True)
    val_loss = run_epoch(val_loader, training=False)

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"training loss={train_loss:.4f} | validation loss={val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())

# ------- OUTPUT BEST VALIDATION CHECKPOINT TO FILES -------

model.load_state_dict(best_state)

# save as pt file
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "best_val_loss": best_val_loss,
        "seed": SEED,
        "sequence_length": 500,
        "threshold": 0.5,
    },
    f"{OUTPUT_DIR}/baseline_cnn.pt",
)

# save as csv
pd.DataFrame(history).to_csv(
    f"{OUTPUT_DIR}/training_history.csv",
    index=False,
)
