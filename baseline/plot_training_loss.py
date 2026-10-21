import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "results/baseline"

# load training data
history = pd.read_csv(f"{BASE_DIR}/training_history.csv")

# Find the epoch with the lowest validation loss
best_idx = history["val_loss"].idxmin()
best_epoch = int(history.loc[best_idx, "epoch"])
best_val_loss = history.loc[best_idx, "val_loss"]

plt.figure(figsize=(10,6))

# plot training loss
plt.plot(
    history["epoch"],
    history["train_loss"],
    label="Training Loss",
    marker="o",
    markersize=3,
)

# plot validation loss
plt.plot(
    history["epoch"],
    history["val_loss"],
    label="Validation Loss",
    marker="o",
    markersize=3,
)

# Mark the best validation-loss epoch
plt.axvline(
    best_epoch,
    linestyle="--",
    label=f"Lowest validation loss (epoch {best_epoch})",
)

plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("CNN Training vs. Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/baseline/loss_curves.png", dpi=150)
plt.show()