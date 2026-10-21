import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

BASE_DIR = "results/baseline"

# read confusion matrix csv file
confusion_matrix_df = pd.read_csv(f"{BASE_DIR}/confusion_matrix.csv", index_col=0)

# convert to numpy array
confusion_matrix = confusion_matrix_df.to_numpy()

# plot
fig, ax = plt.subplots(figsize=(7, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix,
    display_labels=["Non-transit (0)", "Transit (1)"],
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False,
)

ax.set_title("Baseline CNN — Test Set Confusion Matrix")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")

plt.tight_layout()
plt.savefig("results/baseline/confusion_matrix.png", dpi=150)
plt.show()