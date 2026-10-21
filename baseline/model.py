import torch.nn as nn

# ------- DEFINE THE BASELINE CNN MODEL -------

class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # First layer goes from input 1 channel -> 16 features
            # kernel_size 7 - windows of 7 samples viewed
            # padding 3 - pads with 3 0s on each side to prevent length shrinking
            # ReLU applies max(0, x)
            # MaxPool1d downsamples channels to keep strongest activation
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Second layer goes from 16 features -> 32 features
            # kernel_size 5 - windows of 5 samples viewed
            # padding 2 - only 2 0s of padding are needed on second pass
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Third layer goes from 32 features -> 64 features
            # kernal_size 3 - windows of 3 samples viewed
            # padding 1 - only 1 0 of padding is needed for third pass
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            # resize to length 1 and flaten into single vector
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(-1)