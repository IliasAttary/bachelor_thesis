import torch
import torch.nn as nn

class ConvAutoencoder1D(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: 30 → 15 → 7
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),  # 30 → 30
            nn.ReLU(),
            nn.MaxPool1d(2),                             # 30 → 15
            nn.Conv1d(32, 64, kernel_size=3, padding=1), # 15 → 15
            nn.ReLU(),
            nn.MaxPool1d(2),                             # 15 → 7
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Decoder: 7 → 15 → 30
        self.decoder = nn.Sequential(
            # from 7 → (7−1)*2 + 2 + 1 = 15
            nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2, output_padding=1),
            nn.ReLU(),
            # from 15 → (15−1)*2 + 2 + 0 = 30
            nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            # restore channel dim, keeps length = 30
            nn.Conv1d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)