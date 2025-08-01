import numpy as np
import torch
import torch.nn as nn


class ConvAutoencoder1D(nn.Module):
    def __init__(self, window_size: int, latent_channels: int = 2, dropout_p: float = 0.3):
        super().__init__()
        self.window_size = window_size
        self.latent_channels = latent_channels

        # After one MaxPool1d(2): l1 = window_size//2
        l1 = window_size // 2
        # output_padding to invert the single pool
        op = window_size - 2 * l1

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.MaxPool1d(2),  # downsampling 10 -> 5

            nn.Conv1d(64, latent_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(latent_channels),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, 64, kernel_size=2, stride=2, output_padding=op),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Conv1d(32, 1, kernel_size=3, padding=1)  # output layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def encode(self, x: np.ndarray) -> torch.Tensor:
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(x)}")
        if x.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {x.shape}")

        device = next(self.parameters()).device
        x_tensor = (
            torch.from_numpy(x).float()
                               .unsqueeze(0) # batch dim -> (1, window_size)
                               .unsqueeze(1) # channel dim -> (1, 1, window_size)
                               .to(device)
        )

        with torch.no_grad():
            z = self.encoder(x_tensor) # -> (1, latent_channels, l2)

        return z.squeeze(0)            # -> (latent_channels, l2)
