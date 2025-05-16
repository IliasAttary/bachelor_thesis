import numpy as np
import torch
import torch.nn as nn

class ConvAutoencoder1D(nn.Module):
    def __init__(self, window_size: int, latent_channels: int = 8, dropout_p: float = 0.05):
        super().__init__()
        self.window_size = window_size
        self.latent_channels = latent_channels

        # Downsampled lengths after two MaxPool1d(2) layers:
        l1 = window_size // 2
        l2 = l1 // 2

        # Compute output_padding for exact inversion in decoder:
        op1 = l1 - 2 * l2   # to go from l2 -> l1
        op2 = window_size - 2 * l1  # to go from l1 -> window_size

        self.encoder = nn.Sequential(
            nn.Conv1d(1,  32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.MaxPool1d(2), # -> l1

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.MaxPool1d(2), # -> l2

            nn.Conv1d(64, latent_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(latent_channels),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, 64, kernel_size=2, stride=2, output_padding=op1), # -> l1
            nn.InstanceNorm1d(64, affine=True),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2, output_padding=op2 ),
            nn.InstanceNorm1d(32, affine=True),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Conv1d(32, 1, kernel_size=3, padding=1)
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
