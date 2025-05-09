import numpy as np
import torch
import torch.nn as nn
from utils import asTorch

class ConvAutoencoder1D(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

        # Compute the two down‐sampled lengths:
        l1 = window_size // 2       # after first MaxPool1d(2)
        l2 = l1 // 2                # after second MaxPool1d(2)

        # Figure out how much padding to add so that each ConvTranspose1d
        # exactly inverts its corresponding MaxPool1d:
        #   convtranspose output = (in_len − 1)*2 + 2 + output_padding
        op1 = l1 - 2 * l2           # to go l2 → l1
        op2 = window_size - 2 * l1  # to go l1 → window_size

        # Encoder: window_size → l1 → l2
        self.encoder = nn.Sequential(
            nn.Conv1d(1,   32, kernel_size=3, padding=1),  # length stays window_size
            nn.ReLU(),
            nn.MaxPool1d(2),                                 # → l1

            nn.Conv1d(32,  64, kernel_size=3, padding=1),  # stays l1
            nn.ReLU(),
            nn.MaxPool1d(2),                                 # → l2

            nn.Conv1d(64, 128, kernel_size=3, padding=1),  # stays l2
            nn.ReLU()
        )

        # Decoder: l2 → l1 → window_size
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                128, 64,
                kernel_size=2, stride=2,
                output_padding=op1
            ),                                               # → l1
            nn.ReLU(),

            nn.ConvTranspose1d(
                64,  32,
                kernel_size=2, stride=2,
                output_padding=op2
            ),                                               # → window_size
            nn.ReLU(),

            # final smoothing conv to map back to 1 channel
            nn.Conv1d(32, 1, kernel_size=3, padding=1)      # stays window_size
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: np.ndarray) -> torch.Tensor:
        """
        Encode a single univariate window.

        Args:
          x: 1D numpy array of shape (window_size,)
        Returns:
          Tensor of shape (C, L') with no batch dim.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(x)}")

        if x.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {x.shape}")

        # 1) numpy → torch tensor, add batch & channel dims → (1,1,window_size)
        device = next(self.parameters()).device
        x_tensor = (
            torch.from_numpy(x)
                 .float()
                 .unsqueeze(0)   # batch dim → (1, window_size)
                 .unsqueeze(1)   # channel dim → (1, 1, window_size)
                 .to(device)
        )

        # 2) forward through encoder without grad
        with torch.no_grad():
            z = self.encoder(x_tensor)  # → (1, C, L')

        # 3) drop batch dim → (C, L')
        z = z.squeeze(0)

        return z