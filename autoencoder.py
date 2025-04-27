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

    def encode(self, x):
        """
        Encode a single window or batch to embeddings.

        Accepts:
          - x: 1D numpy array of shape (L,)
          - or 2D tensor of shape (1, L) or (L, 1)
          - or 3D tensor of shape (N, C, L)
        Returns:
          - embedding: Tensor of shape (C', L') with no batch dim
        """
        # Convert numpy to torch and add channel
        if isinstance(x, np.ndarray):
            x = asTorch(x)  # -> shape (1,1,L)

        # If 2D tensor, add batch and channel dims
        if x.dim() == 2:
            # assume (L,1) or (1,L)
            if x.size(1) == self.window_size:
                # shape (1,L)
                x = x.unsqueeze(1)  # -> (1,1,L)
            else:
                x = x.unsqueeze(0)  # -> (?,L,1)
                x = x.permute(0,2,1)

        # Ensure tensor on correct device
        device = next(self.parameters()).device
        x = x.to(device)

        # Encode without grad
        with torch.no_grad():
            z = self.encoder(x)  # -> (N, C', L')

        # Remove batch dim if present
        if z.size(0) == 1:
            z = z.squeeze(0)

        return z