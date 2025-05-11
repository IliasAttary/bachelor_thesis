import numpy as np
import torch
import torch.nn as nn

class ConvAutoencoder1D(nn.Module):
    """
    1D Convolutional Autoencoder with a reduced-channel bottleneck (Option 1).

    The encoder reduces channels at the final layer to `latent_channels`, enforcing a
    true bottleneck of size `latent_channels * (window_size // 4)`.
    """
    def __init__(self, window_size: int, latent_channels: int = 8):
        super().__init__()
        self.window_size = window_size
        self.latent_channels = latent_channels

        # Downsampled lengths after two MaxPool1d(2) layers:
        l1 = window_size // 2
        l2 = l1 // 2

        # Compute output_padding for exact inversion in decoder:
        op1 = l1 - 2 * l2   # to go from l2 -> l1
        op2 = window_size - 2 * l1  # to go from l1 -> window_size

        # Encoder: window_size -> l1 -> l2, ending in `latent_channels` features
        self.encoder = nn.Sequential(
            nn.Conv1d(1,  32, kernel_size=3, padding=1),  # -> window_size
            nn.ReLU(),
            nn.MaxPool1d(2),                                # -> l1

            nn.Conv1d(32, 64, kernel_size=3, padding=1),    # -> l1
            nn.ReLU(),
            nn.MaxPool1d(2),                                # -> l2

            # bottleneck channel reduction
            nn.Conv1d(64, latent_channels, kernel_size=3, padding=1),  # -> l2
            nn.ReLU()
        )

        # Decoder: l2 -> l1 -> window_size
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                latent_channels, 64,
                kernel_size=2, stride=2,
                output_padding=op1
            ),                                               # -> l1
            nn.ReLU(),

            nn.ConvTranspose1d(
                64, 32,
                kernel_size=2, stride=2,
                output_padding=op2
            ),                                               # -> window_size
            nn.ReLU(),

            # final conv back to single channel
            nn.Conv1d(32, 1, kernel_size=3, padding=1)      # -> window_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full autoencoder forward: input x -> reconstruction.
        Args:
            x: Tensor of shape (B, 1, window_size)
        Returns:
            recon: Tensor of shape (B, 1, window_size)
        """
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def encode(self, x: np.ndarray) -> torch.Tensor:
        """
        Encode a single univariate window into its latent representation.

        Args:
            x: 1D numpy array of shape (window_size,)
        Returns:
            Tensor of shape (latent_channels, l2)
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(x)}")
        if x.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {x.shape}")

        device = next(self.parameters()).device
        x_tensor = (
            torch.from_numpy(x).float()
                               .unsqueeze(0)   # batch dim -> (1, window_size)
                               .unsqueeze(1)   # channel dim -> (1, 1, window_size)
                               .to(device)
        )

        with torch.no_grad():
            z = self.encoder(x_tensor)  # -> (1, latent_channels, l2)

        return z.squeeze(0)  # -> (latent_channels, l2)
