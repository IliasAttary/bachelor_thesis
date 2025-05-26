from .base import Forecaster

import numpy as np
import torch
import torch.nn as nn


class CNNLSTMForecaster(Forecaster):
    class _CNNLSTMNet(nn.Module):
        def __init__(
            self,
            input_size: int,
            conv_channels: tuple[int, ...],
            kernel_size: int,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            dropout: float
        ):
            super().__init__()
            # build conv1d feature extractor
            conv_layers = []
            in_ch = input_size
            for out_ch in conv_channels:
                conv_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2))
                conv_layers.append(nn.ReLU())
                if dropout > 0:
                    conv_layers.append(nn.Dropout(dropout))
                in_ch = out_ch
            self.conv = nn.Sequential(*conv_layers)

            # LSTM on conv outputs
            self.lstm = nn.LSTM(
                input_size=conv_channels[-1],
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers,
                batch_first=True,
                dropout=dropout if lstm_num_layers > 1 else 0.0
            )
            self.fc = nn.Linear(lstm_hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, input_size=1)
            x = x.permute(0, 2, 1)       # → (batch, channels=1, seq_len)
            x = self.conv(x)             # → (batch, conv_channels[-1], seq_len)
            x = x.permute(0, 2, 1)       # → (batch, seq_len, conv_channels[-1])
            out, _ = self.lstm(x)        # → (batch, seq_len, hidden)
            last = out[:, -1, :]         # → (batch, hidden)
            return self.fc(last)         # → (batch, 1)

    def __init__(
        self,
        conv_channels: tuple[int, ...] = (32, 64),
        kernel_size: int = 3,
        lstm_hidden_size: int = 100,
        lstm_num_layers: int = 1,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 128
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self._CNNLSTMNet(
            input_size=1,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            dropout=dropout
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X: np.ndarray, y: np.ndarray):
        # X: (n_samples, window_size), y: (n_samples,)
        X_t = torch.from_numpy(X.astype(np.float32)).unsqueeze(-1).to(self.device)
        y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(-1).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            avg_loss = epoch_loss / len(dataset)
            if _ == 0 or (_ + 1) % 15 == 0 or (_ + 1) == self.epochs:
                print(f"{_+1}/{self.epochs} {avg_loss:.5f}", end=" | ")                
        self.model.eval()

    def predict(self, x: np.ndarray) -> float:
        # Accepts 1D array (window_size,) or 2D array (1, window_size)
        x_np = x.astype(np.float32)
        if x_np.ndim == 1:
            x_np = x_np[np.newaxis, :]
        elif not (x_np.ndim == 2 and x_np.shape[0] == 1):
            raise ValueError(f"Expected 1D or shape (1, window), got {x_np.shape}")

        x_t = torch.from_numpy(x_np).unsqueeze(-1).to(self.device)  # → (1, window_size, 1)
        self.model.eval()
        with torch.no_grad():
            out = self.model(x_t)  # → (1, 1)
        return float(out.item())