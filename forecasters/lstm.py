from .base import Forecaster

import numpy as np
import torch
import torch.nn as nn

class LSTMForecaster(Forecaster):
    class _LSTMNet(torch.nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
            super().__init__()
            self.lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.fc = torch.nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, input_size)
            out, _ = self.lstm(x)
            # take the last time‐step’s output
            last = out[:, -1, :]             # (batch, hidden_size)
            return self.fc(last)             # (batch, 1)

    def __init__(
        self,
        window_size: int,
        hidden_size: int = 100,
        num_layers: int = 1,
        dropout: float = 0.0,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 128
    ):
        """
        Args:
            window_size: number of past timesteps per sample
            hidden_size: LSTM hidden dimension
            num_layers: number of LSTM layers
            dropout: dropout between LSTM layers
            lr: learning rate
            epochs: training epochs
            batch_size: training batch size
        """
        super().__init__()
        self.window_size = window_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # build the LSTM+FC model
        self.model = self._LSTMNet(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        # loss & optimizer
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X: np.ndarray, y: np.ndarray):
        # to tensors and reshape for LSTM: (batch, seq_len, input_size=1)
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
        """
        x can be either
          • a 1-D array of length window_size, or
          • a 2-D array of shape (1, window_size)
        This will reshape it to (1, window_size, 1) and return a scalar.
        """
        # ensure numpy float32
        x_np = x.astype(np.float32)

        # if 1-D, make it a batch of size one
        if x_np.ndim == 1:
            x_np = x_np[np.newaxis, :]
        elif x_np.ndim == 2 and x_np.shape[0] == 1:
            # already a single batch
            pass
        else:
            raise ValueError(f"Expected x to be 1-D or shape (1, window), got {x_np.shape}")

        # now x_np is (batch=1, seq_len=window_size)
        x_t = torch.from_numpy(x_np).unsqueeze(-1).to(self.device)
        # → (1, window_size, 1)

        self.model.eval()
        with torch.no_grad():
            out = self.model(x_t)  # (1, 1)
        return float(out.item())