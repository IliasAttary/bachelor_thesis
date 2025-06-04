from .base import Forecaster

import numpy as np
import torch


class BiLSTMForecaster(Forecaster):
    class _BiLSTMNet(torch.nn.Module):
        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
        ):
            super().__init__()
            self.lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True,
            )
            # since bidirectional, hidden outputs are hidden_size * 2
            self.fc = torch.nn.Linear(hidden_size * 2, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, input_size)
            out, _ = self.lstm(x)               # → (batch, seq_len, hidden_size * 2)
            last = out[:, -1, :]               # → (batch, hidden_size * 2)
            return self.fc(last)               # → (batch, 1)

    def __init__(
        self,
        hidden_size: int = 100,
        num_layers: int = 1,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 128,
    ):
        """
        Args:
            hidden_size: size of LSTM hidden state (per direction)
            num_layers: number of stacked LSTM layers
            dropout: dropout between LSTM layers
            lr: learning rate
            epochs: training epochs
            batch_size: training batch size
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._BiLSTMNet(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X: np.ndarray, y: np.ndarray, generator=None):
        # build tensors of shape (batch, seq_len, input_size=1)
        X_t = torch.from_numpy(X.astype(np.float32)).unsqueeze(-1).to(self.device)
        y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(-1).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            generator=generator
        )

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
        x: 1D array of length window_size, or 2D shape (1, window_size)
        returns: scalar next-step forecast
        """
        x_np = x.astype(np.float32)
        if x_np.ndim == 1:
            x_np = x_np[np.newaxis, :]
        elif not (x_np.ndim == 2 and x_np.shape[0] == 1):
            raise ValueError(f"Expected 1D or (1, window), got {x_np.shape}")

        x_t = torch.from_numpy(x_np).unsqueeze(-1).to(self.device)  # (1, window, 1)
        self.model.eval()
        with torch.no_grad():
            out = self.model(x_t)  # (1, 1)
        return float(out.item())