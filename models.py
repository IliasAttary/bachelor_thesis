import numpy as np

import torch
import torch.nn as nn

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from kneed import KneeLocator

class Forecaster:
    def __init__(self):
        self.rocs = {"raw": [], "latent": []}
        self.centers = []

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
    
    def compute_kmeans_centers(self, k_min=2, k_max=10, random_state=0):
        # Extract latent windows
        latent_windows = self.rocs.get("latent", [])
        if not latent_windows:
            raise ValueError("No latent windows available for clustering.")

        # Determine integer bounds for k
        k_min_int = max(2, int(k_min))
        k_max_int = max(k_min_int, int(k_max))

        # Stack flattened latent windows into numpy array X
        X = np.vstack([
            (w.detach().cpu().numpy() if isinstance(w, torch.Tensor) else np.array(w))
            .flatten() for w in latent_windows
        ])

        # Compute inertia for each k
        ks = list(range(k_min_int, k_max_int + 1))
        inertias = []
        for k in ks:
            km = KMeans(n_clusters=k, random_state=random_state)
            km.fit(X)
            inertias.append(km.inertia_)

        # Use KneeLocator to pick elbow k (fallback to k_min_int)
        kl = KneeLocator(ks, inertias, curve='convex', direction='decreasing')
        best_k = kl.elbow or k_min_int

        # Final k-means on X
        km_final = KMeans(n_clusters=best_k, random_state=random_state).fit(X)
        centers_flat = km_final.cluster_centers_  # shape (best_k, D)

        # Convert each center back to latent shape and to Tensor
        orig_shape = latent_windows[0].shape  # (C, L)
        centers = []
        for c in centers_flat:
            arr = c.reshape(orig_shape)
            centers.append(torch.tensor(arr, dtype=torch.float32))

        # Preserve rocs; set new centers
        self.centers = centers
        return best_k

# ---- Traditional Model ----

class ARIMAForecaster(Forecaster):
    def __init__(self, order: tuple[int,int,int] = (1,0,0), enforce_stationarity: bool = True, enforce_invertibility: bool = True):
        super().__init__()
        self.order = order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

    def fit(self, X: np.ndarray, y: np.ndarray):
        # No global training—ARIMA will be fit on each window at predict time.
        return self

    def predict(self, x: np.ndarray) -> float:
        # ensure a 1-D array of past values
        x1d = x.ravel().astype(np.float64)

        # fit ARIMA on this window
        model = ARIMA(
            endog=x1d,
            order=self.order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility
        )
        fitted = model.fit()
        return float(fitted.forecast(steps=1)[0])

class ExpSmoothingForecaster(Forecaster):
    def __init__(
        self,
        trend: str = None,
        seasonal: str = None,
        seasonal_periods: int = None,
        smoothing_level: float = None,
        smoothing_slope: float = None,
        smoothing_seasonal: float = None,
        optimized: bool = True
    ):
        """
        Args:
            trend: 'add', 'mul', or None
            seasonal: 'add', 'mul', or None
            seasonal_periods: number of periods in a season (required if seasonal is not None)
            smoothing_level: alpha (0<alpha<1)
            smoothing_slope: beta for trend
            smoothing_seasonal: gamma for seasonal
            optimized: whether to auto-optimize parameters
        """
        super().__init__()
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.smoothing_level = smoothing_level
        self.smoothing_slope = smoothing_slope
        self.smoothing_seasonal = smoothing_seasonal
        self.optimized = optimized

    def fit(self, X: np.ndarray, y: np.ndarray):
        # No global training
        return self

    def predict(self, x: np.ndarray) -> float:
        arr = x.ravel().astype(np.float64)

        if self.trend is None and self.seasonal is None:
            # simple exponential smoothing
            model = SimpleExpSmoothing(arr)
            fitted = model.fit(
                smoothing_level=self.smoothing_level,
                optimized=self.optimized
            )
        else:
            # Holt-Winters exponential smoothing
            model = ExponentialSmoothing(
                arr,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods
            )
            fitted = model.fit(
                smoothing_level=self.smoothing_level,
                smoothing_slope=self.smoothing_slope,
                smoothing_seasonal=self.smoothing_seasonal,
                optimized=self.optimized
            )

        return float(fitted.forecast(1)[0])

# ---- Regression Models ----

class LinearRegressionForecaster(Forecaster):
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, x):
        pred = self.model.predict(x.reshape(1, -1))
        return float(pred[0])

class RandomForestForecaster(Forecaster):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = None
    ):
        super().__init__()
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, x):
        pred = self.model.predict(x.reshape(1, -1))
        return float(pred[0])
    
class SVRForecaster(Forecaster):
    def __init__(self, kernel: str = 'linear', C: float = 10, epsilon: float = 0.1):
        super().__init__()
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, x):
        pred = self.model.predict(x.reshape(1, -1))
        return float(pred[0])

class GradientBoostingForecaster(Forecaster):
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int = 0
    ):
        super().__init__()
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, x):
        pred = self.model.predict(x.reshape(1, -1))
        return float(pred[0])
    
class DecisionTreeForecaster(Forecaster):
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        random_state: int = None
    ):
        """
        Args:
            max_depth: maximum depth of the tree
            min_samples_split: minimum samples required to split an internal node
            min_samples_leaf: minimum samples required to be at a leaf node
            random_state: seed for reproducibility
        """
        super().__init__()
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, x):
        pred = self.model.predict(x.reshape(1, -1))
        return float(pred[0])

# ---- Neural network-based Models ----

class MLPForecaster(Forecaster):
    def __init__(
        self,
        hidden_layer_sizes: tuple = (64, 32),
        activation: str = 'relu',
        solver: str = 'adam',
        learning_rate_init: float = 1e-3,
        max_iter: int = 200,
        early_stopping: bool = True,
        n_iter_no_change: int = 10,
        random_state: int = None
    ):
        super().__init__()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            n_iter_no_change=n_iter_no_change,
            random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, x: np.ndarray) -> float:
        x_flat = x.reshape(1, -1)
        return float(self.model.predict(x_flat)[0])

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
        window_size: int,
        hidden_size: int = 100,
        num_layers: int = 1,
        dropout: float = 0.0,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 128,
    ):
        """
        Args:
            window_size: number of time steps per sample
            hidden_size: size of LSTM hidden state (per direction)
            num_layers: number of stacked LSTM layers
            dropout: dropout between LSTM layers
            lr: learning rate
            epochs: training epochs
            batch_size: training batch size
        """
        super().__init__()
        self.window_size = window_size
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

    def fit(self, X: np.ndarray, y: np.ndarray):
        # build tensors of shape (batch, seq_len, input_size=1)
        X_t = torch.from_numpy(X.astype(np.float32)).unsqueeze(-1).to(self.device)
        y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(-1).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
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
        window_size: int,
        conv_channels: tuple[int, ...] = (32, 64),
        kernel_size: int = 3,
        lstm_hidden_size: int = 100,
        lstm_num_layers: int = 1,
        dropout: float = 0.0,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 128
    ):
        super().__init__()
        self.window_size = window_size
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