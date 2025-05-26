from .base import Forecaster

import numpy as np
from sklearn.neural_network import MLPRegressor


class MLPForecaster(Forecaster):
    def __init__(
        self,
        hidden_layer_sizes: tuple = (64, 32),
        activation: str = 'relu',
        solver: str = 'adam',
        learning_rate_init: float = 1e-3,
        max_iter: int = 400,
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