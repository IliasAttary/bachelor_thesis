import numpy as np
import torch

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR


class Forecaster:
    def __init__(self):
        self.roc =  []
        self.centers = []

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
    
    def compute_kmeans_centers(self):
        pass

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
        n_estimators: int = 10,
        max_depth: int = None,
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
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, epsilon: float = 0.1):
        super().__init__()
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, x: np.ndarray):
        x_flat = x.reshape(1, -1)
        return float(self.model.predict(x_flat)[0])

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

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, x: np.ndarray):
        x_flat = x.reshape(1, -1)
        return float(self.model.predict(x_flat)[0])