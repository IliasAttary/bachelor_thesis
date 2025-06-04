from .base import Forecaster

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


class LinearRegressionForecaster(Forecaster):
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()

    def _fit(self, X, y):
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

    def _fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, x):
        pred = self.model.predict(x.reshape(1, -1))
        return float(pred[0])
    
class SVRForecaster(Forecaster):
    def __init__(self, kernel: str = 'linear', C: float = 10, epsilon: float = 0.1):
        super().__init__()
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)

    def _fit(self, X, y):
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
        random_state: int = None
    ):
        super().__init__()
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )

    def _fit(self, X, y):
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

    def _fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, x):
        pred = self.model.predict(x.reshape(1, -1))
        return float(pred[0])