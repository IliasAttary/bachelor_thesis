import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics.pairwise import (
    euclidean_distances,
    manhattan_distances,
    cosine_distances,
)

# --- wrappers  ---
class BaseModel:
    def fit(self, X, y):
        """Fit the model to training data."""
        raise NotImplementedError

    def predict(self, X):
        """Make predictions for the input X."""
        raise NotImplementedError

class Forecaster:
    def __init__(
        self,
        model,                  # any BaseModel or torch‐based model
        roc_mode: str = "raw",  # "raw" or "cluster"
        n_clusters: int = None, # required if roc_mode=="cluster"
        distance_metric: str = "euclidean"  # "euclidean", "manhattan", "cosine"
    ):
        self.model = model
        self.roc_mode = roc_mode
        self.distance_metric = distance_metric
        self.raw_windows = []    # list of numpy arrays
        self.clusters = []       # list of lists of numpy windows
        self.centers = None      # numpy array of centroids

        if roc_mode == "cluster":
            if n_clusters is None:
                raise ValueError("n_clusters must be set in cluster mode")
            self.clusterer = KMeans(n_clusters=n_clusters)
        else:
            self.clusterer = None

    def _to_numpy(self, w):
        """Convert torch.Tensor → numpy; leave numpy alone."""
        if isinstance(w, torch.Tensor):
            return w.detach().cpu().numpy()
        return w

    def fit(self, X, y):
        # leave X,y in whatever format your model expects
        self.model.fit(X, y)

    def predict(self, X):
        # pass through to underlying model
        return self.model.predict(X)

    def build_rocs(self, windows: np.ndarray, targets: np.ndarray):
        """
        Perform the offline clustering+assignment step:
        windows: shape (N, window_size, f) numpy
        targets: shape (N,) or (N,f) numpy (only used for error calc)
        """
        # 1) cluster windows
        if self.roc_mode == "cluster":
            self.clusterer.fit(windows)
            self.centers = self.clusterer.cluster_centers_
            # assign each window to its KMeans label
            labels = self.clusterer.labels_
            self.clusters = [
                windows[labels == k].tolist()
                for k in range(self.clusterer.n_clusters)
            ]
        else:
            # raw mode: just keep them all
            self.raw_windows = windows.tolist()

    def update_roc(self, new_window):
        """
        At test time, after you pick which model wins, call this to
        enrich its RoC with the latest observed window.
        """
        w = self._to_numpy(new_window)
        if self.roc_mode == "raw":
            self.raw_windows.append(w)
        else:
            # append then occasionally re-cluster:
            self.raw_windows.append(w)
            if len(self.raw_windows) >= self.clusterer.n_clusters:
                arr = np.stack(self.raw_windows, axis=0)
                self.clusterer.fit(arr)
                self.centers = self.clusterer.cluster_centers_
                labels = self.clusterer.labels_
                self.clusters = [
                    arr[labels == k].tolist()
                    for k in range(self.clusterer.n_clusters)
                ]

    def distance_to_roc(self, window):
        """
        Compute distances from `window` to each RoC center (or to each raw window).
        Returns an array of shape (n_centers,) or (n_raw,).
        """
        w = self._to_numpy(window).reshape(1, -1)
        if self.roc_mode == "cluster":
            M = self.centers.reshape(self.centers.shape[0], -1)
        else:
            M = np.stack(self.raw_windows, axis=0).reshape(len(self.raw_windows), -1)

        if self.distance_metric == "euclidean":
            return euclidean_distances(w, M).ravel()
        elif self.distance_metric == "manhattan":
            return manhattan_distances(w, M).ravel()
        elif self.distance_metric == "cosine":
            return cosine_distances(w, M).ravel()
        else:
            raise ValueError(f"Unknown metric {self.distance_metric}")

# --- Models ---
class ARIMAModel(BaseModel):
    def __init__(self, order=(1, 0, 0), target_feature=0):
        self.order = order
        self.target_feature = target_feature
        self.model_fit = None
        self.window_size = None

    def fit(self, X, y):
        """
        Build one ARIMA on the entire training series.
        We reconstruct the “history” by taking the first window
        plus all subsequent targets.
        """
        # remember window length
        self.window_size = X.shape[1]
        
        # build a 1D series of past + targets
        if X.ndim == 2:
            history = list(X[0])
        else:  # X.ndim == 3
            history = list(X[0, :, self.target_feature])
        history.extend(y if X.ndim == 2 else y[:, self.target_feature])

        # one expensive fit
        self.model_fit = ARIMA(history, order=self.order).fit()

    def predict(self, X):
        """
        For each window in X:
          - append that window’s series onto the fitted model state
            (no refit of parameters),
          - then forecast one step ahead of the *end* of that window.
        Returns a 1D numpy array of length = n_windows.
        """
        if self.model_fit is None:
            raise ValueError("Model must be fitted before calling predict().")
        
        forecasts = []
        for i in range(X.shape[0]):
            # extract just the target series for this window
            if X.ndim == 2:
                series = X[i]
            else:
                series = X[i, :, self.target_feature]
            
            # append new observations to the existing state (refit=False → reuse params)
            # .append returns a new Results object with updated state
            rolled = self.model_fit.append(endog=series, refit=False)
            
            # forecast 1 step ahead from the end of that window
            fc = rolled.forecast(steps=1)[0]
            forecasts.append(fc)

        return np.array(forecasts)

class RFModel(BaseModel):
    def __init__(self, n_estimators=10, random_state=0,
                 target_feature=0, multi_output=False):
        """
        single-target : multi_output=False (default)
          – y can be 1D (n_samples,) or 2D (n_samples, n_features) but we only train/predict target_feature
        multi-output : multi_output=True
          – y must be 2D (n_samples, n_outputs) and we train/predict all outputs
        """
        self.model = RandomForestRegressor(n_estimators=n_estimators,
                                           random_state=random_state)
        self.target_feature = target_feature
        self.multi_output = multi_output

    def _flatten_X(self, X):
        # common flatten logic
        if X.ndim == 3:
            n, w, f = X.shape
            return X.reshape(n, w * f)
        elif X.ndim == 2:
            return X
        else:
            raise ValueError("X must be 2D or 3D.")

    def fit(self, X, y):
        """
        X : (n_samples, window_size) or (n_samples, window_size, n_features)
        y : (n_samples,) or (n_samples, n_features)
        """
        X_flat = self._flatten_X(X)

        # decide which y to pass to sklearn
        if y.ndim == 1:
            # always OK: single-target
            y_train = y
        elif y.ndim == 2:
            if self.multi_output:
                # use all columns of y
                y_train = y
            else:
                # only the specified target_feature column
                y_train = y[:, self.target_feature]
        else:
            raise ValueError("y must be 1D or 2D.")

        self.model.fit(X_flat, y_train)

    def predict(self, X):
        """
        Returns:
          – if multi_output=False: a 1D array (n_samples,)
          – if multi_output=True: a 2D array (n_samples, n_outputs)
        """
        X_flat = self._flatten_X(X)
        preds = self.model.predict(X_flat)

        # sanity‐check shapes
        if not self.multi_output and preds.ndim == 2:
            # if somehow sklearn returned multi‐col output, collapse
            return preds[:, self.target_feature]
        return preds
        
class LinearRegressionModel(BaseModel):
    def __init__(self, target_feature=0, multi_output=False):
        """
        - single‑target (default): multi_output=False  
          y can be 1D or 2D, but we only use y[:, target_feature]  
        - multi‑output: multi_output=True  
          y must be 2D, and we train/predict all columns
        """
        self.model = LinearRegression()
        self.target_feature = target_feature
        self.multi_output = multi_output

    def _flatten_X(self, X):
        # turns (n, w, f) → (n, w*f); leaves (n, w) alone
        if X.ndim == 3:
            n, w, f = X.shape
            return X.reshape(n, w * f)
        elif X.ndim == 2:
            return X
        else:
            raise ValueError("X must be 2D or 3D")

    def fit(self, X, y):
        """
        X: (n_samples, window_size) or (n_samples, window_size, n_features)
        y: (n_samples,) or (n_samples, n_features)
        """
        X_flat = self._flatten_X(X)

        # pick the right y for training
        if y.ndim == 1:
            y_train = y
        elif y.ndim == 2:
            if self.multi_output:
                y_train = y
            else:
                y_train = y[:, self.target_feature]
        else:
            raise ValueError("y must be 1D or 2D")

        self.model.fit(X_flat, y_train)

    def predict(self, X):
        """
        Returns:
          - if multi_output=False: 1D array (n_samples,)
          - if multi_output=True: 2D array (n_samples, n_outputs)
        """
        X_flat = self._flatten_X(X)
        preds = self.model.predict(X_flat)

        # guard: if somehow we got 2D but only wanted one column
        if not self.multi_output and preds.ndim == 2:
            return preds[:, self.target_feature]
        return preds