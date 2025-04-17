import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

class BaseModel:
    def fit(self, X, y):
        """Fit the model to training data."""
        raise NotImplementedError

    def predict(self, X):
        """Make predictions for the input X."""
        raise NotImplementedError

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