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
        self.history = None

    def fit(self, X, y):
        """
        Fit an ARIMA model using a continuous series.
        For univariate data:
            X shape: (n_samples, window_size) and y: (n_samples,)
        For multivariate data:
            X shape: (n_samples, window_size, n_features) and y: (n_samples, n_features)
            We only predict the target_feature (default column 0).
        """
        if X.ndim == 2:
            # Univariate: build history from first window and then append each target.
            self.history = list(X[0])
            self.history.extend(y)
            self.model_fit = ARIMA(self.history, order=self.order).fit()
        elif X.ndim == 3:
            # Multivariate: extract the target_feature from X and y.
            self.history = list(X[0, :, self.target_feature])
            # y is (n_samples, n_features); use the target_feature column.
            self.history.extend(y[:, self.target_feature])
            self.model_fit = ARIMA(self.history, order=self.order).fit()
        else:
            raise ValueError("X must be 2D or 3D.")

    def predict(self, X):
        """
        Predict one-step ahead for each validation window.
        Always returns a 1D array with one predicted value per sample.
        """
        if self.model_fit is None:
            raise ValueError("The ARIMA model must be fitted before prediction.")
        forecasts = []
        # Note: We use the fitted ARIMA model to forecast one step ahead.
        for i in range(X.shape[0]):
            fc = self.model_fit.forecast(steps=1)[0]
            forecasts.append(fc)
        return np.array(forecasts)

class RFModel(BaseModel):
    def __init__(self, target_feature=0):
        self.model = RandomForestRegressor(n_estimators=10, random_state=0)
        self.target_feature = target_feature

    def fit(self, X, y):
        """
        Fit the Random Forest model.
        For univariate data, X shape is (n_samples, window_size) and y is (n_samples,).
        For multivariate, X shape is (n_samples, window_size, n_features) and y is (n_samples, n_features).
        We only use the target_feature from y.
        """
        if X.ndim == 3:
            n_samples, window_size, n_features = X.shape
            # Flatten X.
            X_flat = X.reshape((n_samples, window_size * n_features))
            # Use only the target_feature from y.
            y_target = y[:, self.target_feature]
            self.model.fit(X_flat, y_target)
        elif X.ndim == 2:
            self.model.fit(X, y)
        else:
            raise ValueError("X must be 2D or 3D.")

    def predict(self, X):
        """
        Predicts the next value using the RF model.
        For multivariate data, X is flattened before prediction.
        Returns a 1D array.
        """
        if X.ndim == 3:
            n_samples, window_size, n_features = X.shape
            X_flat = X.reshape((n_samples, window_size * n_features))
            return self.model.predict(X_flat)
        elif X.ndim == 2:
            return self.model.predict(X)
        else:
            raise ValueError("X must be 2D or 3D.")
        
class LinearRegressionModel(BaseModel):
    def __init__(self, target_feature=0):
        self.model = LinearRegression()
        self.target_feature = target_feature

    def fit(self, X, y):
        """
        Fit the linear regression model.
        For univariate data, X is (n_samples, window_size) and y is (n_samples,).
        For multivariate, X is (n_samples, window_size, n_features) and y is (n_samples, n_features).
        In the multivariate case, flatten X and use only the target_feature column from y.
        """
        if X.ndim == 3:
            n_samples, window_size, n_features = X.shape
            X_flat = X.reshape((n_samples, window_size * n_features))
            y_target = y[:, self.target_feature]
            self.model.fit(X_flat, y_target)
        elif X.ndim == 2:
            self.model.fit(X, y)
        else:
            raise ValueError("X must be 2D or 3D.")

    def predict(self, X):
        """
        Predict the next value using the linear regression model.
        For multivariate, flatten X and return a 1D array.
        """
        if X.ndim == 3:
            n_samples, window_size, n_features = X.shape
            X_flat = X.reshape((n_samples, window_size * n_features))
            return self.model.predict(X_flat)
        elif X.ndim == 2:
            return self.model.predict(X)
        else:
            raise ValueError("X must be 2D or 3D.")