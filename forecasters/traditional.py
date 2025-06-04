from .base import Forecaster

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing


class ARIMAForecaster(Forecaster):
    def __init__(self, order: tuple[int,int,int] = (1,0,0), 
                 enforce_stationarity: bool = True, 
                 enforce_invertibility: bool = True
                 ):
        super().__init__()
        self.order = order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

    def _fit(self, X: np.ndarray, y: np.ndarray):
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

    def _fit(self, X: np.ndarray, y: np.ndarray):
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