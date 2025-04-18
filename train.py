import numpy as np
import joblib
from data import airline
from utils import *
from models import *


def _make_forecasters(roc_mode, n_clusters, metric):
    return [
        Forecaster(
            model=ARIMAModel(order=(2,1,0)),
            roc_mode=roc_mode,
            n_clusters=n_clusters,
            distance_metric=metric
        ),
        Forecaster(
            model=RFModel(n_estimators=100),
            roc_mode=roc_mode,
            n_clusters=n_clusters,
            distance_metric=metric
        ),
        Forecaster(
            model=LinearRegressionModel(),
            roc_mode=roc_mode,
            n_clusters=n_clusters,
            distance_metric=metric
        ),
    ]

def train(
    dataset,
    window_size: int = 12,
    roc_mode: str = "raw",
    n_clusters: int = 10,
    metric: str = "euclidean",
    save_to: str = None
):
    """
    Train a suite of forecasters on the first 50% of `dataset`, build RoCs on the next 25%.

    Args:
      dataset:  path to CSV (Airline loader) or 1D numpy array of the full series
      window_size:  int, #lags per input window
      roc_mode:  "raw" | "cluster"
      n_clusters: int, only used if roc_mode=="cluster"
      metric:   "euclidean" | "manhattan" | "cosine"
      save_to:  if given, a dir where we dump forecasters.pkl

    Returns:
      List[Forecaster] — all your trained wrappers, with `.raw_windows` or `.centers` populated.
    """
    # --- load & split ---
    if isinstance(dataset, str):
        loader = airline.Airline(path=None, normalize=True, as_numpy=True)
        data = loader.data
    else:
        data = dataset

    X_train, y_train, X_val, y_val, _, _ = windowing(
        data,
        train_window_size=window_size,
        horizon=1,
        train_ratio=0.5,
        val_ratio=0.25,
    )

    # --- instantiate & fit ---
    forecasters = _make_forecasters(roc_mode, n_clusters, metric)
    for f in forecasters:
        f.fit(X_train, y_train)
        if roc_mode == "cluster":
            f.build_rocs(X_val, y_val)
        else:
            # raw mode: just stash all validation windows
            f.raw_windows = [w for w in X_val]

    # --- optional save ---
    if save_to:
        joblib.dump(forecasters, f"{save_to}/forecasters.pkl")

    return forecasters