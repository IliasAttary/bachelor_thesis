import os
import time
import random
import warnings
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import torch

from forecasters import *
from utils import *
from data import *

# ---------------------------
# Setup
# ---------------------------
warnings.filterwarnings("ignore")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

g = torch.Generator()
g.manual_seed(seed)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# Datasets (name, class, length, window_size)
# ---------------------------
ts_configs = [
    ("Temperatures",  Temperatures,  None, 10),  # kaggle (datamarket.com)
    ("Births",        Births,        None, 10),  # Monash
    ("SolarPower",    Solar,         None, 10),  # Monash
    ("WindPower",     Wind,          None, 10),  # Monash
    ("ExchangeRate",  ExchangeRate,  None, 10),  # ECB
    ("Treasury",      Treasury,      None, 10),  # FRED
    ("Air Quality",   Air,           None, 10),  # UCI
    ("OfficeTemp",    OfficeTemp,    None, 10),  # NAB
]

# ---------------------------
# Baseline model factory
# ---------------------------
def make_baselines():
    """
    Return dict {display_name: model_instance} with unified .fit/.predict API.
    - ARIMA: ARIMAForecaster
    - ETS:   ExpSmoothingForecaster
    - LSTM:  LSTMForecaster (single reasonable config)
    """
    return {
        "ARIMA": ARIMAForecaster(order=(1, 0, 0)),
        "ETS":   ExpSmoothingForecaster(),
        "LSTM":  LSTMForecaster(
                    hidden_size=16,
                    num_layers=1,
                    dropout=0.0,
                    lr=1e-3,
                    epochs=30,
                    batch_size=64
                ),
    }

# ---------------------------
# Train/Eval loop
# ---------------------------
rows = []

for ds_name, DS, length, window_size in ts_configs:
    print(f"\n→ Dataset: {ds_name} (window_size={window_size})")

    # Load & window (assumes your windowing() does z-normalization using train stats)
    ts = DS(length=length)
    X_train, y_train, X_val, y_val, X_test, y_test = windowing(ts.data, window_size)

    baselines = make_baselines()

    for model_name, model in baselines.items():
        print(f"    • Training {model_name} … ", end="", flush=True)
        t0 = time.perf_counter()
        model.fit(X_train, y_train, generator=g)
        fit_time = time.perf_counter() - t0
        print(f"done in {fit_time:.2f}s")

        print(f"      Predicting {model_name} … ", end="", flush=True)
        t1 = time.perf_counter()
        preds = [model.predict(w) for w in X_test]
        pred_time = time.perf_counter() - t1
        print(f"done in {pred_time:.2f}s")

        mse = mean_squared_error(y_test, preds)

        rows.append({
            "dataset": ds_name,
            "model": model_name,
            "window_size": window_size,
            "n_test_windows": len(X_test),
            "test_mse": float(mse),
            "fit_time_s": float(fit_time),
            "pred_time_s": float(pred_time),
            "total_time_s": float(fit_time + pred_time),
        })

# ---------------------------
# Save results
# ---------------------------
df = pd.DataFrame(rows).sort_values(["dataset", "test_mse"])
out_path = os.path.join(RESULTS_DIR, "baseline.csv")
df.to_csv(out_path, index=False)

print("\nSaved:", out_path)
print(df.to_string(index=False))
