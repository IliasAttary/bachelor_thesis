import os
import torch
import numpy as np
import pandas as pd

class Births:
    def __init__(self, path=None):
        # Determine file path
        if path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(base, 'us_births_dataset.csv')

        # Read CSV, parse 'Date' as datetime index
        self.df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        # Save raw copy before normalization
        self.raw = self.df

        # Create 1‑D numeric array
        series = self.df["Births"].astype(float)
        self.index = series.index
        self.data = series.to_numpy(copy=True)

    def inverse_transform(self, arr):
        if not self.normalize:
            return arr
        else:
            return arr * self.std + self.mean