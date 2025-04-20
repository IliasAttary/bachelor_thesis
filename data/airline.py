import os
import torch
import numpy as np
import pandas as pd

class Airline:
    def __init__(self, path=None, normalize=True):
        # Determine file path
        if path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(base, 'airline-passengers.csv')

        # Read CSV, parse 'Month' as datetime index
        self.df = pd.read_csv(path, parse_dates=["Month"], index_col="Month")
        # Save raw copy before normalization
        self.raw = self.df

        # Create 1‑D numeric array
        series = self.df["Passengers"].astype(float)
        self.index = series.index
        self.data = series.to_numpy(copy=True)

        # Normalization
        self.normalize = normalize
        self.mean = self.data.mean()
        self.std = self.data.std()
        if normalize:
            self.data = (self.data - self.mean) / self.std

    def asTorch(self):
        tensor = torch.from_numpy(self.data).float().unsqueeze(1)
        return tensor
    
    def inverse_transform(self, arr):
        if not self.normalize:
            return arr
        else:
            return arr * self.std + self.mean