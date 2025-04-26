import os
import torch
import numpy as np
import pandas as pd

class Rossmann:
    def __init__(self, path=None, normalize=True):
        # Determine file path
        if path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(base, 'rossmann.csv')

        # Read CSV, parse 'Date' as datetime index
        self.df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")

        # Drop the 'StateHoliday' column (Non-Numerical Features)
        if 'StateHoliday' in self.df.columns:
            self.df = self.df.drop(columns=['StateHoliday'])

        self.raw = self.df.copy()
        self.index = self.df.index

        # Extract all numeric features
        self.data = self.df.select_dtypes(include=[np.number]).to_numpy(copy=True)

        # Normalization
        self.normalize = normalize
        self.mean = self.data.mean(axis=0)
        self.std = self.data.std(axis=0)
        if normalize:
            self.data = (self.data - self.mean) / self.std

    def asTorch(self):
        tensor = torch.from_numpy(self.data).float()
        return tensor

    def inverse_transform(self, arr):
        if not self.normalize:
            return arr
        else:
            return arr * self.std + self.mean