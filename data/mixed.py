import os
import numpy as np
import pandas as pd

class Mixed:
    def __init__(self, path=None, length=None):
        # Determine file path
        if path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(base, 'GRD_SUD.csv')

        # Read CSV, parse 'Date' as datetime index
        self.df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        # Keep only last "length" elements
        if length is not None:
            self.df = self.df.iloc[-length:]
        # Save raw copy
        self.raw = self.df.copy()

        # Create 1‑D numeric array
        series = self.df["Value"].astype(float)
        self.index = series.index
        self.data = series.to_numpy(copy=True)

        # Save mean and standard deviation
        self.mean = self.data.mean()
        self.std = self.data.std()

    def inverse_transform(self, arr, isnormalized = True):
        if not isnormalized:
            return arr 
        else: 
            return arr * self.std + self.mean