import os
import numpy as np
import pandas as pd

class PowerDemand:
    def __init__(self, path=None, length=None):
        # Determine file path
        if path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(base, 'power_demand_kw_per_3h.csv')

        # Read CSV with date parsing
        self.df = pd.read_csv(path, parse_dates=True, index_col=0)
        # Keep only the last "length" entries, if specified
        if length is not None:
            self.df = self.df.iloc[-length:]
        # Save raw copy
        self.raw = self.df.copy()

        # Extract and clean numeric series
        series = self.df.iloc[:, 0].astype(float)
        self.index = series.index
        self.data = series.to_numpy(copy=True)

        # Compute stats
        self.mean = self.data.mean()
        self.std = self.data.std()

    def inverse_transform(self, arr, isnormalized=True):
        if not isnormalized:
            return arr
        return arr * self.std + self.mean