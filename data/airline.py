import os
import pandas as pd
import torch
import numpy as np

class Airline:
    """
    Loader for the monthly Airline Passengers dataset from a local CSV file.

    Args:
        path (str): Optional path to 'airline-passengers.csv'. If None, assumes CSV is
                    alongside this file.
        normalize (bool): If True, apply z-score normalization (mean/stdev).
        as_numpy (bool): If True, .data is a NumPy 1D array; otherwise, a torch.FloatTensor.

    Attributes:
        data (np.ndarray or torch.Tensor): Time series data (normalized if requested).
        means (float or None): Mean used for normalization.
        stds (float or None): Std dev used for normalization.
        index (pd.DatetimeIndex): Original datetime index of the series.
    """
    def __init__(self, path=None, normalize=True, as_numpy=True):
        # Determine file path
        if path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(base, 'airline-passengers.csv')

        # Read CSV, parse 'Month' as datetime index
        df = pd.read_csv(path, parse_dates=['Month'], index_col='Month')

        # Extract passenger counts by column name for clarity
        series = df['Passengers'].astype(float)
        self.index = series.index
        values = series.values.copy()

        # Store normalize flag
        self.normalize = normalize
        # Z-score normalization if requested
        if normalize:
            self.means = values.mean()
            self.stds = values.std()
            values = (values - self.means) / self.stds
        else:
            self.means = None
            self.stds = None

        # Store data as numpy array or torch tensor
        self.as_numpy = as_numpy
        if as_numpy:
            self.data = values
        else:
            self.data = torch.from_numpy(values).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def inverse_transform(self, arr):
        """
        Reverse z-score transform: x_orig = x_norm * std + mean.
        If normalize=False, returns arr unchanged.
        Supports numpy arrays and torch tensors.
        """
        if not self.normalize:
            return arr
        if isinstance(arr, np.ndarray):
            return arr * self.stds + self.means
        if isinstance(arr, torch.Tensor):
            return arr * self.stds + self.means
        raise TypeError("Input must be numpy array or torch Tensor.")

    def as_series(self) -> pd.Series:
        """
        Return the loaded data as a pandas Series indexed by the original dates.
        """
        data = self.data if self.as_numpy else self.data.numpy()
        return pd.Series(data, index=self.index)


# ------------------- Test Suite -------------------
if __name__ == '__main__':
    print("Running Airline loader tests...")

    # Reference series via pandas
    base = os.path.dirname(os.path.abspath(__file__))
    ref_df = pd.read_csv(
        os.path.join(base, 'airline-passengers.csv'),
        parse_dates=['Month'], index_col='Month'
    )
    ref_series = ref_df['Passengers'].astype(float)

    # Test 1: raw numpy data
    loader_raw = Airline(normalize=False, as_numpy=True)
    assert isinstance(loader_raw.data, np.ndarray)
    np.testing.assert_allclose(loader_raw.data, ref_series.values, rtol=1e-6)
    assert loader_raw.means is None and loader_raw.stds is None
    assert len(loader_raw) == len(ref_series)
    assert loader_raw[0] == ref_series.values[0]

    # Test 2: normalized numpy data
    loader_norm_np = Airline(normalize=True, as_numpy=True)
    data_norm = loader_norm_np.data
    assert np.isclose(data_norm.mean(), 0, atol=1e-6)
    assert np.isclose(data_norm.std(), 1, atol=1e-6)
    recovered = loader_norm_np.inverse_transform(data_norm)
    np.testing.assert_allclose(recovered, ref_series.values, rtol=1e-6)

    # Test 3: normalized torch data
    loader_norm_t = Airline(normalize=True, as_numpy=False)
    assert isinstance(loader_norm_t.data, torch.Tensor)
    tvals = loader_norm_t.data
    inv = loader_norm_t.inverse_transform(tvals)
    np.testing.assert_allclose(inv.numpy(), ref_series.values, rtol=1e-6)

    # Test 4: as_series()
    series_np = loader_norm_np.as_series()
    assert isinstance(series_np, pd.Series)
    assert series_np.index.equals(ref_series.index)

    # Test 6: Plot the time series
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    ts = loader_raw.as_series()
    plt.plot(ts.index, ts.values, marker='o')
    plt.title('Monthly International Airline Passengers')
    plt.xlabel('Month')
    plt.ylabel('Passengers')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("All Airline loader tests passed!")
