import numpy as np
import torch
import torch.nn.functional as F
from tslearn.metrics import dtw as tslearn_dtw

def split_time_series(data, train_ratio=0.5, val_ratio=0.25):
    # Calculate indices for the splits.
    L = len(data)
    train_end = int(L * train_ratio)
    val_end = int(L * (train_ratio + val_ratio))
    
    # Extract each split.
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    return train, val, test

def sliding_window(data, window_size, horizon=1):
    X = []
    y = []

    # For univariate data:
    if data.ndim == 1:
        for i in range(len(data) - window_size - horizon + 1):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size+horizon-1])
        X = np.array(X)
        y = np.array(y)

    # For multivariate data:
    elif data.ndim == 2:
        for i in range(len(data) - window_size - horizon + 1):
            # Slice keeping feature axis intact.
            X.append(data[i:i+window_size, :])
            y.append(data[i+window_size+horizon-1, :])
        X = np.array(X)
        y = np.array(y)
    else:
        raise ValueError("Data must be 1D or 2D numpy array.")
        
    return X, y

def windowing(data, window_size, normalize = True, horizon=1, train_ratio=0.5, val_ratio=0.25):
    # Split the data into train, validation, and test.
    train, val, test = split_time_series(data, train_ratio, val_ratio)
    
    if normalize:
        mean = train.mean()
        std = train.std()
        train = (train - mean) / std
        val = (val - mean) / std
        test = (test - mean) / std
    else:
        mean, std = None, None

    # Create sliding windows for each split:
    X_train, y_train = sliding_window(train, window_size, horizon)
    X_val, y_val = sliding_window(val, window_size, horizon)
    X_test, y_test = sliding_window(test, window_size, horizon)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def compute_distance(win1, win2, metric="euclidean"):
    """
    Args:
        win1, win2: numpy.ndarray or torch.Tensor of the same shape
        metric (str): "euclidean", "manhattan", or "cosine"
    """
    if type(win1) != type(win2):
        raise TypeError(f"Both windows must be of the same type: first window is {type(win1)}")

    if win1.shape != win2.shape:
        raise ValueError(f"Shape mismatch: {win1.shape} vs {win2.shape}")

    # If numpy arrays
    if isinstance(win1, np.ndarray):
        if metric == "euclidean":
            return np.linalg.norm(win1 - win2)
        elif metric == "manhattan":
            return np.sum(np.abs(win1 - win2))
        elif metric == "cosine":
            num = np.dot(win1.flatten(), win2.flatten())
            denom = np.linalg.norm(win1) * np.linalg.norm(win2)
            return 1 - num / denom
        elif metric == "dtw":
            return tslearn_dtw(win1.reshape(-1, 1), win2.reshape(-1, 1))
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    # If torch tensors
    elif isinstance(win1, torch.Tensor):
        if metric == "euclidean":
            return torch.norm(win1 - win2).item()
        elif metric == "manhattan":
            return torch.sum(torch.abs(win1 - win2)).item()
        elif metric == "cosine":
            return 1 - F.cosine_similarity(win1.flatten(), win2.flatten(), dim=0).item()
        elif metric == "dtw":
            # tslearn expects shape (n_timestamps, n_features) so we transpose (C, L) → (L, C)
            win1_np = win1.detach().cpu().numpy()
            win2_np = win2.detach().cpu().numpy()
            return tslearn_dtw(win1_np.T, win2_np.T)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    else:
        raise TypeError("Unsupported input type. Use NumPy arrays or PyTorch tensors.")