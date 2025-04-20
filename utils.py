import numpy as np

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

def windowing(data, train_window_size, horizon=1, train_ratio=0.5, val_ratio=0.25):
    # Split the data into train, validation, and test.
    train, val, test = split_time_series(data, train_ratio=train_ratio, val_ratio=val_ratio)
    
    # Create sliding windows for each split using your sliding_window() function.
    # For training:
    X_train, y_train = sliding_window(train, window_size=train_window_size, horizon=horizon)
    # For validation:
    X_val, y_val = sliding_window(val, window_size=train_window_size, horizon=horizon)
    # For test:
    X_test, y_test = sliding_window(test, window_size=train_window_size, horizon=horizon)
    
    return X_train, y_train, X_val, y_val, X_test, y_test