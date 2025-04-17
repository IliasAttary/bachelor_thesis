import numpy as np
import torch


def _stack(windows, ref):
    """
    Stack a list of windows into a single array/tensor matching ref’s type.
    """
    if isinstance(ref, torch.Tensor):
        return torch.stack(windows)
    else:
        return np.array(windows)

def split_time_series(data, train_ratio=0.5, val_ratio=0.25):
    """
    Splits the time series data into train, validation, and test sets.
    
    Parameters:
        data (np.array or torch.Tensor): 1D or 2D array/tensor.
        train_ratio (float): Percentage for training.
        val_ratio (float): Percentage for validation.
        
    Returns:
        train, val, test: Three arrays/tensors of the same type as `data`.
    """
    # Calculate indices for the splits.
    L = data.shape[0]
    train_end = int(L * train_ratio)
    val_end = int(L * (train_ratio + val_ratio))
    # Extract each split.
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    return train, val, test

def sliding_window(data, window_size, horizon=1):
    """
    Creates overlapping sliding windows.
    For univariate data (1D), output shapes:
      X: (num_samples, window_size)
      y: (num_samples,)
    For multivariate data (2D with shape (N, f)), output shapes:
      X: (num_samples, window_size, f)
      y: (num_samples, f)
    
    Parameters:
        data (np.ndarray or torch.Tensor): 1D or 2D array/tensor.
        window_size (int): Number of time steps per input window.
        horizon (int): How many steps ahead to predict.
        
    Returns:
        X, y windows.
    """
    X = []
    y = []
    N = data.shape[0]

    # For univariate data:
    if data.ndim == 1:
        for i in range(N - window_size - horizon + 1):
            X.append(data[i : i + window_size])
            y.append(data[i + window_size + horizon - 1])

    # For multivariate data:
    elif data.ndim == 2:
        for i in range(N - window_size - horizon + 1):
            X.append(data[i : i + window_size, :])
            y.append(data[i + window_size + horizon - 1, :])
    else:
        raise ValueError("Data must be 1D or 2D array/tensor.")

    # If no windows, return empty of the correct shape/type
    if len(X) == 0:
        if isinstance(data, torch.Tensor):
            # torch.new_empty preserves device & dtype
            feature_dim = () if data.ndim == 1 else (data.shape[1],)
            return (
                data.new_empty((0, window_size) + feature_dim),
                data.new_empty((0,) + feature_dim)
            )
        else:
            feature_dim = () if data.ndim == 1 else (data.shape[1],)
            return (
                np.empty((0, window_size) + feature_dim),
                np.empty((0,) + feature_dim)
            )

    return _stack(X, data), _stack(y, data)

def windowing(data, train_window_size, horizon=1, train_ratio=0.5, val_ratio=0.25):
    """
    Splits the full time series into train/val/test, then creates sliding windows on each.
    
    Parameters:
        data (np.ndarray or torch.Tensor): Full series (N,) or (N, f).
        train_window_size (int): Window size for inputs.
        horizon (int): Steps ahead to predict.
        train_ratio (float): Fraction for train split.
        val_ratio (float): Fraction for validation split.
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
        (each an np.ndarray or torch.Tensor matching the type of `data`)
    """
    # Split the data into train, validation, and test.
    train, val, test = split_time_series(data, train_ratio, val_ratio)
    
    # Create sliding windows for each split using your sliding_window() function.
    # For training:
    X_train, y_train = sliding_window(train, train_window_size, horizon)
    # For validation:
    X_val,   y_val   = sliding_window(val,   train_window_size, horizon)
    # For test:
    X_test,  y_test  = sliding_window(test,  train_window_size, horizon)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == '__main__':
    # --- Test 1: split_time_series (Univariate) ---
    print("Test 1: split_time_series (Univariate)")
    uni_data = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    train, val, test = split_time_series(uni_data, train_ratio=0.5, val_ratio=0.25)
    assert np.array_equal(train, np.array([10,11,12,13,14,15])), "Train split incorrect."
    assert np.array_equal(val,   np.array([16,17,18])),            "Validation split incorrect."
    assert np.array_equal(test,  np.array([19,20,21])),            "Test split incorrect."
    print("  Passed.\n")

    # --- Test 2: sliding_window (Univariate) ---
    print("Test 2: sliding_window (Univariate)")
    X, y = sliding_window(train, window_size=3, horizon=1)
    expected_X = np.array([[10,11,12],[11,12,13],[12,13,14]])
    expected_y = np.array([13,14,15])
    assert X.shape == (3,3), f"Expected shape (3,3) but got {X.shape}."
    assert np.array_equal(X, expected_X), "X windows do not match expected output."
    assert np.array_equal(y, expected_y), "y targets do not match expected output."
    print("  Passed.\n")
    
    # --- Test 3: windowing (Univariate) ---
    print("Test 3: windowing (Univariate)")
    X_train, y_train, X_val, y_val, X_test, y_test = windowing(uni_data, train_window_size=3, horizon=1, train_ratio=0.5, val_ratio=0.25)
    assert X_train.shape == (3,3), f"X_train expected shape (3,3) but got {X_train.shape}."
    assert X_val.shape[0] == 0, f"X_val expected 0 windows but got {X_val.shape[0]}."
    assert X_test.shape[0] == 0, f"X_test expected 0 windows but got {X_test.shape[0]}."
    print("  Passed.\n")
    
    # --- Test 4: sliding_window (Multivariate) ---
    print("Test 4: sliding_window (Multivariate)")
    multi_data = np.array([
        [10,100],[11,101],[12,102],[13,103],[14,104],[15,105],
        [16,106],[17,107],[18,108],[19,109],[20,110],[21,111]
    ])
    X_multi, y_multi = sliding_window(multi_data, window_size=3, horizon=1)
    assert X_multi.shape == (9,3,2), f"Expected X_multi shape (9,3,2) but got {X_multi.shape}."
    expected_first_window = np.array([[10,100],[11,101],[12,102]])
    expected_first_target = np.array([13,103])
    assert np.array_equal(X_multi[0], expected_first_window), "First multivariate window incorrect."
    assert np.array_equal(y_multi[0], expected_first_target), "First multivariate target incorrect."
    print("  Passed.\n")
    
    # --- Test 4b: sliding_window (Multivariate >2 features) ---
    print("Test 4b: sliding_window (Multivariate >2 features)")
    # Create 8 timesteps, 3 features each: values 0..23 reshaped to (8,3)
    multi_data_3f = np.arange(8 * 3).reshape(8, 3)
    X_3f, y_3f = sliding_window(multi_data_3f, window_size=3, horizon=1)
    # Expect num_windows = 8 - 3 - 1 + 1 = 5
    assert X_3f.shape == (5, 3, 3), f"Expected shape (5,3,3) but got {X_3f.shape}."
    # First window should be rows 0,1,2
    expected_X0 = np.array([[ 0,  1,  2],
                            [ 3,  4,  5],
                            [ 6,  7,  8]])
    expected_y0 = np.array([ 9, 10, 11])
    assert np.array_equal(X_3f[0], expected_X0), "First 3-feature window incorrect."
    assert np.array_equal(y_3f[0], expected_y0),   "First 3-feature target incorrect."
    print("  Passed.\n")

    # --- Test 5: windowing (Multivariate) ---
    print("Test 5: windowing (Multivariate)")
    X_train_mv, y_train_mv, X_val_mv, y_val_mv, X_test_mv, y_test_mv = windowing(multi_data, train_window_size=3, horizon=1, train_ratio=0.5, val_ratio=0.25)
    assert X_train_mv.shape == (3,3,2), f"Expected X_train_mv shape (3,3,2) but got {X_train_mv.shape}."
    assert X_val_mv.shape[0] == 0, "Expected 0 windows for X_val_mv."
    assert X_test_mv.shape[0] == 0, "Expected 0 windows for X_test_mv."
    print("  Passed.\n")
    
    # --- Test 6: Error Check ---
    print("Test 6: Error on unsupported data dimensions")
    bad_data = np.random.rand(10,2,3)
    try:
        _ = sliding_window(bad_data, window_size=3, horizon=1)
    except ValueError as e:
        print("  Passed. Caught ValueError as expected:", e)
    else:
        raise AssertionError("Expected ValueError for data with ndim > 2.")
    print("\nAll tests passed successfully!")

    # --- Torch tests ---

    print("\n--- Running same tests on torch.Tensor inputs ---")

    # Torch Test 1
    print("Torch Test 1: split_time_series (Univariate)")
    uni_t = torch.tensor([10,11,12,13,14,15,16,17,18,19,20,21], dtype=torch.float)
    t_train, t_val, t_test = split_time_series(uni_t, train_ratio=0.5, val_ratio=0.25)
    assert torch.equal(t_train, torch.tensor([10,11,12,13,14,15], dtype=torch.float))
    assert torch.equal(t_val,   torch.tensor([16,17,18], dtype=torch.float))
    assert torch.equal(t_test,  torch.tensor([19,20,21], dtype=torch.float))
    print("  Passed.\n")

    # Torch Test 2
    print("Torch Test 2: sliding_window (Univariate)")
    X_t, y_t = sliding_window(t_train, window_size=3, horizon=1)
    expected_X_t = torch.tensor([[10,11,12],[11,12,13],[12,13,14]], dtype=torch.float)
    expected_y_t = torch.tensor([13,14,15], dtype=torch.float)
    assert X_t.shape == (3,3)
    assert torch.equal(X_t, expected_X_t)
    assert torch.equal(y_t, expected_y_t)
    print("  Passed.\n")

    # Torch Test 3
    print("Torch Test 3: windowing (Univariate)")
    Xt_tr, yt_tr, Xt_v, yt_v, Xt_te, yt_te = windowing(uni_t, train_window_size=3, horizon=1, train_ratio=0.5, val_ratio=0.25)
    assert Xt_tr.shape == (3,3)
    assert Xt_v.shape[0] == 0
    assert Xt_te.shape[0] == 0
    print("  Passed.\n")

    # Torch Test 4
    print("Torch Test 4: sliding_window (Multivariate)")
    multi_t = torch.tensor([
        [10,100],[11,101],[12,102],[13,103],[14,104],[15,105],
        [16,106],[17,107],[18,108],[19,109],[20,110],[21,111]
    ], dtype=torch.float)
    X_mt, y_mt = sliding_window(multi_t, window_size=3, horizon=1)
    assert X_mt.shape == (9,3,2)
    assert torch.equal(X_mt[0], torch.tensor([[10,100],[11,101],[12,102]], dtype=torch.float))
    assert torch.equal(y_mt[0], torch.tensor([13,103], dtype=torch.float))
    print("  Passed.\n")

    # --- Torch Test 4b: sliding_window (Multivariate >2 features) ---
    print("Torch Test 4b: sliding_window (Multivariate >2 features)")
    multi_t_3f = torch.arange(8 * 3, dtype=torch.float).reshape(8, 3)
    X_t3f, y_t3f = sliding_window(multi_t_3f, window_size=3, horizon=1)
    assert X_t3f.shape == (5, 3, 3), f"Torch: expected shape (5,3,3) but got {tuple(X_t3f.shape)}."
    assert torch.equal(X_t3f[0], torch.tensor(expected_X0, dtype=torch.float)), "Torch: first window incorrect."
    assert torch.equal(y_t3f[0], torch.tensor(expected_y0, dtype=torch.float)),   "Torch: first target incorrect."
    print("  Passed.\n")    

    # Torch Test 5
    print("Torch Test 5: windowing (Multivariate)")
    X_tr_m, y_tr_m, X_v_m, y_v_m, X_te_m, y_te_m = windowing(multi_t, train_window_size=3, horizon=1, train_ratio=0.5, val_ratio=0.25)
    assert X_tr_m.shape == (3,3,2)
    assert X_v_m.shape[0] == 0
    assert X_te_m.shape[0] == 0
    print("  Passed.\n")

    # Torch Test 6
    print("Torch Test 6: Error on unsupported data dimensions")
    bad_t = torch.rand(10,2,3)
    try:
        _ = sliding_window(bad_t, window_size=3, horizon=1)
    except ValueError:
        print("  Passed. Caught ValueError as expected.")
    else:
        raise AssertionError("Expected ValueError for ndim > 2")
    print("\nAll torch tests passed successfully!")    