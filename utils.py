import numpy as np

def split_time_series(data, train_ratio=0.5, val_ratio=0.25):
    """
    Splits the time series data into train, validation, and test sets.
    
    Parameters:
        data (np.array): 1D or 2D numpy array.
        train_ratio (float): Percentage for training.
        val_ratio (float): Percentage for validation.
        
    Returns:
        train, val, test: Three numpy arrays.
    """
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
    """
    Creates overlapping sliding windows.
    For univariate data (1D), output shapes:
      X: (num_samples, window_size)
      y: (num_samples,)
    For multivariate data (2D with shape (N, f)), output shapes:
      X: (num_samples, window_size, f)
      y: (num_samples, f)
    
    Parameters:
        data (np.array): 1D or 2D array.
        window_size (int): Number of time steps per input window.
        horizon (int): How many steps ahead to predict.
        
    Returns:
        X, y windows.
    """
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
    """
    Splits the full time series into training, validation, and test portions and then creates
    overlapping sliding windows for each portion.

    Parameters:
        data (np.array): The full time series (univariate: shape (N,), or multivariate: shape (N, f)).
        train_window_size (int): The window size (number of past time steps) to use for both training and prediction.
        horizon (int): How many steps ahead to predict. Default is 1.
        train_ratio (float): Fraction of the data used for training.
        val_ratio (float): Fraction of the data used for validation.
                          The remaining part will be used for testing.
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test

        Where:
         - X_train: Training inputs, shape = (num_train_samples, train_window_size) for univariate,
                    or (num_train_samples, train_window_size, f) for multivariate.
         - y_train: Training targets, shape = (num_train_samples,) for univariate,
                    or (num_train_samples, f) for multivariate.
         - X_val: Validation inputs, same idea as X_train.
         - y_val: Validation targets.
         - X_test: Test inputs.
         - y_test: Test targets.
    """
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

if __name__ == '__main__':
    # --- Test 1: split_time_series (Univariate) ---
    print("Test 1: split_time_series (Univariate)")
    uni_data = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    # Expect: train = first 6 points, val = next 3 points, test = last 3 points (because 12 * 0.5=6, 12*0.75=9)
    train, val, test = split_time_series(uni_data, train_ratio=0.5, val_ratio=0.25)
    assert np.array_equal(train, np.array([10, 11, 12, 13, 14, 15])), "Train split incorrect."
    assert np.array_equal(val, np.array([16, 17, 18])), "Validation split incorrect."
    assert np.array_equal(test, np.array([19, 20, 21])), "Test split incorrect."
    print("  Passed.\n")

    # --- Test 2: sliding_window (Univariate) ---
    print("Test 2: sliding_window (Univariate)")
    # Use train split from above: [10, 11, 12, 13, 14, 15]
    window_size = 3
    horizon = 1
    X, y = sliding_window(train, window_size=window_size, horizon=horizon)
    # Calculation: length = 6, window_size = 3, horizon = 1 → number of windows = 6 - 3 - 1 + 1 = 3.
    expected_X = np.array([
        [10, 11, 12],
        [11, 12, 13],
        [12, 13, 14]
    ])
    expected_y = np.array([13, 14, 15])
    assert X.shape == (3, window_size), f"Expected shape (3,{window_size}) but got {X.shape}."
    assert np.array_equal(X, expected_X), "X windows do not match expected output."
    assert np.array_equal(y, expected_y), "y targets do not match expected output."
    print("  Passed.\n")
    
    # --- Test 3: windowing (Univariate) ---
    print("Test 3: windowing (Univariate)")
    # Using the same uni_data, with train_ratio=0.5 and val_ratio=0.25.
    # With window_size = 3 and horizon = 1.
    X_train, y_train, X_val, y_val, X_test, y_test = windowing(uni_data, train_window_size=3, horizon=1, train_ratio=0.5, val_ratio=0.25)
    # For train split ([10..15]): len=6 → number of windows = 6 - 3 - 1 + 1 = 3.
    assert X_train.shape == (3, 3), f"X_train expected shape (3,3) but got {X_train.shape}."
    # For validation split ([16,17,18]): len=3 → 3 - 3 - 1 + 1 = 0 windows.
    # Many implementations of sliding_window return empty arrays when len(data) equals window_size.
    # Depending on intended behavior, here we expect an empty array.
    assert X_val.shape[0] == 0, f"X_val expected 0 windows but got {X_val.shape[0]}."
    # For test split ([19,20,21]): len=3 → expect 0 windows as well.
    assert X_test.shape[0] == 0, f"X_test expected 0 windows but got {X_test.shape[0]}."
    print("  Passed.\n")
    
    # --- Test 4: sliding_window (Multivariate) ---
    print("Test 4: sliding_window (Multivariate)")
    # Create a simple multivariate dataset with 12 timesteps and 2 features.
    multi_data = np.array([
        [10, 100],
        [11, 101],
        [12, 102],
        [13, 103],
        [14, 104],
        [15, 105],
        [16, 106],
        [17, 107],
        [18, 108],
        [19, 109],
        [20, 110],
        [21, 111]
    ])
    window_size = 3
    horizon = 1
    X_multi, y_multi = sliding_window(multi_data, window_size=window_size, horizon=horizon)
    # For multivariate, expected shape X: (num_samples, window_size, features)
    # There should be 12 - 3 - 1 + 1 = 9 samples.
    assert X_multi.shape == (9, 3, 2), f"Expected X_multi shape (9,3,2) but got {X_multi.shape}."
    # Check that the first window is correct.
    expected_first_window = np.array([[10, 100], [11, 101], [12, 102]])
    assert np.array_equal(X_multi[0], expected_first_window), "First window in multivariate case is incorrect."
    # And y_multi should be (9,2)
    expected_first_target = np.array([13, 103])
    assert np.array_equal(y_multi[0], expected_first_target), "First target in multivariate case is incorrect."
    print("  Passed.\n")
    
    # --- Test 5: windowing (Multivariate) ---
    print("Test 5: windowing (Multivariate)")
    # Let's use the same multivariate data for windowing.
    # Use train_ratio=0.5 and val_ratio=0.25. For 12 points: train=6, val=3, test=3.
    X_train_mv, y_train_mv, X_val_mv, y_val_mv, X_test_mv, y_test_mv = windowing(multi_data, train_window_size=3, horizon=1, train_ratio=0.5, val_ratio=0.25)
    # For train: train data has shape (6,2), so expected number of windows = 6 - 3 - 1 + 1 = 3.
    assert X_train_mv.shape == (3, 3, 2), f"Expected X_train_mv shape (3,3,2) but got {X_train_mv.shape}."
    # For validation: val data has 3 time steps, so no window can be formed.
    assert X_val_mv.shape[0] == 0, "Expected 0 windows for X_val_mv."
    # For test: test data has 3 time steps, so no window again.
    assert X_test_mv.shape[0] == 0, "Expected 0 windows for X_test_mv."
    print("  Passed.\n")
    
    # --- Test 6: Error Check ---
    print("Test 6: Error on unsupported data dimensions")
    # Create a 3D array and ensure it raises ValueError.
    bad_data = np.random.rand(10, 2, 3)
    try:
        _ = sliding_window(bad_data, window_size=3, horizon=1)
    except ValueError as e:
        print("  Passed. Caught ValueError as expected:", e)
    else:
        raise AssertionError("Expected ValueError for data with ndim > 2.")
    
    print("\nAll tests passed successfully!")