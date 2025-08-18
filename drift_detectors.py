import numpy as np
from collections import deque


class DriftDetector():
    def __init__(self):
        self.drift_idx = []

    def update(self, value: float, idx: int):
        pass

    def reset(self):
        pass

    def get_drift_points(self):
        return self.drift_idx

class HDDM_TwoSided_Detector(DriftDetector):
    """
    Two-sided HDDM: detects both upward and downward significant changes.
    Based on Hoeffding's inequality with adaptive count.
    """

    def __init__(self, delta=0.001):
        super().__init__()
        self.delta = delta
        self.total = 0.0
        self.count = 0
        self.min_mean = float('inf')
        self.max_mean = float('-inf')

    def update(self, value: float, idx: int) -> bool:
        self.total += value
        self.count += 1

        mean = self.total / self.count
        self.min_mean = min(self.min_mean, mean)
        self.max_mean = max(self.max_mean, mean)

        # Hoeffding bound with adaptive n
        eps = np.sqrt((1 / (2 * self.count)) * np.log(1 / self.delta))

        drift_detected = False
        # Upward drift detection
        if mean > self.min_mean + eps:
            drift_detected = True
        # Downward drift detection
        elif mean < self.max_mean - eps:
            drift_detected = True

        if drift_detected:
            self.drift_idx.append(idx)
            self.reset()
            return True

        return False

    def reset(self):
        self.total = 0.0
        self.count = 0
        self.min_mean = float('inf')
        self.max_mean = float('-inf')