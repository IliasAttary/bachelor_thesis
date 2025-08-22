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

class FixedRefHoeffdingDetector(DriftDetector):

    def __init__(self, mu_ref: float, r: float, sigma: float = 0.99):
        super().__init__()
        self.mu0 = float(mu_ref)
        self.r = float(r)
        self.sigma = float(sigma)

        # running stats for post-validation window (since last reset)
        self.count = 0
        self.total = 0.0

    def _eps(self) -> float:
        # W = self.count; guard W>=1
        W = max(1, self.count)
        return np.sqrt((self.r * self.r * np.log(2.0 / self.sigma)) / (2.0 * W))

    def update(self, value: float, idx: int) -> bool:
        x = float(value)
        self.total += x
        self.count += 1

        mu_j = self.total / self.count
        if abs(mu_j - self.mu0) > self._eps():
            # drift detected: record and re-baseline
            self.drift_idx.append(idx)
            self.mu0 = mu_j
            self.total = 0.0
            self.count = 0
            return True
        return False

    def reset(self, reset_reference: bool = False, new_mu_ref: float | None = None):
        # Clear the post-drift window. Optionally reset the reference mean
        self.total = 0.0
        self.count = 0
        if reset_reference:
            if new_mu_ref is None:
                raise ValueError("new_mu_ref must be provided when reset_reference=True")
            self.mu0 = float(new_mu_ref)