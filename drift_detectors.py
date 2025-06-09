import numpy as np


class DriftDetector():
    def __init__(self):
        self.drift_idx = []

    def update(self, value: float, idx: int):
        pass

    def reset(self):
        pass

    def get_drift_points(self):
        return self.drift_idx


class ZScoreReconstructionDriftDetector(DriftDetector): 
    def __init__(self, 
                 buffer_size: int = 200, 
                 z_threshold: float = 2.5,
                 retain_fraction: float = 0.25):
        super().__init__()
        self.buffer_size = buffer_size
        self.z_threshold = z_threshold
        self.retain_fraction = retain_fraction
        self.buffer = []

    def update(self, reconstruction_error: float, idx: int) -> bool:
        self.buffer.append(reconstruction_error)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        # if buffer is full, compute z-score and check for drift
        if len(self.buffer) == self.buffer_size:
            mean = np.mean(self.buffer)
            std = np.std(self.buffer)
            if std > 0:
                z_score = (reconstruction_error - mean) / std
                # if error is z_threshold stds above the mean, then a drift is detected
                if z_score > self.z_threshold:
                    self.drift_idx.append(idx)
                    self.reset()
                    return True

        return False

    def reset(self):
        # retain a portion of the buffer after drift
        retain = int(self.retain_fraction * self.buffer_size)
        self.buffer = self.buffer[-retain:] if retain > 0 else []


class PercentileReconstructionDriftDetector(DriftDetector):
    def __init__(self, 
                 buffer_size: int = 200, 
                 percentile: float = 95,
                 retain_fraction: float = 0.25):
        super().__init__()
        self.buffer_size = buffer_size
        self.percentile = percentile
        self.retain_fraction = retain_fraction
        self.buffer = []

    def update(self, reconstruction_error: float, idx: int) -> bool:
        self.buffer.append(reconstruction_error)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        # if buffer is full, check if error exceeds percentile threshold
        if len(self.buffer) == self.buffer_size:
            threshold = np.percentile(self.buffer, self.percentile)
            # if error exceeds the given percentile, then a drift is detected
            if reconstruction_error > threshold:
                self.drift_idx.append(idx)
                self.reset()
                return True  # drift detected

        return False  # no drift

    def reset(self):
        # retain a portion of the buffer after drift
        retain = int(self.retain_fraction * self.buffer_size)
        self.buffer = self.buffer[-retain:] if retain > 0 else []


class TrendDetector(DriftDetector):
    """""
    Based on: Jaworski, M., Rutkowski, L., Angelov, P. (2020): 
    Concept Drift Detection Using Autoencoders in Data Streams Processing.
    """""
    def __init__(self, lambda_: float = 0.98, threshold: float = 0.001):
        super().__init__()
        self.lambda_ = lambda_
        self.threshold = threshold

        self.TC = 0.0     # ∑ t * C(M_t)
        self.T = 0.0      # ∑ t
        self.C_sum = 0.0  # ∑ C(M_t)
        self.T2 = 0.0     # ∑ t^2
        self.n = 0.0      # ∑ 1
        self.t = 0        # current timestep (of the full window)

    def update(self, reconstruction_error: float, idx: int) -> bool:
        self.t += 1
        t = self.t

        λ = self.lambda_

        # Update exponentially decayed stats
        self.TC   = λ * self.TC   + t * reconstruction_error
        self.T    = λ * self.T    + t
        self.C_sum= λ * self.C_sum+ reconstruction_error
        self.T2   = λ * self.T2   + t * t
        self.n    = λ * self.n    + 1

        # Compute trend score Q(C, t)
        numerator = self.n * self.TC - self.T * self.C_sum
        denominator = self.n * self.T2 - self.T ** 2

        if denominator == 0:
            return False  # not enough variation yet

        Q = numerator / denominator

        if Q > self.threshold:
            self.drift_idx.append(idx)
            self.reset()
            return True

        return False

    def reset(self):
        self.TC = 0.0
        self.T = 0.0
        self.C_sum = 0.0
        self.T2 = 0.0
        self.n = 0.0
        self.t = 0


class HDDM_A_Detector(DriftDetector):
    """
    HDDM_A: Hoeffding Drift Detection Method using absolute differences.
    Based on: Baena-Garcia et al. (2006)
    """

    def __init__(self, delta=0.0001):
        super().__init__()
        self.delta = delta
        self.total = 0.0
        self.min_mean = float('inf')
        self.count = 0

    def update(self, error: float, idx: int) -> bool:
        self.total += error
        self.count += 1

        mean = self.total / self.count
        self.min_mean = min(self.min_mean, mean)

        # Hoeffding bound
        eps = np.sqrt((1 / (2 * self.count)) * np.log(1 / self.delta))

        if mean > self.min_mean + eps:
            self.drift_idx.append(idx)
            self.reset()
            return True  # drift detected

        return False  # no drift

    def reset(self):
        self.total = 0.0
        self.min_mean = float('inf')
        self.count = 0

class HDDM_W_Detector(DriftDetector):
    """
    HDDM_W: Hoeffding Drift Detection Method using a weighted (exponentially decayed) moving average.
    Based on: Baena-Garcia et al. (2006)
    """

    def __init__(self, delta: float = 0.0001, alpha: float = 0.98):
        """
        Parameters:
        - delta: confidence level for Hoeffding bound (smaller = more conservative)
        - alpha: decay factor for exponential weighting (closer to 1 = longer memory)
        """
        super().__init__()
        self.delta = delta
        self.alpha = alpha
        self.weighted_mean = 0.0
        self.min_weighted_mean = float('inf')
        self.weighted_count = 0.0  # sum of decayed weights

    def update(self, error: float, idx: int) -> bool:
        # Exponentially decayed mean update
        self.weighted_mean = self.alpha * self.weighted_mean + (1 - self.alpha) * error
        self.weighted_count = self.alpha * self.weighted_count + (1 - self.alpha)

        # Track min of weighted mean over time
        self.min_weighted_mean = min(self.min_weighted_mean, self.weighted_mean)

        # Compute Hoeffding bound using sum of decayed weights
        eps = np.sqrt((1 / (2 * self.weighted_count)) * np.log(1 / self.delta))

        if self.weighted_mean > self.min_weighted_mean + eps:
            self.drift_idx.append(idx)
            self.reset()
            return True

        return False

    def reset(self):
        self.weighted_mean = 0.0
        self.min_weighted_mean = float('inf')
        self.weighted_count = 0.0