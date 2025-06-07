import numpy as np


class DriftDetector():
    def __init__(self):
        self.drift_idx = []

    def update(self, value: float, t: int):
        pass

    def reset(self):
        pass

    def get_drift_points(self):
        return self.drift_idx


class ReconstructionErrorDetector(DriftDetector):
    def __init__(self, 
                 buffer_size: int = 200, 
                 z_threshold: float = 2.5,
                 use_fixed_threshold=False, 
                 retain_fraction: float = 0.25):
        
        super().__init__()
        self.buffer_size = buffer_size
        self.z_threshold = z_threshold
        self.use_fixed_threshold = use_fixed_threshold
        self.retain_fraction = retain_fraction
        self.buffer = []

    def update(self, reconstruction_error: float, idx: int):
        self.buffer.append(reconstruction_error)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        if len(self.buffer) == self.buffer_size:
            if self.use_fixed_threshold:
                mean = np.mean(self.buffer)
                std = np.std(self.buffer)
                if std > 0:
                    z_score = (reconstruction_error - mean) / std
                    if z_score > self.z_threshold:
                        self.drift_idx.append(idx)
                        self.reset()
            else:
                # Adaptive mode: use empirical percentile as threshold
                threshold = np.percentile(self.buffer, 95)
                if reconstruction_error > threshold:
                    self.drift_idx.append(idx)
                    self.reset()

    def reset(self):
        retain = int(self.retain_fraction * self.buffer_size)
        self.buffer = self.buffer[-retain:] if retain > 0 else []


class DynamicCUSUMDriftDetector(DriftDetector):
    def __init__(self, buffer_size: int = 200, 
                 margin_factor: float = 0.05,
                 threshold_factor: float = 5.0,
                 retain_fraction: float = 0.25):
        
        super().__init__()
        self.buffer_size = buffer_size
        self.margin_factor = margin_factor
        self.threshold_factor = threshold_factor
        self.retain_fraction = retain_fraction
        self.buffer = []
        self.cusum = 0

    def update(self, reconstruction_error: float, t: int):
        self.buffer.append(reconstruction_error)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        if len(self.buffer) < self.buffer_size:
            return

        target_mean = np.mean(self.buffer)
        buffer_std = np.std(self.buffer)

        drift_margin = self.margin_factor * buffer_std
        threshold = self.threshold_factor * drift_margin

        self.cusum += reconstruction_error - target_mean - drift_margin
        self.cusum = max(0, self.cusum)

        if self.cusum > threshold:
            self.drift_idx.append(t)
            self.reset()

    def reset(self):
        retain = int(self.retain_fraction * self.buffer_size)
        self.buffer = self.buffer[-retain:] if retain > 0 else []
        self.cusum = 0


class LatentDistanceDetector(DriftDetector):
    def __init__(self, 
                 buffer_size=200, 
                 threshold=None, 
                 percentile=95, 
                 retain_fraction=0.25):
        
        super().__init__()
        self.threshold = threshold
        self.percentile = percentile
        self.buffer_size = buffer_size
        self.retain_fraction = retain_fraction
        self.buffer = []

    def update(self, value: float, t: int):
        self.buffer.append(value)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        if self.threshold is not None:
            if value > self.threshold:
                self.drift_idx.append(t)
        elif len(self.buffer) == self.buffer_size:
            dynamic_thresh = np.percentile(self.buffer, self.percentile)
            if value > dynamic_thresh:
                self.drift_idx.append(t)
                self.reset()

    def reset(self):
        retain = int(self.buffer_size * self.retain_fraction)
        self.buffer = self.buffer[-retain:] if retain > 0 else []


class HDDM_A_Detector(DriftDetector):
    """
    HDDM_A: Hoeffding Drift Detection Method using absolute differences.
    Based on: Baena-Garcia et al. (2006)
    """

    def __init__(self, delta=0.0001):
        super().__init__()
        self.delta = delta
        self.window = []
        self.total = 0.0
        self.min_mean = float('inf')
        self.count = 0

    def update(self, value: float, t: int):
        self.window.append(value)
        self.total += value
        self.count += 1

        mean = self.total / self.count
        self.min_mean = min(self.min_mean, mean)

        # Hoeffding bound
        eps = np.sqrt((1 / (2 * self.count)) * np.log(1 / self.delta))

        if mean > self.min_mean + eps:
            self.drift_idx.append(t)
            self.reset()

    def reset(self):
        super().reset()
        self.window = []
        self.total = 0.0
        self.min_mean = float('inf')
        self.count = 0