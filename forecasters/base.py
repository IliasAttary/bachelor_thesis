import numpy as np
import torch
from sklearn.cluster import KMeans
from kneed import KneeLocator


class Forecaster:
    def __init__(self):
        self.rocs = {"raw": [], "latent": []}
        self.centers = {"raw": [], "latent": []}

    def fit(self, X, y, generator=None):
        return self._fit(X, y)

    def predict(self, X):
        pass

    def compute_kmeans_centers(self, mode="latent", k_min=2, k_max=10, random_state=None):
        if mode not in self.rocs:
            raise ValueError(f"Invalid mode: {mode}. Must be 'latent' or 'raw'.")

        # Extract windows
        windows = self.rocs[mode]
        if not windows:
            raise ValueError(f"No {mode} windows available for clustering.")

        # Determine integer bounds for k
        k_min_int = max(2, int(k_min))
        k_max_int = max(k_min_int, int(k_max))

        # Flatten windows
        X = np.vstack([
            (w.detach().cpu().numpy() if isinstance(w, torch.Tensor) else np.array(w)).flatten()
            for w in windows
        ])

        # Compute inertia for each k
        ks = list(range(k_min_int, k_max_int + 1))
        inertias = []
        for k in ks:
            km = KMeans(n_clusters=k, random_state=random_state)
            km.fit(X)
            inertias.append(km.inertia_)

        # Use KneeLocator to pick elbow k (fallback to k_min_int)
        kl = KneeLocator(ks, inertias, curve='convex', direction='decreasing')
        best_k = kl.elbow or k_min_int

        # Final clustering
        km_final = KMeans(n_clusters=best_k, random_state=random_state).fit(X)
        centers_flat = km_final.cluster_centers_
        orig_shape = windows[0].shape

        # Format cluster centers
        if mode == "latent":
            centers = [
                torch.tensor(c.reshape(orig_shape), dtype=torch.float32)
                for c in centers_flat
            ]
        else:  # mode == "raw"
            centers = [
                c.reshape(orig_shape)  # Keep as numpy
                for c in centers_flat
            ]

        # Store correctly
        self.centers[mode] = centers
        return best_k