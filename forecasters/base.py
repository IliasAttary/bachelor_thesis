import numpy as np
import torch
from sklearn.cluster import KMeans
from kneed import KneeLocator

class Forecaster:
    def __init__(self):
        self.rocs = {"raw": [], "latent": []}
        self.centers = []

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
    
    def compute_kmeans_centers(self, k_min=2, k_max=10, random_state=0):
        # Extract latent windows
        latent_windows = self.rocs.get("latent", [])
        if not latent_windows:
            raise ValueError("No latent windows available for clustering.")

        # Determine integer bounds for k
        k_min_int = max(2, int(k_min))
        k_max_int = max(k_min_int, int(k_max))

        # Stack flattened latent windows into numpy array X
        X = np.vstack([
            (w.detach().cpu().numpy() if isinstance(w, torch.Tensor) else np.array(w))
            .flatten() for w in latent_windows
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

        # Final k-means on X
        km_final = KMeans(n_clusters=best_k, random_state=random_state).fit(X)
        centers_flat = km_final.cluster_centers_  # shape (best_k, D)

        # Convert each center back to latent shape and to Tensor
        orig_shape = latent_windows[0].shape  # (C, L)
        centers = []
        for c in centers_flat:
            arr = c.reshape(orig_shape)
            centers.append(torch.tensor(arr, dtype=torch.float32))

        # Preserve rocs; set new centers
        self.centers = centers
        return best_k