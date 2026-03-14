import numpy as np
from scipy.spatial.distance import cdist

class FuzzyCMeans:
    def __init__(self, n_clusters=3, m=2, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n_samples = X.shape[0]
        U = np.random.dirichlet(np.ones(self.n_clusters), size=n_samples)
        for _ in range(self.max_iter):
            U_old = U.copy()
            um = U ** self.m
            centroids = (um.T @ X) / np.sum(um.T, axis=1, keepdims=True)
            dist = cdist(X, centroids, metric='euclidean')
            dist = np.fmax(dist, 1e-10)
            U = 1.0 / (dist ** (2/(self.m-1)))
            U = U / np.sum(U, axis=1, keepdims=True)
            if np.linalg.norm(U - U_old) < self.tol:
                break
        self.centroids = centroids
        self.U = U
        return self

    def predict(self, X):
        dist = cdist(X, self.centroids, metric='euclidean')
        dist = np.fmax(dist, 1e-10)
        U = 1.0 / (dist ** (2/(self.m-1)))
        U = U / np.sum(U, axis=1, keepdims=True)
        return U

    def get_segmentation_map(self, X):
        U = self.predict(X)
        return np.argmax(U, axis=1)

    def get_membership_map(self, X):
        U = self.predict(X)
        return U
