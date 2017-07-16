import numpy as np
import scipy
from sklearn.cluster import KMeans


def denormalized_spectral_clustering(X, k):
    D = np.diag(X.sum(axis=0))
    L = D - X

    eigen_value, eigen_vector = scipy.linalg.eigh(L)
    positive = eigen_value >= 0
    eigen_value = eigen_value[positive]
    eigen_vector = eigen_vector.T[positive]
    indexes = eigen_value.argsort()
    features = eigen_vector[indexes]
    features = features[:k]

    kmeans = KMeans(n_clusters=k, random_state=0).fit(features.T)
    return kmeans.labels_


def normalized_spectral_clustering(X, k):
    D = np.diag(X.sum(axis=0))
    L = D - X

    D_inv = np.linalg.inv(D)
    L_sym = np.dot(
        np.dot(
            scipy.linalg.sqrtm(D_inv), L
        ),
        scipy.linalg.sqrtm(D_inv)
    )

    eigen_value, eigen_vector = scipy.linalg.eigh(L_sym)
    positive = eigen_value >= 0
    eigen_value = eigen_value[positive]
    eigen_vector = eigen_vector.T[positive]
    indexes = eigen_value.argsort()
    features = eigen_vector[indexes]
    features = features[:k]

    features = features/np.linalg.norm(features)

    kmeans = KMeans(n_clusters=k, random_state=0).fit(features.T)
    return kmeans.labels_


def random_walk_normalized_spectral_clustering(X, k):
    D = np.diag(X.sum(axis=0))
    L = D - X

    D_inv = np.linalg.inv(D)
    L_rw = np.dot(D_inv, L)

    eigen_value, eigen_vector = scipy.linalg.eigh(L_rw)
    positive = eigen_value >= 0
    eigen_value = eigen_value[positive]
    eigen_vector = eigen_vector.T[positive]
    indexes = eigen_value.argsort()
    features = eigen_vector[indexes]
    features = features[:k]

    kmeans = KMeans(n_clusters=k, random_state=0).fit(features.T)
    return kmeans.labels_
