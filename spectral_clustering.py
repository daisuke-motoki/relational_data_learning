import networkx as ntx
import numpy as np
import scipy
from sklearn.cluster import KMeans

def to_adjacency_matrix(ntx_graph):
    n_node = len(ntx_graph.node)
    matrix = np.identity(n_node, dtype="int")
    for node in ntx_graph:
        edges = ntx_graph.edge[node]
        relates = np.array(list(edges.keys()))
        matrix[node, relates] = 1

    return matrix


def draw_matrix(matrix, filename="tmp.png", borders=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.matshow(matrix)

    if borders:
        for border in borders:
            plt.axhline(border-0.5, c="red")
            plt.axvline(border-0.5, c="red")

    plt.savefig(filename)


def sort_matrix(matrix, labels, n_clusters):
    indexes = np.array([], dtype="int")
    borders = list()
    border = 0
    for i in range(n_clusters):
        tmp_ind = np.where(labels == i)[0]
        border += len(tmp_ind)
        borders.append(border)
        indexes = np.hstack((indexes, tmp_ind))

    new_matrix = matrix[:, indexes]
    new_matrix = new_matrix[indexes]
    return new_matrix, borders


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


if __name__ == "__main__":

    karate_graph = ntx.karate_club_graph()
    karate_matrix = to_adjacency_matrix(karate_graph)
    draw_matrix(karate_matrix, "original.png")

    n_clusters = 2
    # ---------------------------------
    # denormalized sepectral clustering
    # ---------------------------------
    labels = denormalized_spectral_clustering(karate_matrix, n_clusters)
    new_karate_matrix, borders = sort_matrix(karate_matrix, labels, n_clusters)
    draw_matrix(new_karate_matrix, "denormalized.png", borders)

    # ------------------------------
    # normalized spectral clustering
    # ------------------------------
    labels = normalized_spectral_clustering(karate_matrix, n_clusters)
    new_karate_matrix, borders = sort_matrix(karate_matrix, labels, n_clusters)
    draw_matrix(new_karate_matrix, "normalized.png", borders)

    # ----------------------------------
    # random-walk normalizaed clustering
    # ----------------------------------
    labels = random_walk_normalized_spectral_clustering(karate_matrix, n_clusters)
    new_karate_matrix, borders = sort_matrix(karate_matrix, labels, n_clusters)
    draw_matrix(new_karate_matrix, "random_walk_normalized.png", borders)
