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


def draw_matrix(matrix, filename="tmp.png", borders_x=None, borders_y=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.matshow(matrix)

    if borders_x:
        for border in borders_x:
            plt.axvline(border-0.5, c="red")
    if borders_y:
        for border in borders_y:
            plt.axhline(border-0.5, c="blue")

    plt.savefig(filename)


def sort_matrix(matrix, labels, n_clusters, axis="xy"):
    indexes = np.array([], dtype="int")
    borders = list()
    border = 0
    for i in range(n_clusters):
        tmp_ind = np.where(labels == i)[0]
        border += len(tmp_ind)
        borders.append(border)
        indexes = np.hstack((indexes, tmp_ind))

    if axis == "xy":
        new_matrix = matrix[:, indexes]
        new_matrix = new_matrix[indexes, :]
    elif axis == "x":
        new_matrix = matrix[:, indexes]
    elif axis == "y":
        new_matrix = matrix[indexes, :]

    return new_matrix, borders
