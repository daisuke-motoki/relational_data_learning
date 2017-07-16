import networkx as ntx
from lib.spectral_clustering import denormalized_spectral_clustering
from lib.spectral_clustering import normalized_spectral_clustering
from lib.spectral_clustering import random_walk_normalized_spectral_clustering
from lib.utils import draw_matrix
from lib.utils import sort_matrix
from lib.utils import to_adjacency_matrix


if __name__ == "__main__":

    karate_graph = ntx.karate_club_graph()
    karate_matrix = to_adjacency_matrix(karate_graph)
    draw_matrix(karate_matrix, "original.png")

    n_clusters = 2
    # ---------------------------------
    # denormalized sepectral clustering
    # ---------------------------------
    labels = denormalized_spectral_clustering(karate_matrix,
                                              n_clusters)
    new_karate_matrix, borders = sort_matrix(karate_matrix,
                                             labels,
                                             n_clusters)
    draw_matrix(new_karate_matrix, "denormalized.png",
                borders, borders)

    # ------------------------------
    # normalized spectral clustering
    # ------------------------------
    labels = normalized_spectral_clustering(karate_matrix,
                                            n_clusters)
    new_karate_matrix, borders = sort_matrix(karate_matrix,
                                             labels,
                                             n_clusters)
    draw_matrix(new_karate_matrix, "normalized.png",
                borders, borders)

    # ----------------------------------
    # random-walk normalizaed clustering
    # ----------------------------------
    labels = random_walk_normalized_spectral_clustering(karate_matrix,
                                                        n_clusters)
    new_karate_matrix, borders = sort_matrix(karate_matrix,
                                             labels,
                                             n_clusters)
    draw_matrix(new_karate_matrix, "random_walk_normalized.png",
                borders, borders)
