import networkx as ntx
from lib.stochastic_blockmodel import StochasticBlockModel
from lib.utils import to_adjacency_matrix


if __name__ == "__main__":
    karate_graph = ntx.karate_club_graph()
    karate_matrix = to_adjacency_matrix(karate_graph)

    n_cluster_row = 3
    n_cluster_col = 3
    sbm = StochasticBlockModel(karate_matrix,
                               n_cluster_row,
                               n_cluster_col)
    sbm.initialize()
    Z1_means, Z2_means = sbm.fit(burn_in=100,
                                 n_sample=1000,
                                 sample_step=10,
                                 results_file="samples")
