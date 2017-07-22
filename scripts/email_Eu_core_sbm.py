import numpy as np
from lib.stochastic_blockmodel import StochasticBlockModel
from lib.utils import draw_matrix
from lib.utils import sort_matrix


def read_email_data(data_file, shape):
    """
    """
    data = np.zeros(shape, dtype=np.int)
    # with open(data_file, "r") as file_:
    for line in open(data_file, "r"):
        # line = file_.readline()
        from_, to_ = line.replace("\n", "").split(" ")
        from_, to_ = int(from_), int(to_)
        if from_ < shape[0] and to_ < shape[1]:
            data[from_][to_] += 1

    return data


if __name__ == "__main__":
    data_file = "../data/email-Eu-core.txt"
    email_data = read_email_data(data_file, (100, 100))

    n_cluster_row = 41
    n_cluster_col = 41
    sbm = StochasticBlockModel(email_data,
                               n_cluster_row,
                               n_cluster_col)
    sbm.initialize(seed=0)
    row_labels, col_labels = sbm.fit(burn_in=100,
                                     n_sample=200,
                                     sample_step=1,
                                     results_file="samples")
    sort1_matrix, borders1 = sort_matrix(email_data,
                                         row_labels,
                                         n_cluster_row,
                                         axis="y")
    sort2_matrix, borders2 = sort_matrix(sort1_matrix,
                                         col_labels,
                                         n_cluster_col,
                                         axis="x")
    draw_matrix(sort2_matrix, "email_sbm_sort.png", borders2, borders1)
