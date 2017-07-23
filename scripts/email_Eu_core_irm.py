import numpy as np
from lib.infinite_relational_model import InfiniteRelationalModel
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

    irm = InfiniteRelationalModel(email_data)
    irm.initialize(seed=0)
    row_labels, col_labels = irm.fit(burn_in=100,
                                     n_sample=500,
                                     sample_step=1,
                                     results_file="samples")

    sort1_matrix, borders1 = sort_matrix(email_data,
                                         row_labels,
                                         axis="y")
    sort2_matrix, borders2 = sort_matrix(sort1_matrix,
                                         col_labels,
                                         axis="x")
    draw_matrix(sort2_matrix, "email_irm_sort.png", borders2, borders1)
