import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


if __name__ == "__main__":
    filename = "samples_0.pkl"
    data = pickle.load(open(filename, "rb"))
    keys = list(data.keys())

    pdfname = "z_probs.pdf"
    pdf_pages = PdfPages(pdfname)
    figs = plt.figure(figsize=(8.27, 11.69), dpi=100)
    n_plot = 1
    n_page = 1
    for key in keys:
        print("Printing {}".format(key))
        Z = data[key]
        n_sample, n_node, n_class = Z.shape
        for node in range(n_node):
            x = np.arange(n_sample)
            ys = data[key][:, node, :]

            sub_fig = figs.add_subplot(5, 1, n_plot)
            sub_fig.set_title("{} node{}".format(key, node))
            sub_fig.grid()
            for i_class in range(n_class):
                y = ys[:, i_class]
                sub_fig.plot(x, y)
            n_plot += 1

            if n_plot > 5:
                if n_page == 1:
                    figs.tight_layout()
                pdf_pages.savefig(figs)
                figs.clear()
                n_plot = 1
    if n_plot > 1:
        if n_page == 1:
            figs.tight_layout()
        pdf_pages.savefig(figs)
    pdf_pages.close()
