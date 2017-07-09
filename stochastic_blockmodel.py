import pickle
import numpy as np
from numpy import exp
from scipy.special import loggamma
from progressbar import ProgressBar
from spectral_clustering import to_adjacency_matrix


class StochasticBlockModel:
    """
    """
    def __init__(self, X, n_cluster_row=None, n_cluster_col=None):
        self.X = X
        self.Xt = X.T
        self.N1, self.N2 = X.shape
        self.K = n_cluster_row
        self.L = n_cluster_col

        self.alpha_k = np.array([1/self.K]*self.K)
        self.alpha_l = np.array([1/self.L]*self.L)
        self.a0 = 0.5
        self.b0 = 0.5

        self._is_ready = False

    def initialize(self, trained_file=None):
        """
        """
        if trained_file:
            pass
        else:
            Z1 = np.zeros((self.N1, self.K))
            Z1[:, :] = np.random.dirichlet([1/self.K]*self.K)
            Z2 = np.zeros((self.N2, self.L))
            Z2[:, :] = np.random.dirichlet([1/self.L]*self.L)

        self.Z1 = Z1
        self.Z2 = Z2
        self._is_ready = True

    def _process(self, steps, sample_step=None):
        """
        """
        sampled_Z1 = None
        sampled_Z2 = None
        if isinstance(sample_step, int):
            nsample = int(steps/sample_step)
            sampled_Z1 = np.zeros([nsample] + np.array(self.Z1.shape).tolist())
            sampled_Z2 = np.zeros([nsample] + np.array(self.Z2.shape).tolist())
        bar = ProgressBar()
        for step in bar(range(steps)):
            for i in range(self.N1):
                p_z1 = self._sampling(i, axis=0)
                self._update(i, p_z1, axis=0)
            for j in range(self.N2):
                p_z2 = self._sampling(j, axis=1)
                self._update(j, p_z2, axis=1)
            if isinstance(sample_step, int):
                # reserve sample
                if step % sample_step == 0:
                    sampled_Z1[int(step/sample_step)] = self.Z1
                    sampled_Z2[int(step/sample_step)] = self.Z2
        return sampled_Z1, sampled_Z2

    def _sampling(self, index, axis):
        """
        """
        if axis == 0:
            X = self.X
            Z_r = self.Z1
            Z_c = self.Z2
            M_r = np.sum(self.Z1, axis=0)
            M_c = np.sum(self.Z2, axis=0)
            alpha = self.alpha_k
        elif axis == 1:
            X = self.Xt
            Z_r = self.Z2
            Z_c = self.Z1
            M_r = np.sum(self.Z2, axis=0)
            M_c = np.sum(self.Z1, axis=0)
            alpha = self.alpha_l
        else:
            raise ValueError("Wrong axis. Should be 0 or 1.")

        N_pos = np.dot(np.dot(X.T, Z_r).T, Z_c)
        N_neg = np.dot(np.dot((1 - X).T, Z_r).T, Z_c)

        M_r_hat = M_r - Z_r[index]
        sum_x_z2_pos = np.dot(X[index], Z_c)
        N_hat_pos = N_pos - Z_r[index, np.newaxis].T * sum_x_z2_pos
        sum_x_z2_neg = np.dot((1 - X[index]), Z_c)
        N_hat_neg = N_neg - Z_r[index, np.newaxis].T * sum_x_z2_neg
        alpha1_hat_k = alpha + M_r_hat
        a_hat = self.a0 + N_hat_pos
        b_hat = self.b0 + N_hat_neg

        p_z_term1 = (loggamma(a_hat + b_hat) -
                     loggamma(a_hat) -
                     loggamma(b_hat))
        p_z_term2_nu = (loggamma(a_hat + sum_x_z2_pos) +
                        loggamma(b_hat + sum_x_z2_neg))
        p_z_term2_de = loggamma(a_hat + b_hat + M_c)
        p_z = (alpha1_hat_k *
               exp(p_z_term1 +
                   p_z_term2_nu -
                   p_z_term2_de).prod(axis=1).real)
        p_z /= p_z.sum()

        return p_z

        # M1 = np.sum(self.Z1, axis=0)
        # M2 = np.sum(self.Z2, axis=0)
        # n_pos = np.einsum("ikjl, ij", np.tensordot(self.Z1, self.Z2, axes=0), self.X)
        # n_neg = np.einsum("ikjl, ij", np.tensordot(self.Z1, self.Z2, axes=0), 1 - self.X)
        # m1_hat = lambda i: M1 - self.Z1[i]
        # n_pos_hat = lambda i: n_pos - np.einsum("kjl, j", np.tensordot(self.Z1, self.Z2, axes=0)[i], self.X[i])
        # n_neg_hat = lambda i: n_neg - np.einsum("kjl, j", np.tensordot(self.Z1, self.Z2, axes=0)[i], 1 - self.X[i])
        # alpha_1_hat = lambda i: self.alpha_k + m1_hat(i)
        # a_hat = lambda i: self.a0 + n_pos_hat(i)
        # b_hat = lambda i: self.b0 + n_neg_hat(i)
        # aihat = a_hat(index)
        # bihat = b_hat(index)
        #
        # p_z1i_left = loggamma(aihat + bihat) - loggamma(aihat) - loggamma(bihat)
        # p_z1i_right_upper = loggamma(aihat + np.dot(self.X[index], self.Z2)) + loggamma(bihat + np.dot((1 - self.X[index]), self.Z2))
        # p_z1i_right_lower = loggamma(aihat + bihat + M2)
        # p_z1i = alpha_1_hat(index) * (exp(p_z1i_left + p_z1i_right_upper - p_z1i_right_lower)).prod(axis=1)
        # p_z1i = p_z1i.real
        # p_z1i = p_z1i / p_z1i.sum()

    def _update(self, index, p_z, axis):
        """
        """
        if axis == 0:
            self.Z1[index] = p_z
        elif axis == 1:
            self.Z2[index] = p_z
        else:
            raise ValueError("Wrong axis. Should be 0 or 1.")

    def fit(self, burn_in=None, epochs=1, n_sample=100, sample_step=1,
            results_file=None):
        """
        """
        if not self._is_ready:
            raise ValueError("Need to initialize before fit.")

        if isinstance(burn_in, int):
            print("Burning before sampling.")
            self._process(burn_in, None)

        print("Sampling.")
        Z1_means = np.zeros([epochs] + np.array(self.Z1.shape).tolist())
        Z2_means = np.zeros([epochs] + np.array(self.Z2.shape).tolist())
        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch, epochs))
            sampled_Z1, sampled_Z2 = self._process(n_sample, sample_step)
            Z1_means[epoch] = sampled_Z1.mean(axis=0)
            Z2_means[epoch] = sampled_Z2.mean(axis=0)

            # save samples into file
            if isinstance(results_file, str):
                results_file_name = "{}_{}.pkl".format(results_file, epoch)
                data = dict(
                    Z1_samples=sampled_Z1,
                    Z2_samples=sampled_Z2
                )
                pickle.dump(data, open(results_file_name, "wb"))

        return Z1_means, Z2_means


if __name__ == "__main__":
    import networkx as ntx
    karate_graph = ntx.karate_club_graph()
    karate_matrix = to_adjacency_matrix(karate_graph)
    # karate_matrix = np.random.normal([[1]*6]*7)

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
