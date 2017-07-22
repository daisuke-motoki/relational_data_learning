import pickle
import numpy as np
from numpy import exp
from scipy.special import loggamma
from progressbar import ProgressBar


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

    def initialize(self, trained_file=None, seed=None):
        """
        """
        if isinstance(seed, int):
            np.random.seed(seed)
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
                # sample
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

        Z1_means = Z1_means.mean(axis=0)
        Z2_means = Z2_means.mean(axis=0)

        Z1_labels = Z1_means.argmax(axis=1)
        Z2_labels = Z2_means.argmax(axis=1)

        return Z1_labels, Z2_labels
