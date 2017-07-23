import pickle
import numpy as np
from numpy import exp
from scipy.special import loggamma
from progressbar import ProgressBar


class InfiniteRelationalModel:
    """
    """
    def __init__(self, X, n_cluster_row=1, n_cluster_col=1,
                 alpha_k=0.5, alpha_l=0.5, a0=0.5, b0=0.5,
                 ):
        self.X = X
        self.Xt = X.T
        self.N1, self.N2 = X.shape
        self.K_init = n_cluster_row
        self.L_init = n_cluster_col

        self.alpha_k = alpha_k
        self.alpha_l = alpha_l
        self.a0 = a0
        self.b0 = b0

        self.K_ids = np.arange(n_cluster_row)
        self.L_ids = np.arange(n_cluster_col)

        self._is_ready = False

    def initialize(self, trained_file=None, seed=None):
        """
        """
        if isinstance(seed, int):
            np.random.seed(seed)
        if trained_file:
            pass
        else:
            Z1 = np.zeros((self.N1, self.K_init), dtype="float32")
            assigned_indexes = np.random.randint(0, self.K_init, self.N1)
            tmp_index = np.arange(self.N1)
            Z1[tmp_index, assigned_indexes] = 1.

            Z2 = np.zeros((self.N2, self.L_init), dtype="float32")
            assigned_indexes = np.random.randint(0, self.L_init, self.N2)
            tmp_index = np.arange(self.N2)
            Z2[tmp_index, assigned_indexes] = 1.

        self.Z1 = Z1
        self.Z2 = Z2
        self._is_ready = True

    def _process(self, steps, sample_step=None):
        """
        """
        sampled_Z1 = None
        sampled_Z2 = None
        sampled_K_ids = None
        sampled_L_ids = None
        if isinstance(sample_step, int):
            sampled_Z1 = list()
            sampled_Z2 = list()
            sampled_K_ids = list()
            sampled_L_ids = list()
        bar = ProgressBar()
        for step in bar(range(steps)):
            for i in range(self.N1):
                self._release(i, axis=0)
                p_z1, p_z1_new = self._sampling(i, axis=0)
                self._update(i, p_z1, p_z1_new, axis=0)
            for j in range(self.N2):
                self._release(j, axis=1)
                p_z2, p_z2_new = self._sampling(j, axis=1)
                self._update(j, p_z2, p_z2_new, axis=1)
            if isinstance(sample_step, int):
                # sample
                if step % sample_step == 0:
                    sampled_Z1.append(self.Z1)
                    sampled_Z2.append(self.Z2)
                    sampled_K_ids.append(self.K_ids)
                    sampled_L_ids.append(self.L_ids)
        return sampled_Z1, sampled_Z2, sampled_K_ids, sampled_L_ids

    def _release(self, index, axis):
        """
        """
        if axis == 0:
            table_count = self.Z1.sum(axis=0) - self.Z1[index]
            used = table_count > 0
            self.K_ids = self.K_ids[used]
            self.Z1 = self.Z1[:, used]
        elif axis == 1:
            table_count = self.Z2.sum(axis=0) - self.Z2[index]
            used = table_count > 0
            self.L_ids = self.L_ids[used]
            self.Z2 = self.Z2[:, used]
        else:
            raise ValueError("Wrong axis. Should be 0 or 1.")

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
        # alpha1_hat_k = alpha + M_r_hat
        a_hat = self.a0 + N_hat_pos
        b_hat = self.b0 + N_hat_neg

        # For existing clusters
        p_z_term1 = (loggamma(a_hat + b_hat) -
                     loggamma(a_hat) -
                     loggamma(b_hat))
        p_z_term2_nu = (loggamma(a_hat + sum_x_z2_pos) +
                        loggamma(b_hat + sum_x_z2_neg))
        p_z_term2_de = loggamma(a_hat + b_hat + M_c)
        p_z_exist = (M_r_hat *
                     exp(p_z_term1 +
                         p_z_term2_nu -
                         p_z_term2_de).prod(axis=1).real)

        # For new cluster
        p_z_term1 = (loggamma(self.a0 + self.b0) -
                     loggamma(self.a0) -
                     loggamma(self.b0))
        p_z_term2_nu = (loggamma(self.a0 + sum_x_z2_pos) +
                        loggamma(self.b0 + sum_x_z2_neg))
        p_z_term2_de = loggamma(self.a0 + self.b0 + M_c)
        p_z_new = (alpha *
                   exp(p_z_term1 +
                       p_z_term2_nu -
                       p_z_term2_de).prod(axis=0).real)
        total = p_z_exist.sum() + p_z_new
        p_z_exist /= total
        p_z_new /= total

        return p_z_exist, p_z_new

    def _update(self, index, p_z, p_z_new, axis):
        """
        """
        if axis == 0:
            p_z_tmp = np.concatenate([p_z, np.array([p_z_new])])
            p_z_tmp /= p_z_tmp.sum()
            new_id = self.K_ids.max()+1
            choices = self.K_ids.tolist() + [new_id]
            assigned_id = np.random.choice(choices, p=p_z_tmp)
            if assigned_id == new_id:
                new_Z1 = np.zeros((self.Z1.shape[0], 1), dtype="float32")
                self.Z1 = np.concatenate([self.Z1, new_Z1], axis=1)
                self.K_ids = np.array(choices)
            self.Z1[index] = 0.
            self.Z1[index][choices.index(assigned_id)] = 1.
        elif axis == 1:
            p_z_tmp = np.concatenate([p_z, np.array([p_z_new])])
            p_z_tmp /= p_z_tmp.sum()
            new_id = self.L_ids.max()+1
            choices = self.L_ids.tolist() + [new_id]
            assigned_id = np.random.choice(choices, p=p_z_tmp)
            if assigned_id == new_id:
                new_Z2 = np.zeros((self.Z2.shape[0], 1), dtype="float32")
                self.Z2 = np.concatenate([self.Z2, new_Z2], axis=1)
                self.L_ids = np.array(choices)
            self.Z2[index] = 0.
            self.Z2[index][choices.index(assigned_id)] = 1.
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
        Z1_counts = list()
        Z2_counts = list()
        Z1_ids = list()
        Z2_ids = list()
        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch, epochs))
            sampled_Z1, sampled_Z2, sampled_K_ids, sampled_L_ids = \
                self._process(n_sample, sample_step)
            Z1_count, Z1_id = self.merge_count(sampled_Z1,
                                               sampled_K_ids,
                                               self.N1)
            Z2_count, Z2_id = self.merge_count(sampled_Z2,
                                               sampled_L_ids,
                                               self.N2)
            Z1_counts.append(Z1_count)
            Z2_counts.append(Z2_count)
            Z1_ids.append(Z1_id)
            Z2_ids.append(Z2_id)

            # save samples into file
            if isinstance(results_file, str):
                results_file_name = "{}_{}.pkl".format(results_file, epoch)
                data = dict(
                    Z1_samples=sampled_Z1,
                    Z2_samples=sampled_Z2,
                    Z1_ids=sampled_K_ids,
                    Z2_ids=sampled_L_ids
                )
                pickle.dump(data, open(results_file_name, "wb"))

        Z1_all_count, Z1_all_id = self.merge_count(Z1_counts, Z1_ids, self.N1)
        Z2_all_count, Z2_all_id = self.merge_count(Z2_counts, Z2_ids, self.N2)

        Z1_labels = Z1_all_count.argmax(axis=1)
        Z2_labels = Z2_all_count.argmax(axis=1)

        return Z1_labels, Z2_labels

    @staticmethod
    def merge_count(sampled_Z, sampled_ids, nrow):
        """
        """
        unique_ids = list()
        for sampled_id in sampled_ids:
            for id_ in sampled_id:
                if id_ in unique_ids:
                    pass
                else:
                    unique_ids.append(id_)
        assigned_counts = np.zeros((nrow, max(unique_ids)+1))
        assigned_ids = np.arange(max(unique_ids)+1)
        for Z, ids in zip(sampled_Z, sampled_ids):
            assigned_counts[:, ids] += Z

        return assigned_counts, assigned_ids
