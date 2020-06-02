"""
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np
import math


def initialize_variational_parameters(num_rows_of_image, num_cols_of_image, K):
    """ Helper function to initialize variational distributions before each E-step.
    Args:
                num_rows_of_image: Integer representing the number of rows in the image
                num_cols_of_image: Integer representing the number of columns in the image
                K: The number of latent states in the MRF
    Returns:
                q: 3-dimensional numpy matrix with shape [num_rows_of_image, num_cols_of_image, K]
     """
    q = np.random.random((num_rows_of_image, num_cols_of_image, K))
    for row_num in range(num_rows_of_image):
        for col_num in range(num_cols_of_image):
            q[row_num, col_num, :] = q[row_num, col_num, :] / sum(q[row_num, col_num, :])
    return q


def initialize_theta_parameters(K):
    """ Helper function to initialize theta before begining of EM.
    Args:
                K: The number of latent states in the MRF
    Returns:
                mu: A numpy vector of dimension [K] representing the mean for each of the K classes
                sigma: A numpy vector of dimension [K] representing the standard deviation for each of the K classes
    """
    mu = np.zeros(K)
    sigma = np.zeros(K) + 10
    for k in range(K):
        mu[k] = np.random.randint(10, 240)
    return mu, sigma


class MRF(object):
    def __init__(self, J, K, n_em_iter, n_vi_iter):
        self.J = J
        self.K = K
        self.n_em_iter = n_em_iter
        self.n_vi_iter = n_vi_iter
        self.q = None
        self.mu = None
        self.sigma = None
        self.X = None

    def fit(self, *, X):
        self.X = X
        """ Fit the model.
                Args:
                X: A matrix of floats with shape [num_rows_of_image, num_cols_of_image]
        """
        # Please use helper function 'initialize_theta_parameters' to initialize theta at the start of EM
        #     Ex:  mu, sigma = initialize_theta_parameters(self.K)
        # Please use helper function 'initialize_variational_parameters' to initialize q at the start of each E step
        #     Ex:  q = initialize_variational_parameters(X.shape[0], X.shape[1], self.K)
        self.mu, self.sigma = initialize_theta_parameters(self.K)
        for _ in range(self.n_em_iter):
            self.E_step()
            self.M_step()

        self.E_step()

    def predict(self, X):
        """ Predict.
        Args:
                X: A matrix of floats with shape [num_rows_of_image, num_cols_of_image]

        Returns:
                A matrix of ints with shape [num_rows_of_image, num_cols_of_image].
                    - Each element of this matrix should be the most likely state according to the trained model for the pixel corresponding to that row and column
                    - States should be encoded as {0,..,K-1}
        """
        return np.argmax(self.q, 2)

    def E_step(self):
        self.q = initialize_variational_parameters(self.X.shape[0], self.X.shape[1], self.K)
        for _ in range(self.n_vi_iter):
            for i in range(self.X.shape[0]):
                for j in range(self.X.shape[1]):
                    N_s = self.detect(i, j)
                    for k in range(self.K):
                        self.q[i, j, k] = self.calq(i, j, k, N_s)

    def M_step(self):
        for k in range(self.K):
            nomi_mu = 0
            deno_ms = 0
            for row_num in range(self.X.shape[0]):
                for col_num in range(self.X.shape[1]):
                    nomi_mu += self.q[row_num, col_num, k] * self.X[row_num][col_num]
                    deno_ms += self.q[row_num, col_num, k]
            self.mu[k] = nomi_mu / deno_ms
            nomi_sigma = 0
            for row_Num in range(self.X.shape[0]):
                for col_Num in range(self.X.shape[1]):
                    nomi_sigma += self.q[row_Num, col_Num, k] * ((self.X[row_Num][col_Num] - self.mu[k]) ** 2)
            self.sigma[k] = math.sqrt(nomi_sigma / deno_ms)

    def calq(self, row, col, k, neighbour):
        nomi = self.gaus(self.X[row][col], k) * np.exp(self.sum_neighbour(neighbour, k))
        denomi = 0
        for k_justify in range(self.K):
            denomi += self.gaus(self.X[row][col], k_justify) * np.exp(self.sum_neighbour(neighbour, k_justify))
        return nomi / denomi

    def gaus(self, x, k):
        return (1 / (math.sqrt(2 * math.pi) * self.sigma[k])) * np.exp(
            -0.5 * (x - self.mu[k]) ** 2 / self.sigma[k] ** 2)

    def detect(self, row, col):
        N_s = []
        if row + 1 <= self.X.shape[0] - 1:
            N_s.append((row + 1, col))
        if row - 1 >= 0:
            N_s.append((row - 1, col))
        if col + 1 <= self.X.shape[1] - 1:
            N_s.append((row, col + 1))
        if col - 1 >= 0:
            N_s.append((row, col - 1))
        return N_s

    def sum_neighbour(self, neighbour, k):
        sum_NS = 0
        for ns in neighbour:
            sum_NS += self.J * self.q[ns[0], ns[1], k]
        return sum_NS
