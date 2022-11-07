import numpy as np
import random

class SVM():
    def __init__(self, max_iter=20, C=2, e=0.0002):

        self.max_iter = max_iter
        self.C = C
        self.epsilon = e
        self.kernel = self.gaussian_kernel_function

    def fit(self, X, y):

        print("Setup --- Gaussian kernel --- C: " + str(self.C) + " --- epsilon: " + str(self.epsilon) + "\n")
        n_observations, m_features = X.shape
        alpha_vector = np.zeros(n_observations)
        iter = 0

        while True:

            iter += 1
            print("Iteration " + str(iter) + " - (max iterations: " + str(self.max_iter) + ")")

            # Saving previous alpha values
            alpha_vector_old = np.copy(alpha_vector)


            for j in range(0, n_observations):

                if j%30 == 0:
                    print("step: " + str(j/30) + " of " + str(n_observations/30))

                i = self.generate_random_number(n_observations, j)

                k_ij = self.compute_eta(X[i, :], X[j, :])

                if k_ij == 0:
                    continue


                L = self.compute_L(y[i], y[j], alpha_vector[i], alpha_vector[j])
                H = self.compute_H(y[i], y[j], alpha_vector[i], alpha_vector[j])


                # Compute model parameters - bottleneck in the algorithm
                self.w = self.compute_w(alpha_vector, y, X)
                self.b = self.compute_b(X, y, self.w)


                # compute errors
                error_i = self.compute_error(X[i, :], y[i], self.w, self.b)
                error_j = self.compute_error(X[j, :], y[j], self.w, self.b)



                # compute new alpha values
                alpha_j_prev = alpha_vector[j]
                alpha_vector[j] = self.compute_new_alpha_j(alpha_vector[j],
                                                           y[j], error_i,
                                                           error_j, k_ij, L, H)

                alpha_vector[i] = self.compute_new_alpha_i(alpha_vector[j],
                                                           alpha_j_prev,
                                                           alpha_vector[i],
                                                           y[j], y[i])




            # Compute the norm of the difference of our old and new alpha vector
            # this is needed for checking for convergence
            norm_of_difference = np.linalg.norm(alpha_vector - alpha_vector_old)

            print("Current norm value: " + str(norm_of_difference) + " algorithm stops at " + str(self.epsilon) + " \n")

            # checking if the algorithm is converging
            if self.epsilon > norm_of_difference:
                break

            if self.max_iter <= iter:
                print("Maximum number of iterations reached!")
                return

        # Compute final w and b after convergence
        self.w = self.compute_w(alpha_vector, y, X)
        self.b = self.compute_b(X, y, self.w)


    def predict(self, X):
        return self.compute_predictions(X, self.w, self.b)


    ############## Helper functions ##############

    def compute_new_alpha_i(self, alpha_j, alpha_prev_j , alpha_i, y_j, y_i):
        return alpha_i + y_i * y_j * (alpha_prev_j - alpha_j)

    def compute_new_alpha_j(self, alpha_j, y_j, error_i, error_j, k_ij, L, H ):

        tmp = alpha_j + float(y_j * (error_i - error_j)) / k_ij

        tmp = max(tmp, L)
        final_alpha_value = min(tmp, H)

        return final_alpha_value

    def compute_predictions(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b)

    def compute_error(self, x_k, y_k, w, b):
        return self.compute_predictions(x_k, w, b) - y_k

    def compute_b(self, X_mtx, y_vector, w_vector):

        tmp = y_vector - np.dot(w_vector.T, X_mtx.T)

        return np.mean(tmp)

    def compute_w(self, alpha_vector, y_vector, X_mtx):

        tmp = np.multiply(alpha_vector, y_vector)

        return np.dot(X_mtx.T, tmp)

    def compute_L(self, y_i, y_j, alpha_i, alpha_j):

        if y_i != y_j:
            return max(0, alpha_j - alpha_i)
        else:
            return max(0, alpha_j + alpha_i - self.C)

    def compute_H(self, y_i, y_j, alpha_i, alpha_j):

        if y_i != y_j:
            return min(self.C, self.C + alpha_j - alpha_i)
        else:
            return min(self.C, alpha_j + alpha_i)


    def generate_random_number(self, interval, j):

        i = random.randint(0, interval - 1)

        while i == j:
            i = random.randint(0, interval - 1)

        return i

    def compute_eta(self, x_i, x_j):
        return -2 * self.kernel(x_i, x_j) \
               + self.kernel(x_i, x_i) \
               + self.kernel(x_j, x_j)

    def gaussian_kernel_function(self, x1, x2, sigma=1):
        return np.exp(-1* (np.linalg.norm(x1 - x2, 2)) ** 2 / (2 * sigma ** 2))
