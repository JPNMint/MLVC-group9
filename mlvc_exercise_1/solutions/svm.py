import torch as np
import random


class SVM():

    # In the constructor we can set several hyperparameters
    # Maximum number of Iterations
    # C - upper boundary for alpha values
    # epsilon - threshold value to tell if algorithm converged

    def __init__(self, max_iter=40, C=2, e=0.0002):

        self.max_iter = max_iter
        self.C = C
        self.epsilon = e
        self.device = "cuda" if np.cuda.is_available() else "cpu"

        # We decided to use a gaussian kernel
        # we could also use a different kernel function here
        self.kernel = self.gaussian_kernel_function

    def fit(self, X, y):
        device = self.device

        print("Setup --- Gaussian kernel --- C: " + str(self.C) + " --- epsilon: " + str(self.epsilon) + "\n")
       
        X = np.from_numpy(X).to(device)
        y = np.from_numpy(y).to(device)
        
        n_observations, m_features = X.shape
        alpha_vector = np.zeros(n_observations).to(device)
        iter = 0

        # SMO (Sequential Minimal Optimization)

        # The idea of SMO is to optimize two variables at a time (i,j) instead of
        # optimizing all the variables at once
        # This leads to 1D quadratic optimization problems which are easy and fast to solve

        while True:

            # We iterate till convergence or max_iter is reached
            iter += 1
            print("Iteration " + str(iter) + " - (max iterations: " + str(self.max_iter) + ")")

            # Saving previous alpha values
            alpha_vector_old = np.clone(alpha_vector)

            for j in range(0, n_observations):

                #if j%900 == 0:
                #    print("step: " + str(j/900) + " of " + str(n_observations/900))

                i = self.generate_random_number(n_observations, j)

                # eta helps us to decide if we want to continue to the next iteration
                k_ij = self.compute_eta(X[i, :], X[j, :])

                if k_ij == 0:
                    continue


                # We compute L and H
                # Alpha j will be bounded between L and H
                # Then alpha i will be between 0 and C
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
            # this is needed for checking for convergenc
            sub = np.sub(alpha_vector, alpha_vector_old)
            norm_of_difference = np.linalg.norm(sub)

            print("Current norm value: " + str(norm_of_difference) + " algorithm stops at " + str(self.epsilon) + " \n")

            # checking if the algorithm is converging
            if self.epsilon > norm_of_difference:
                break

            # stop if algorithm reacht max iter
            if self.max_iter <= iter:
                print("Maximum number of iterations reached!")
                return

        # Compute final w and b after convergence
        self.w = self.compute_w(alpha_vector, y, X)
        self.b = self.compute_b(X, y, self.w)


    # Here we predict the results for any test set
    def predict(self, X):
        device = self.device

        X = np.from_numpy(X).to(device)
        final = self.compute_predictions(X, self.w, self.b)
        final = final.cpu()
        final = final.numpy()


        return final

    ############## Helper functions ##############

    # Function for computing the alpha i's
    def compute_new_alpha_i(self, alpha_j, alpha_prev_j , alpha_i, y_j, y_i):
        return alpha_i + y_i * y_j * (alpha_prev_j - alpha_j)

    # Function for computing the alpha j's
    def compute_new_alpha_j(self, alpha_j, y_j, error_i, error_j, k_ij, L, H ):

        tmp = alpha_j + float(y_j * (error_i - error_j)) / k_ij

        tmp = max(tmp, L)
        final_alpha_value = min(tmp, H)

        return final_alpha_value

    # Function for computing the predictions
    def compute_predictions(self, X, w, b):
        return np.sign(np.matmul(w.T.float(), X.T.float()) + b)

    # Function for computing the error
    def compute_error(self, x_k, y_k, w, b):
        return self.compute_predictions(x_k, w, b) - y_k

    # Function for computing the bias
    def compute_b(self, X_mtx, y_vector, w_vector):

        tmp = y_vector - np.matmul(w_vector.T.float(), X_mtx.T.float())

        return np.mean(tmp.float())

    # Function for computing the weights
    def compute_w(self, alpha_vector, y_vector, X_mtx):

        tmp = np.multiply(alpha_vector, y_vector)

        return np.matmul(X_mtx.T.float(), tmp.float())

    # Function for computing the lower bound for alpha j
    def compute_L(self, y_i, y_j, alpha_i, alpha_j):

        if y_i != y_j:
            return max(0, alpha_j - alpha_i)
        else:
            return max(0, alpha_j + alpha_i - self.C)

    # Function for computing the upper bound for alpha j
    def compute_H(self, y_i, y_j, alpha_i, alpha_j):

        if y_i != y_j:
            return min(self.C, self.C + alpha_j - alpha_i)
        else:
            return min(self.C, alpha_j + alpha_i)


    # Function for creating a random number to choose random i
    def generate_random_number(self, interval, j):

        i = random.randint(0, interval - 1)

        while i == j:
            i = random.randint(0, interval - 1)

        return i

    # Function for computing eta
    def compute_eta(self, x_i, x_j):
        return -2 * self.kernel(x_i, x_j) \
               + self.kernel(x_i, x_i) \
               + self.kernel(x_j, x_j)

    # The kernel function we decided to use - Gaussian Kernel
    def gaussian_kernel_function(self, x1, x2, sigma=1):

        tmp = np.sub(x1, x2).float()
        
        return np.exp(-1* (np.linalg.norm(tmp, 2)) ** 2 / (2 * sigma ** 2))
