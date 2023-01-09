########### TO-DO ###########
# 1. Implement gaussian_process without noise
#   --> See: gaussian_process(X1, y1, X2, kernel_func):
# 2. Implement gaussian_process WITH noise
#   --> See: gaussian_process_noise(X1, y1, X2, kernel_func, n1, sigma_noise):
import scipy
import numpy as np
def gaussian_process(X1, y1, X2, kernel_func):
    """
    Calculate the posterior mean and covariance matrix for y2
    based on the corresponding input X2, the observations (y1, X1), 
    and the prior kernel function.

    Args:
        X1 (numpy.ndarray): Training points
        y1 (numpy.ndarray): Labels (targets in lecture) of X1
        X2 (numpy.ndarray): Test points
        kernel_func (sklearn.gaussian_process.kernels or custom): Kernel function

    Returns:
        μ2 (numpy.ndarray): Posterior mean
        covariance2 (numpy.ndarray): Posterior covariance

    """
    #Kernel of X1 X1
    kernel_output_X1_X1 = kernel_func(X1,X1)

    #Kernel of original and test
    kernel_output_X1_X2 = kernel_func(X1,X2)

    # Solve
    result = scipy.linalg.solve(kernel_output_X1_X1, kernel_output_X1_X2, assume_a='pos').T

    # Posterior mean
    μ2 = np.matmul(result,y1)
    kernel_output_X2_X2 = kernel_func(X2, X2)
    # Posterior covariance
    covariance2 = kernel_output_X2_X2 - np.matmul(result,kernel_output_X1_X2)
    return μ2, covariance2  # mean, covariance


# Gaussian process posterior with noisy obeservations
def gaussian_process_noise(X1, y1, X2, kernel_func, n1, sigma_noise):
    """
    Calculate the posterior mean and covariance matrix for y2
    based on the corresponding input X2, the noisy observations 
    (y1, X1), the prior kernel function and the number of training samples (n1).

    Args:
        X1 (numpy.ndarray): Training points
        y1 (numpy.ndarray): Labels (targets in lecture) of X1
        X2 (numpy.ndarray): Test points
        kernel_func (sklearn.gaussian_process.kernels or custom): Kernel function
        n1 (int): len(X1)
        sigma_noise (float): Noise, standard deviation

    Returns:
        μ2 (numpy.ndarray): Posterior mean
        covariance2 (numpy.ndarray): Posterior covariance
    """
    # Kernel of X1 X1 and noise
    kernel_output_X1_X1 = kernel_func(X1, X1) + ((sigma_noise ** 2) * np.eye(n1))

    #Kernel of original and test
    kernel_output_X1_X2 = kernel_func(X1, X2)

    # Solve
    result = scipy.linalg.solve(kernel_output_X1_X1, kernel_output_X1_X2, assume_a='pos').T

    # Posterior mean
    μ2 = np.matmul(result,y1)
    kernel_output_X2_X2 = kernel_func(X2, X2)
    # Posterior covariance
    covariance2 = kernel_output_X2_X2 - np.matmul(result,kernel_output_X1_X2)


    return μ2, covariance2  # mean, covariance
