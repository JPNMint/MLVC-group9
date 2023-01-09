import matplotlib.pyplot as plt
import numpy as np
import scipy


# Define the exponentiated quadratic
def exponentiated_quadratic(xa, xb):
    """Exponentiated quadratic  with sigma=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)


def plot_gp_results(f_sin, X1, X2, y1, y2, domain, μ2, sigma2):
    """Plot the postior distribution and some samples"""

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
    # Plot the distribution of the function (mean, covariance)
    ax1.plot(X2, f_sin(X2), 'b--', label='$sin(x)$')
    ax1.fill_between(X2.flat,
                     μ2 - 2 * sigma2,
                     μ2 + 2 * sigma2,
                     color='red',
                     alpha=0.15,
                     label='$2\sigma_{2|1}$')
    ax1.plot(X2, μ2, 'r-', lw=2, label='$\mu_{2|1}$')
    ax1.plot(X1, y1, 'ko', linewidth=2, label='$(x_1, y_1)$')
    ax1.set_xlabel('$x$', fontsize=13)
    ax1.set_ylabel('$y$', fontsize=13)
    ax1.set_title('Distribution of posterior and prior data')
    ax1.axis([
        domain[0], domain[1],
        np.min(f_sin(X2)) - (0.5 * np.max(f_sin(X2))),
        np.max(f_sin(X2)) + (0.5 * np.max(f_sin(X2)))
    ])
    ax1.legend()
    # Plot some samples from this function
    ax2.plot(X2, y2.T, '-')
    ax2.set_xlabel('$x$', fontsize=13)
    ax2.set_ylabel('$y$', fontsize=13)
    ax2.set_title('5 different function realizations from posterior')
    ax2.axis([
        domain[0], domain[1],
        np.min(f_sin(X2)) - (0.5 * np.max(f_sin(X2))),
        np.max(f_sin(X2)) + (0.5 * np.max(f_sin(X2)))
    ])
    #ax2.set_xlim([-6, 6])
    plt.tight_layout()
    plt.show()
