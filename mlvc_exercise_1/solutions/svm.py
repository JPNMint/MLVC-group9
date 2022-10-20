import numpy as np

########### TO-DO ###########
# 1. Implement fit
#   --> See: def fit(self, X, y):
# 2. Implement predict
#   --> See: def predict(self, X):


class SVM:
    """Implements the support vector machine"""
    def __init__(self):
        """Initialize perceptron.
        """
        pass

    def fit(self, X, y):
        """ Training function.

        Args:
            X (numpy.ndarray): Inputs.
            y (numpy.ndarray): labels/target.

        Returns:
            None
        """
        # n_observations -> number of training examples
        # m_features -> number of features 
        n_observations, m_features = X.shape

    def predict(self, X):
        """ Prediction function.

        Args:
            X (numpy.ndarray): Inputs.

        Returns:
            Class label of X
        """

        return
