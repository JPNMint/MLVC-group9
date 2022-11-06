from random import randint

import numpy as np
from tqdm import trange


########### TO-DO ###########
# 1. Implement perceptron using the weights self.w
#   --> See: def perc(self,X): 
# 2. Implement perceptron update
#   --> See: self.w = self.w
# 3. Implement prediction
#   --> See: def predict(self, X):
#       - X is data

class Perceptron:
    """Implements the single layer perceptron.
    """

    def __init__(self, lr=0.5, epochs=100):
        """Initialize perceptron with learning rate and epochs.

        Args:
            lr (float): Learning rate
            epochs (int): Number of training epochs
            
        Attributes:
            w (numpy.ndarray): Weights of the perceptron

        """
        self.lr = lr
        self.epochs = epochs
        self.w =  None
        self.error = 0.

        self.bias = None
    def perc(self,X):
        """ Perceptron function.

        Args:
            X (numpy.ndarray): input vectors

        Returns:
            class labels of X
        """
        #Transpose weights for dot product between X and weights and also add bias
        output = np.dot(self.w.T, X) + self.bias
        #the output will then be fed into an activation function which gives us the final prediction
        #trying out different activation functions

        # Tried sign, sigmoid, ReLu and tanh
        #activation = np.sign(output)
        #activation = self.sigmoid(output)
        #activation = np.maximum(0, output) #ReLu
        activation = np.tanh(output)

        # tanh and sign performed best, I suspect that we have a lot of negative values which get turned to 0 when using the other
        # functions, while tanh and sign can return negative values up to -1
        #return activation
        return activation



    def fit(self, X, y):
        """ Training function.

        Args:
            X (numpy.ndarray): Inputs.
            y (numpy.ndarray): labels/target.

        Returns:
            List of the number of miss-classifications per epoch
        """

        # n_observations -> number of training examples n: 9k
        # m_features -> number of features  n: 16k
        n_observations, m_features = X.shape

        #Initialize weights with zero
        self.w = np.zeros(m_features)


        self.bias = 0

        y_cor = np.array([1 if i > 0 else 0 for i in y])
        # Empty list to store how many examples were
        # misclassified at every iteration.
        miss_classifications = []

        # Training.
        for epoch in trange(self.epochs):
            # predict all items from the dataset original
            predictions = self.perc(np.transpose(X, [1, 0]))
            # compare with gt
            predictions = y_cor - predictions
            if ((predictions == 0).all()):
                print(f'No errors after {epoch} epochs. Training successful!')
            else:
                #sample one prediction at random
                n = randint(0,n_observations-1)
                prediction_for_update = self.perc(X[n,:])
                # calculate error of random sample true - pred
                self.error = (y[n] - prediction_for_update)
                # update bias
                self.bias = self.bias + self.lr * self.error
                #update weights
                for i in range(m_features):
                    self.w[i] = self.w[i] + self.lr * self.error * X[n,:][i]




            # Appending number of misclassified examples
            # at every iteration.
            miss_classifications.append(predictions.shape[0] - np.sum(predictions==0))
            #print(miss_classifications)
        return miss_classifications


    def predict(self, X):
        """ Prediction function.

        Args:
            X (numpy.ndarray): Inputs.

        Returns:
            Class label of X
        """

        # basically same as fit function
        output = np.dot(self.w.T, np.transpose(X, [1, 0]))
        #tanh delivered the most promising result
        y_pred = np.tanh(output)
        #np.sign(output)
        #self.sigmoid(output)

        return y_pred
    # activation functions for fit and predict to try out
    def sigmoid(self, X):
        return 1/(1+np.exp(-X))

    def unit_step(self, X):
        return np.where(X >= 0, 1, 0)



