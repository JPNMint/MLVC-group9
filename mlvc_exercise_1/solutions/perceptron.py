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

        #add bias
        self.bias = None
    def perc(self,X):
        """ Perceptron function.

        Args:
            X (numpy.ndarray): input vectors

        Returns:
            class labels of X
        """
        #print("in perc")
        output = np.dot(self.w, X)
        #activation = self.sigmoid(output)
        act = np.where(output > 0, 1, 0)
        return act


        # for ix, i in enumerate(X):
        #     # TODO
        #     output = self.predict(i)
        #     y_pred = self.sigmoid(output)

    
    def fit(self, X, y):
        """ Training function.

        Args:
            X (numpy.ndarray): Inputs.
            y (numpy.ndarray): labels/target.

        Returns:
            List of the number of miss-classifications per epoch
        """

        # n_observations -> number of training examples 9k
        # m_features -> number of features  16k
        n_observations, m_features = X.shape

        #Initialize weights with zero
        self.w = np.zeros(m_features)


        self.bias = 1
        
        # Empty list to store how many examples were 
        # misclassified at every iteration.
        miss_classifications = []
        
        # Training.
        for epoch in trange(self.epochs):

            # predict all items from the dataset
            predictions = self.perc(np.transpose(X, [1, 0]))
            # compare with gt
            predictions = y - predictions
            if ((predictions == 0).all()):
                print(f'No errors after {epoch} epochs. Training successful!')
            else:
                #sample one prediction at random
                n = randint(0,n_observations-1)
                prediction_for_update = self.perc(X[n,:])
                error = y - prediction_for_update
                # update the weights of the perceptron from the random sample
                #TODO self.w update
                self.w = self.w + self.lr * np.dot(X.T, error)


                #self.w += np.dot(error,X)*self.lr # to be corrected by you


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

        return self.sigmoid(np.dot(X, self.w))

    def sigmoid(self, X):
        return 1/(1+np.exp(-X))


