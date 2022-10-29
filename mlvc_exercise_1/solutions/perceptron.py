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
        predicted = []
        #output = np.dot(self.w *X)+ self.bias
        #y_pred = self.unit_step(output)
        #predicted.append(y_pred)

        #np.dot(self.w, X.T[1])

        output = np.dot(self.w.T, X)
        #for i, idx in enumerate(X):

            #idx has 9k entries
            #output = np.dot(idx, self.w)+ self.bias #w.T*X+Bias
            #y_pred = self.unit_step(output)
            #predicted.append(y_pred)
        #output = np.dot(self.w.T, X)

        activation = self.sigmoid(output)

        return activation


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


        self.bias = 0

        y_cor = np.array([1 if i > 0 else 0 for i in y])
        # Empty list to store how many examples were
        # misclassified at every iteration.
        miss_classifications = []
        
        # Training.
        for epoch in trange(self.epochs):
            #for idx, i in enumerate(X):
                #output = self.prec(x_i)




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
                error = y - prediction_for_update
                # update the weights of the perceptron from the random sample
                update = self.lr * error
                delta_w = np.dot(np.transpose(X, [1, 0]), update)
                #TODO self.w update
                self.w += delta_w
                self.bias +=  update


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
        output = np.dot(self.w.T, X)

        y_pred = self.sigmoid(output)

        return y_pred

    def sigmoid(self, X):
        return 1/(1+np.exp(-X))

    def unit_step(self, X):
        return np.where(X >= 0, 1, 0)



