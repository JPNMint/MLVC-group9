import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

########### TO-DO ###########
# 1. Add cross entropy loss:
#   --> See: class SquareLoss(Loss)
# 2. Add activation function (SIGMOID)
#   --> See: self.activation_functions AND self.activ_derivative
#   --> Tanh is given as example
# 3. Implement forward pass
#   --> See: def fit()
# 4. Implement backpropagation and weight updates
#   --> See: def backprop()
# 5. Impement prediction
#   --> See: def predict(self, X, y):
#       - X is data
#       - y is label

class SquareLoss(object):
    """A loss function and its gradient."""

    def loss(self, y, y_pred):
        """ Returns the loss of the predictions y_pred when y are their corresponding values.

        Args:
            y (numpy.ndarray): --> label/target.
            y_pred (numpy.ndarray): --> prediction.

        Returns:
            Returns the loss
        """
        return 0.5 * np.power((y - y_pred), 2)

    def delta(self, y, y_pred):
        """ Calculates the exterior derivative for the loss.

        Args:
            y (numpy.ndarray): --> label/target.
            y_pred (numpy.ndarray): --> prediction.

        Returns:
            Returns the loss delta
        """
        return -(y - y_pred)

    def acc(self, y, y_pred):
        """ Calculates the accuracy (percentage of the correctly classified samples).

        Args:
            y (numpy.ndarray): --> label/target.
            y_pred (numpy.ndarray): --> prediction.

        Returns:
            Returns the accuracy
        """
        y_pred = y_pred[:,0]
        assert y.ndim == 1 and y.size == y_pred.size
        y_pred = y_pred > 0.5
        return (y == y_pred).sum().item() / y.size
    
class CrossEntropyLoss(object):
    """A loss function and its gradient."""

    def loss(self, y, y_pred):
        """ Returns the loss of the predictions y_pred when y are their corresponding values.

        Args:
            y (numpy.ndarray): --> label/target.
            y_pred (numpy.ndarray): --> prediction.

        Returns:
            Returns the loss
        """
        return -(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))

    def delta(self, y, y_pred):
        """ Calculates the exterior derivative for the loss.

        Args:
            y (numpy.ndarray): --> label/target.
            y_pred (numpy.ndarray): --> prediction.

        Returns:
            Returns the loss delta
        """
        return -((y/y_pred) - ((1-y)/(1-y_pred)))

    def acc(self, y, y_pred):
        """ Calculates the accuracy (percentage of the correctly classified samples).

        Args:
            y (numpy.ndarray): --> label/target.
            y_pred (numpy.ndarray): --> prediction.

        Returns:
            Returns the accuracy
        """
        y_pred = y_pred[:,0]
        assert y.ndim == 1 and y.size == y_pred.size
        y_pred = y_pred > 0.5
        return (y == y_pred).sum().item() / y.size

class MultiLayerPerceptron():
    """Implements a three layer perceptron with variable amount of features"""
    def __init__(self, input_dim=16384, hidden_dim=10, hidden_dim2=4, output_dim=1, lr=0.005, epochs=100, activation="sigmoid", loss="square", classes=2):
        """Initialize MLP.

        Args:
            input_dim (int): input dimension of the flattened input (128 x 128) --> 16384
            hidden_dim (int): first hidden layer output dimension
            hidden_dim2 (int): second hidden layer output dimension
            output_dim (int): output dimension of the last layer
            lr (float): learning rate
            epochs (int): number of epochs
            activation (string): Activation function (sigmoid or tanh)
            classes (int): Number of classes in the dataset

            
        Additional Attributes:
            w (numpy.ndarray): Weights of the perceptron
            activation_functions (dict): A dict with possible activation functions for the network
            activ_derivative (dict): A dict with the derivatives of the possible activation functions
            bias_hidden_value (int): Starting value for the hidden bias
            bias_hidden2_value (int): Starting value for the second hidden bias
            bias_output_value (int): Starting value for the output bias
            activation_function (function): Selected activation function
            derivative_function (function): Derivative of the selected activation function
            hidden_weight (numpy.ndarray): Initialized weight of the first hidden layer
            hidden_weight2 (numpy.ndarray): Initialized weight of the second hidden layer
            output_weight (numpy.ndarray): Initialized weight of the output layer
            hidden_bias (numpy.ndarray): Initialized bias of the first hidden layer
            hidden_bias2 (numpy.ndarray): Initialized bias of the second hidden layer
            output_bias (numpy.ndarray): Initialized bias of the output layer
            loss (function): Selected loss function
            epoch_array (list): list of epochs ([0,1,2,3,...])
            error_array (list): list of the losses per epoch

        """

        self.activation_functions = {
                'tanh': (lambda x: np.tanh(x)),
                'sigmoid': (lambda x: 1.0/(1.0 + np.exp(-x))),
               }

        # Derivatives of the activation functions ASSUMING that x has already been passed throught the activation function.
        self.activ_derivative = {
                'tanh': (lambda x: 1-x**2),
                'sigmoid': (lambda x: x*(1.0-x))
                }        
        
        self.input_dims = input_dim                                         # Input  Dimensions
        self.hidden_dims = hidden_dim                                       # Hidden Dimensions
        self.hidden_dims2 = hidden_dim2                                     # Hidden Dimensions 2
        self.output_dims = output_dim                                       # Output Dimenstions
        self.lr = lr                                                        # Learning rate
        self.epochs = epochs                                                # Epochs
        self.bias_hidden_value = 0                                          # Bias HiddenLayer
        self.bias_hidden2_value = 0                                         # Bias Hidden2Layer
        self.bias_output_value = 0                                          # Bias OutputLayer
        self.activation_function = self.activation_functions[activation]    # Activation Function
        self.derivative_function = self.activ_derivative[activation]        # Derivative Function
        self.classes_number = classes                                       # Classes in Dataset
    
        'Starting Bias and Weights'
        self.hidden_weight = self.init_weights(self.input_dims, self.hidden_dims)
        self.hidden_weight2 = self.init_weights(self.hidden_dims, self.hidden_dims2)
        self.output_weight = self.init_weights(self.hidden_dims2, self.output_dims)
        self.hidden_bias = np.full((self.hidden_dims), self.bias_hidden_value, dtype=np.float32)
        self.hidden_bias2 = np.full((self.hidden_dims2), self.bias_hidden2_value, dtype=np.float32)
        self.output_bias = np.full((self.output_dims), self.bias_output_value, dtype=np.float32)

        if loss == "square":
            self.loss = SquareLoss()
        else:
            self.loss = CrossEntropyLoss()

        self.epoch_array = []
        self.error_array = []
    
    def init_weights(self, x, y, random=False):
        """ Weight initializer.

        Args:
            x (int): input size of the weight
            y (int): output size of the weight
            random (bool): random initialization or "pytorch-style" initialization

        Returns:
            weight numpy.ndarray
        """
        if random:
            weight = np.random.random((x,y)).astype(np.float32) * 0.001
        else:
            limit   = 1 / math.sqrt(y)
            weight  = np.random.uniform(-limit, limit, (x, y))
        return weight

    def backprop(self, loss_delta, inputs, forward_1, forward_2, forward_3):
        """ Implements the backpropagation.

        Args:
            To be determined by you :)

        Returns:
            To be determined by you :)
        """

        # Backpropagation phase
        
        # Updating the weights and bias

        delta = np.multiply(loss_delta,self.derivative_function(forward_3))
        self.output_weight -= self.lr*np.outer(delta.T, forward_2).T # to be corrected by you
        delta2 = (delta@self.output_weight.T)*(self.derivative_function(forward_2))
        self.hidden_weight2 -= self.lr*np.outer(delta2.T, forward_1).T # to be corrected by you
        delta3 = (delta2@self.hidden_weight2.T)*(self.derivative_function(forward_1))
        self.hidden_weight -= self.lr*np.outer(delta3.T, inputs).T # to be corrected by you
        
        self.hidden_bias -= self.lr*delta3 # to be corrected by you
        self.hidden_bias2 -= self.lr*delta2 # to be corrected by you
        self.output_bias -= self.lr*delta # to be corrected by you
        
    def show_err_graphic(self,list_errors,list_indices_epochs):
        """ Loss plotter.

        Args:
            list_errors (list): list of the losses per epoch 
            list_indices_epochs (list): list of epochs ([0,1,2,3,...])

        Returns:
            None
        """

        plt.figure(figsize=(9,4))
        plt.plot(list_indices_epochs, list_errors, "m-", marker=11)
        plt.xlabel("Number of Epochs")
        plt.ylabel("Loss");
        plt.title("Error Minimization")
        plt.show()

    def predict(self, X, y):
        """ Returns the predictions for every element of X.

        Args:
            X (numpy.ndarray): --> Inputs.
            y (numpy.ndarray): --> labels/target.

        Returns:
            dataframe (pandas dataframe): Table of all predictions
            accuracy (float): Percentage of correct predictions
        """

        X = (X - np.min(X)) / (np.max(X) - np.min(X))
        'Forward Propagation'
        # Pass through hidden fully-connected layer

        forward_1 = self.activation_function(X@self.hidden_weight + self.hidden_bias) # to be corrected by you

        # Pass through hidden fully-connected layer 2

        forward_2 = self.activation_function(forward_1@self.hidden_weight2 + self.hidden_bias2) # to be corrected by you

        # Pass through output fully-connected layer

        predictions_raw = self.activation_function(forward_2@self.output_weight + self.output_bias) # to be corrected by you
    
        predictions = (predictions_raw > 0.5)#.reshape(predictions_raw.shape[0])
        
        # EXAMPLE:
        # predictions_raw = [[0.47667593] [0.47190604] [0.52146905] [0.48481489] [0.5378639 ]]
        # predictions = predictions_raw > 0.5
        # e.g. predictions = [False False True False True True]
        
        accuracy = self.loss.acc(y, predictions)
        
        array_score = []
        for i in range(len(predictions)):
            y_pred = predictions[i]
            if y_pred == False: 
                array_score.append([i, 'Circle', int(predictions[i]), predictions_raw[i], y[i]])
            elif y_pred == True:
                 array_score.append([i, 'Square', int(predictions[i]), predictions_raw[i], y[i]])
                    
        dataframe = pd.DataFrame(array_score, columns=['_id', 'class', 'Pred', 'Conf', 'GT'])
        return dataframe, accuracy

    
    def fit(self, X, y): 
        """ Loops over the dataset to train the network.

        Args:
            X (numpy.ndarray): --> Inputs.
            y (numpy.ndarray): --> labels/target.

        Returns:
            None
        """
        # n_observations -> number of training examples
        # m_features -> number of features 
        n_observations, m_features = X.shape

        self.list_indices_epochs = []
        self.list_errors = []

        # Normalize Dataset
        X = (X - np.min(X)) / (np.max(X) - np.min(X))
        m = 0
        pbar = trange(self.epochs)
        for current_epoch in pbar:
            loss = 0
            for idx, inputs in enumerate(X): 
                # Stage 1 - Forward Propagation

                # Pass through hidden fully-connected layer
                forward_1 = self.activation_function(inputs@self.hidden_weight + self.hidden_bias) # to be corrected by you
                # Pass through hidden fully-connected layer 2

                forward_2 = self.activation_function(forward_1@self.hidden_weight2 + self.hidden_bias2) # to be corrected by you
                if np.max(np.abs(inputs@self.hidden_weight + self.hidden_bias)) > m:
                    m = np.max(np.abs(forward_1@self.hidden_weight2 + self.hidden_bias2))
                # Pass through output fully-connected layer

                forward_3 = self.activation_function(forward_2@self.output_weight + self.output_bias) # to be corrected by you
                
                # Calculate loss
                loss_delta = self.loss.delta(y[idx], forward_3) # to be corrected by you
                
                
                # Stage 2 - Backpropagation to update weights

                self.backprop(loss_delta, inputs, forward_1, forward_2, forward_3) # to be corrected by you
                
                loss += self.loss.loss(y[idx], forward_3)
                
            #pbar.set_description("Epoch Error: %s" % str(m))  
            pbar.set_description("Epoch Error: %s" % str(loss/X.shape[0]))
            self.list_errors.append(loss/X.shape[0])
            self.list_indices_epochs.append(current_epoch)
            loss = 0
        
        # self.list_errors = [array([0.72250545]), array([0.70937303]), array([0.70296456]), array([0.69953981]), array([0.69750547])]
        # self.list_indices_epochs = [0, 1, 2, 3, 4]
             
        self.show_err_graphic(self.list_errors, self.list_indices_epochs)
