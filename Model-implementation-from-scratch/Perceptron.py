import numpy as np


def activation(x):
    return np.where( x >0, 1, 0)


class Perceptron:
    
    def __init__(self, learning_rate = 0.01, n_iters = 1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = activation
        self.weights = None
        self.bias = None
        
    def fit( self, X, y):
        
        n_samples, n_features = X.shape
        
        # initialize the parameters with random weight initialization
        
        self.weights = np.random.randn(n_features)
        self.bias = 0
        
        y_ = np.where( y >0, 1, 0)
        
        # learn weights
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = x_i @ self.weights + self.bias
                y_predicted = self.activation_func( linear_output)
                
                update = self.lr * ( y_[idx] - y_predicted)
                self.weights += update*x_i
                self.bias +=update
                
                
    
    def predict(self, X):
        linear_output=  X @ self.weights + self.bias
        y_predicted = self.activation_func(linear_output)
        
        return y_predicted
    
    