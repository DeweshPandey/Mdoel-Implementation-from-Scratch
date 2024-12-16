import numpy as np

# 1. Training : during training we basically want to obtain the prior P(y) (frequency in training example) , 
# the new information P(xi|y) ( this is basically probibility of a feature occuring given a class ) is called class conditional probability
# for a categorical feature P(xi|y) can be obtained using histgram of xi in class y
# for a continuous feature P{xi|y) can be obtained using Gaussian Distribution of xi in class y 
# gaussian distribution is obtained from a mean and standard deviation 
# basically its setting up the formula an not actually training 

# 2. Predicitons are made using the Bayes Theorem P(y|x) ~ P(y)*P(xi|y) class of y for which the likelihood or loglikelihood is higher  
# i.e. y = argmax(y) ( loglikelihood for given class for a given test example)



class NaiveBayes:
    
    # no need of __init__() function since this Classifer doesn't need any parameter or hyperparameter
    
    def fit(self, X, y): 
        # fit the data  
        n_samples, n_features = X.shape
        self._classes = np.unique(y) # all unqiue classes
        n_classes= len(self._classes)
        
        # calculation mean , variance and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype = np.float64) # initialization with zeros for correct shape
        self._var = np.zeros((n_classes, n_features), dtype = np.float64)
        self._priors = np.zeros(n_classes, dtype = np.float64)
        
        for idx, c in enumerate( self._classes): # traverse through each class
            X_c = X[y==c]  
            self._mean[idx, :]  = np.mean(X_c,axis = 0)
            self._var[idx, :]  = np.var(X_c,axis = 0)
            self._priors[idx] = X_c.shape[0]/float(n_samples)
        
        
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []
        
        # calculate the porsterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x))) + prior
            posteriors.append(posterior)
            
        # return the class with higest posterior
        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        
        numerator = np.exp( - (( x- mean)**2)/(2*var))
        denominator = np.sqrt(2*np.pi*var)    
        return numerator/denominator