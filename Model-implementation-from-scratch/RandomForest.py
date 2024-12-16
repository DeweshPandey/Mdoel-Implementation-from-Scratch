import numpy as np
from collections import Counter
from DecisionTree import DecisionTree


class RandomForest:
    
    def __init__( self, n_trees=10, max_depth =10, min_sample_split=2, n_features=None):
        self.n_trees= n_trees # number of Decision trees to be used for voting for output
        self.max_depth = max_depth # maximum depth of any particular tree
        self.min_sample_split = min_sample_split # minimum number of samples
        self.n_features = n_features # number of features to be used by each tree
        self.trees = [] #  variable to store all the trees

    def fit( self, X, y ):
        self.trees =[] # empty list of trees
        
        for _ in range(self.n_trees): # creating n number of Decision Trees
            tree =DecisionTree(max_depth = self.max_depth, 
                               min_samples_splits = self.min_sample_split, 
                               n_features = self.n_features  
                               ) 
            X_sample , y_sample = self._bootstrap_samples(X, y) # helper function to generate a new dataset from the original 
            tree.fit( X_sample, y_sample) # fitting the different dataset to tree  on each iteration
            self.trees.append(tree)
                   
    def _bootstrap_samples(self, X, y): 
        n_samples = X.shape[0] # to get number of record in the data set
        idxs = np.random.choice( n_samples, n_samples//3 , replace = True) # the special feature to random forest algo is \
            # it random picks the records/training examples from dataset may contain the duplicate or may not 
            # this same technique is applied on selection of features or columns generating on a new dataset altogether
        return X[idxs], y[idxs]
    
    def _most_common_value(self, y): 
        # helper function to get majority class
        counter = Counter(y) # creating the counter object
        return counter.most_common(1)[0][0] # most_common([n]) outputs the list of tuples of n most commone elements/class in y. as [(value1, #occurance) , (value2, #occurance), ...]
        
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # this find prediction of each sample in test for each tree
        # thus it is a list of lists [[], [], []]
        # now we find the prediciton of for a particular sample from mojority voting of prediciton of all trees for particulare given sample
        tree_preds =np.swapaxes( predictions, 0,1) # interchanges the axes
        predictions = np.array([self._most_common_value( pred) for pred in tree_preds])
        return predictions
        