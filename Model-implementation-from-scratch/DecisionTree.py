
import numpy as np
from collections import Counter

# creating a node class separately to handle all the properties of the node object 
class Node: 
    def __init__(self, feature=None, threshold = None, left = None, right= None, *, value=None ): # by adding astrix now value must be passed by name
        self.feature = feature # a node should have the information about the splitting feature or column
        self.threshold =threshold # a node should have the information about splitting criteria/ value if the split feature is a numerical of multiclass 
        self.left = left # should point to the next child left node object
        self.right =right # should poin to the next child right node object
        self.value =value # should contain the value if it is the leaf node
        
    def is_leaf_node(self): # to check if given node is Leaf node
        return self.value is not None
    
    

# creating a Decision Tree from the instances of the Node class or Node object 
# recursive implementation

class DecisionTree:
    def __init__(self,  min_samples_splits=2, max_depth = 100 , n_features = None):
        self.min_samples_splits = min_samples_splits # a stopping criteria for tree division where the splitting stops for a particular node is the samples in nodes are less than this value
        self.max_depth = max_depth # another stopping criteria limmits the level of splitting
        self.n_features = n_features # make no. of columns to be used fixed. Needed particularly in random forest implementation
        self.root = None # to keep track of root of the Tree during traversing 
        
    def fit(self , X, y): # to fit the model
        
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features) # a quick check if the n_features greater than original existing features in dataset
        self.root = self._grow_tree(X, y) # a helper function that would return the root of the tree 
        
    def _grow_tree(self , X, y, depth = 0): # for recursive split and growth of tree from root 
        
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        
        # step 1: check the stopping criteria
        if (depth >= self.max_depth or n_labels ==1 or n_samples <= self.min_samples_splits): 
            # if the stopping criteria is met than the node is leaf node and must have a Value
            leaf_value= self._most_common_value(y) # finding the value/class for leaf node based upon majority class using a helper function 
            return Node(value = leaf_value) # creating a leaf node with value
        
        # step 2: find the best split
        feat_idxs = np.random.choice(n_feats, self.n_features , replace =False) # it adds randomness to the col selected. selects only n_features from n_feats for spliting
        best_feat, best_thres  = self._best_split( X, y , feat_idxs) #  ahelper function to find the best feature and threshold for the splti
        
        # step 3: create child nodes or creating the split
        left_idxs, right_idxs= self._split( X[:, best_feat], best_thres)
        
        left = self._grow_tree( X[left_idxs, :], y[left_idxs], depth+1 ) # creating the child left tree
        right = self._grow_tree( X[right_idxs, :], y[right_idxs], depth+1)

        return Node( best_feat, best_thres, left , right )
        
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1 # stores the highest gain informaiton 
        split_idx , split_threshold = None, None # store feat_idx, threshold  for which the gain is max 
        
        for feat_idx in feat_idxs:
            X_column = X[: ,feat_idx]
            thresholds = np.unique(X_column) 
            
            for thr in thresholds:
                # calculate the information gain for every combinaton of threshold in every column
                gain = self._information_gain(y ,X_column ,thr) # another helper function
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx= feat_idx
                    split_threshold = thr
                    
        return split_idx, split_threshold
        
    def _information_gain(self,y, X_column, threshold  ):
        # get the parent entropy
        parent_entropy = self._entropy(y) # function to calculate the entropy
        # create children
        left_idxs , right_idxs = self._split( X_column, threshold) # helper function to split the parent in child based on threshold 
        
        if len(left_idxs) ==0 or len(right_idxs)==0:
            return 0 # i.e. infomation gain from this split is 0 and the class is already pure thus stopping the split 
        
        # calculate the weighted avg entropy of children
        n= len(y)
        n_l , n_r = len(left_idxs), len(right_idxs) 
        e_l , e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs]) # we the value of y corresponding to these indices for split where for each class the probability can be calculated
        child_entropy = (n_l)/n*e_l + (n_r/n)*e_r
        # calculate IG
        information_gain = parent_entropy - child_entropy
        return information_gain 
        
    def _split(self,  X_column, threshold):
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs= np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs # gives the indexes for which the split split is carried out on the X_column 
        
    def _entropy(self , y):
        
        # create a histogram array
        hist = np.bincount(y)
        p_X=   hist/len(y)
        entropy = -np.sum(p*np.log(p) for p in p_X if p>0) # implementing the formula
        return entropy
        
    def _most_common_value(self, y): 
        # helper function to get majority class
        counter = Counter(y) # creating the counter object
        return counter.most_common(1)[0][0] # most_common([n]) outputs the list of tuples of n most commone elements/class in y. as [(value1, #occurance) , (value2, #occurance), ...]
        
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X]) # we traverse the tree for every record x in X 
        # it will start with root node 
    
    def _traverse_tree(self, x, node:Node):
        # this helper function with get the x and node on which it starts. Function is implemented in recursive way
        
        if node.is_leaf_node(): # check if node leaf node then get the value
            return node.value 
    
        if x[node.feature] <= node.threshold:
            return self._traverse_tree( x, node.left)
        return self._traverse_tree( x , node.right)
            

"""
points not considered 
1. Alpha pruning of tree : to penalize the tree for overfitting
2. Regression based approach where threshold is the mean of 2 readings 
3. Ginny score
4. every features is considered only once .....need to think 
5. In Regression Trees : SSR + aplha* node leaf are considered for pruning . alpha is hyperparameter found using k fold cross validation
"""