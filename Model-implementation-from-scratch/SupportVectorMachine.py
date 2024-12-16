import numpy as np

# in SVM misclassification is allowed that whats make the model more robust to outlier
# basic idea is to make a hyperplane between class ( in any dimensions) 
#wide street approach : trying the separate the classes by crate a street of width 2/||w|| between the classes, 
# where w is the vector such that w is perpendicular the direction of running street 
# task is to maximize the street width and minimize ||w|| 
# constraint include let there exist x_i and b such that 
#               w.x_i = b or  w.x_i - b = 0 , conveying the centre of the street 
# CASE 1 : now for a x to exist in a + class  at boundary of street :
#               w.x_i -b = +1 , for some i  , then for x to be in + class w.x_i -b >= 1 for some i
# CASE 2 : similarty for x to exist in a - class at boundary of stree :
#               w.x_i -b = -1 for some i , then for x to be in - class w.x_i -b <= -1 for some i
# let y_i be +1  for CASE 1  and -1 for CASE -1
# that CASE 1 & 2 combined  as :
#               y_i*( w.x_i -b ) >= 1
# therefore the loss function or Lagragian : L = lambda* ||w||^2 - (1/m)*(summation of max( 0 , y_i*(w.x_i-b)))
# kernel can also be used to transform the data into higer dimension to get more efficient hyperplane
# polynomial kernal = > ( 1+ wx)^n , n is degree of polynomial
# expoenential or radial basis kernel exp( - ( ||X|| - ||X'|| )^2/(2*sigma^2) ) , also called RBF kernal

class SVM:
    
    def __init__( self, learning_rate= 0.01, lambda_ =  0.01, n_iterations = 1000):
        self.lr = learning_rate
        self.lambda_param = lambda_ #the regularization parameter in the Loss Function or Lagrangian 
        self.n_iters = n_iterations
        self.w = None 
        self.b = None
        
    def fit(self, X, y):
        n_samples , n_features= X.shape
        
        y_ = np.where(y <=0, -1, 1 ) # here we check where the y is in which class
        
        #  the weights and biase with random initialization
        self.w = np.zeros( n_features)
        self.b = 0
        
        for _ in range(self.n_iters): # running the gradient descent n number of time
            for idx , x_i  in enumerate(X):
                condition = y_[idx]*(np.dot(x_i,self.w) -self.b ) >= 1
                if condition: # if the condition is true i.e. the predicition lies on the actual side of the hyperplane 
                    self.w -= self.lr*(2*self.lambda_param*self.w) # then we upadata the parameter according the modified loss function 
                    self.b # no update as the gradient of loss function w.r.t. to b is 0
                else: # when prediction is incorrect classification
                    self.w -= self.lr*(2*self.lambda_param*self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr*(y_[idx])  
    
    def predict(self, X):
        approx = np.dot( X,self.w) -self.b # using the learned parameter for prediction
        return np.sign(approx) # converts the prediction in -1 and +1 based upon the sign ( or the side of the hyperplane which they lie)