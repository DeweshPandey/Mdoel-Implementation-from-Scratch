from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DecisionTree
from RandomForest import RandomForest
from NaiveBayes import NaiveBayes
from PrincipleComponentAnalysis import PCA
from Perceptron import Perceptron
from SupportVectorMachine import SVM
from kMeansClustering import KMeans
from KNN import KNN
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
np.random.seed(42)


data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train , X_test, y_train, y_test = train_test_split(
    X, y, test_size =0.2, random_state = 2
)

clf = DecisionTree(max_depth= 20)
clf.fit(X_train, y_train)
predicitions = clf.predict(X_test)

def accuracy( y_test , y_pred):
    return np.sum(y_test == y_pred)/len(y_test)
    
    
    
print(accuracy(y_test, predicitions))

# clf = RandomForest(max_depth= 20, n_trees = 10)
# clf.fit(X_train, y_train)
# predicitions = clf.predict(X_test)


# print(accuracy(y_test, predicitions)) 

# clf = NaiveBayes()
# clf.fit(X_train, y_train)
# predicitions = clf.predict(X_test)


# print(accuracy(y_test, predicitions)) 

# data = datasets.load_iris()
# X=data.data
# y = data.target
# pca = PCA(2)
# pca.fit(X)
# X_projected = pca.transform(X)

# print(" Shape of X : ", X.shape)
# print(" Shape of projected_X or transformed X: ", X_projected.shape)

# x1 = X_projected[:, 0]
# x2 = X_projected[:, 1]

# plt.scatter(
#     x1, x2, c=y , edgecolors= "none", alpha = 0.8, cmap = plt.cm.get_cmap("viridis", 3)
    
# )

# plt.xlabel("Principal Component 1")
# plt.ylabel("Principle Component 2")
# plt.colorbar()
# plt.show()
 

X,y = datasets.make_blobs(
    n_samples = 500, n_features = 2 , centers = 3, shuffle = True, random_state= 40
)

# X_train, X_test, y_train , y_test = train_test_split(
#     X, y, test_size= 0.2, random_state = 123
# )

# percep = Perceptron()
# percep.fit(X_train, y_train)
# predictions= percep.predict(X_test)

# print(accuracy(y_test, predictions))

# fig = plt.figure()
# ax= fig.add_subplot(1,1,1)
# plt.scatter(X_train[:,0], X_train[:,1], marker="o" ,c= y_train)
# x0_1 = np.amin( X_train[:,0])
# x0_2 = np.amax(X_train[:,0])

# x1_1 = ( -percep.weights[0]*x0_1 - percep.bias)/percep.weights[1]
# x1_2 = ( -percep.weights[0]*x0_2 - percep.bias)/percep.weights[1]

# ax.plot([x0_1, x0_2],[x1_1 , x1_2], "k")

# ymin = np.amin(X_train[:,1])
# ymax = np.amax(X_train[:,1])
# ax.set_ylim([ymin-3 , ymax+3])

# plt.show()

# y = np.where(y ==0, -1, 1)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y , test_size = 0.2, random_state = 123
# )

# clf = SVM()
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)

# print( "SVM classification accuracy: ", accuracy( y_test, predictions))

# def visualize_svm():
    
#     def get_hyperplane_value(x, w, b ,offset):
#         return (-w[0]*x +b+ offset)/w[1]
    
#     fig = plt.figure()
#     ax= fig.add_subplot( 1,1,1)
#     plt.scatter( X[:,0 ], X[:, 1], marker ="o", c=y)
    
#     x0_1 = np.amin(X[:, 0])
#     x0_2 = np.amax(X[:,0])
    
#     x1_1 = get_hyperplane_value(x0_1, clf.w , clf.b, 0)
#     x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)
    
#     x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b , -1)
#     x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)
    
#     x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
#     x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)
    
#     ax.plot( [ x0_1, x0_2], [x1_1,x1_2], "y--")
#     ax.plot( [x0_1, x0_2], [x1_1_m , x1_2_m], "k")
#     ax.plot( [x0_1, x0_2],[x1_1_p, x1_2_p], "k")
    
#     x1_min = np.amin(X[:, 1])
#     x1_max = np.amax(X[:, 1])
#     ax.set_ylim(x1_min-3, x1_max+3)
    
#     plt.show()
    
# visualize_svm()
    
# print(X.shape)

# clusters=  len(np.unique(y))
# print(clusters)
# k = KMeans( K = clusters, max_iters = 150, plot_steps=True)
# y_pred = k.predict(X)

# k.plot()   


iris = datasets.load_iris()
X, y = iris.data, iris.target

cmap = ListedColormap(["#FF0000", '#00FF00','#0000FF'])

X_train , X_test , y_train, y_test  = train_test_split(
    X, y, test_size = 0.2 , random_state= 1234
) 



plt.figure()
plt.scatter( X[:,2], X[:,3] , c= y , cmap= cmap, edgecolors= "k", s=20)
plt.show()

clf = KNN(k=5)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)

print(predictions)

print( accuracy(y_test, predictions))