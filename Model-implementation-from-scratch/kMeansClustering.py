import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance( x1 , x2):
    return np.linalg.norm(x1-x2)
    
    
    
class KMeans:
    
    def __init__( self, K=5, max_iters = 100 , plot_steps = False):
        
        self.K =K
        self.max_iters = max_iters 
        self.plot_steps = plot_steps
        
        self.clusters = [[] for _ in range(self.K)] # list of sample indices for each cluster
        self.centeroids = [] # the centres ( mean vectors ) for each cluster
        
    def predict(self, X):
        
        self.X= X
        self.n_samples , self.n_features = X.shape
        
        # random initialize of centeroids 
        random_sample_idxs = np.random.choice( self.n_samples, self.K , replace= False)
        self.centeroids = self.X[random_sample_idxs]
        # self.centeroids = [ self.X[idx] for idx in random_sample_idxs]
        
        # optimization of cluster centeroids
        for _ in range(self.max_iters):
            #assign samples x_i to the closest centeroids
            self.clusters = self._create_clusters( self.centeroids) # every time the new clusters is created ( not modified)
            
            if self.plot_steps:
                self.plot()
            # calculate the new centeroids from clusters
            centeroids_old = self.centeroids # storeing the old centeroids or previous centeroid value to compare the convergence
            self.centeroids = self._get_centeroids(self.clusters)  # calculating the centeroids from the assigned above cluster
            
            if self._is_converged( centeroids_old, self.centeroids):
                break
            
            if self.plot_steps:
                self.plot()
                
        # classify the samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)
    
    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluseter it is assigned to
        labels = np.empty( self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx]= cluster_idx
        return labels
            
    def _get_centeroids(self, clusters):
        # assign the mean value of the cluster to the centeroids
        centeroids = np.zeros(( self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            centeroids[cluster_idx] = np.mean(self.X[cluster] , axis =0)
            
        return centeroids            
            
    def _create_clusters(self, centeroids):
        # assign the samples to the closest centeroids
        clusters = [[] for _ in range(self.K)]
        for idx , sample in enumerate(self.X):
            centeroid_idx =  self._closest_centeroid(sample, centeroids)
            clusters[centeroid_idx].append(idx)
        
        return clusters
            
    def _closest_centeroid(self, sample, centeroids):
        # distance of the current sample to each centeroid
        distances = [euclidean_distance(sample, point ) for point in centeroids]
        closest_idx = np.argmin(distances)

        return closest_idx
                

        
        
    def _is_converged(self, centeroids_old, centeroids):
        # check the distances between old and new centeroids for all centeroids 
        distances = [euclidean_distance(centeroids_old[i], centeroids[i]) for i in range(self.K)]
        return sum(distances)==0
    
    def plot(self):
        
        fig, ax = plt.subplots( figsize = (12, 8))            
        
        for  i, index in enumerate( self.clusters):
            point = self.X[index].T
            ax.scatter(*point)
        
        for point in self.centeroids:
            ax.scatter(*point, marker= "x", color ="black", linewidth= 2)
            
        plt.show()        
            
            
            
            # for m in range(self.n_samples):
            #     min = float("inf")
            #     # clustcode= []
            #     for c in range(self.K):
            #         dist = np.linalg.norm(self.X[m]- self.centeroids[c] )
            #         if dist < min :
            #             min = dist
            #             clustcode = (c, m)
                        
            #     if _ ==1:
            #         self.clusters[clustcode[0]].append(self.X[clustcode[1]])
            #     else:
            #         self.clusters[clustcode[0]]
            
            # self.centeroids []
            
        
        
        
        
        
        
        
        