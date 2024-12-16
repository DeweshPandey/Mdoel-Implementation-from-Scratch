# Scratch Implementation of Basic Machine Learning Models using OOP

This repository contains scratch implementations of basic machine learning models using Object-Oriented Programming (OOP) principles. The goal of this repository is to provide clear and concise implementations of popular machine learning algorithms without relying on external libraries like scikit-learn, while maintaining good code structure and readability.

### Models Implemented:
- **K-Nearest Neighbors (KNN)**
- **K-Means Clustering**
- **Decision Tree**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **Naive Bayes**
- **Perceptron**
- **Principal Component Analysis (PCA)**

## Model Descriptions

### K-Nearest Neighbors (KNN)
KNN is a simple, instance-based learning algorithm. It classifies a data point based on the majority class of its nearest neighbors in the feature space. The number of neighbors (k) is a parameter that needs to be chosen carefully. The algorithm is easy to implement but can become computationally expensive for large datasets.

### K-Means Clustering
K-Means is an unsupervised learning algorithm used for clustering. It groups data points into k clusters based on feature similarity. The algorithm iteratively assigns data points to the nearest cluster centroid and recalculates the centroid until convergence.

### Decision Tree
A decision tree is a supervised learning algorithm that splits the data into subsets based on the feature that results in the most significant information gain. The tree structure is recursive, with each node representing a decision based on feature values, and the leaves representing the final output or class.

### Random Forest
Random Forest is an ensemble learning method that creates multiple decision trees and aggregates their predictions to improve accuracy and reduce overfitting. It introduces randomness by training each tree on a random subset of the data and features.

### Support Vector Machine (SVM)
SVM is a supervised learning algorithm used for classification and regression tasks. It finds a hyperplane that best separates the classes in the feature space. SVM is effective in high-dimensional spaces and is particularly useful for binary classification.

### Naive Bayes
Naive Bayes is a probabilistic classifier based on Bayes' Theorem. It assumes that features are independent, hence the "naive" assumption. It is fast and effective for text classification tasks, particularly when dealing with large datasets.

### Perceptron
A perceptron is a simple neural network model used for binary classification. It consists of a single layer of weights that are adjusted during training based on the input data. Perceptron works well for linearly separable problems but can struggle with non-linearly separable data.

### Principal Component Analysis (PCA)
PCA is an unsupervised dimensionality reduction technique. It transforms the data into a new coordinate system where the greatest variances are captured in the first principal components. PCA is useful for reducing the complexity of data while retaining as much information as possible.

## Features
- Each model is implemented using object-oriented principles, allowing for easy extension and reuse of code.
- The code is well-commented, with explanations for both covered and uncovered cases.
- Models are implemented from scratch with minimal dependencies, focusing on the core logic of each algorithm.
## Dependencies

- **Python 3.x** (required)
- **Numpy** (required)

You can install the necessary dependencies using pip:

```bash
pip install numpy
git clone https://github.com/yourusername/ml-scratch-implementation.git
