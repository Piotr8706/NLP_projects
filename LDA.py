import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

class LDA:
    """
    Citing Wikipedia:
    In natural language processing, latent Dirichlet allocation (LDA) is a Bayesian network 
    (and, therefore, a generative statistical model) for modeling automatically extracted 
    topics in textual corpora. The LDA is an example of a Bayesian topic model. In this, 
    observations (e.g., words) are collected into documents, and each word's presence is attributable 
    to one of the document's topics. Each document will contain a small number of topics.
    """
    def __init__(self):
        self.means = None
        self.cov_matrix = None
        self.weights = None

    def fit(self, X, y):
        # Compute the mean vectors for each class
        self.means = {}
        for label in np.unique(y):
            self.means[label] = np.mean(X[y == label], axis=0)
        
        # Compute the within-class scatter matrix
        n_features = X.shape[1]
        S_w = np.zeros((n_features, n_features))
        for label in np.unique(y):
            class_scatter = np.cov(X[y == label].T, bias=True)
            S_w += class_scatter
        
        # Compute the between-class scatter matrix
        overall_mean = np.mean(X, axis=0)
        S_b = np.zeros((n_features, n_features))
        for label, mean_vec in self.means.items():
            n = X[y == label].shape[0]
            mean_vec = mean_vec.reshape(n_features, 1)
            overall_mean = overall_mean.reshape(n_features, 1)
            S_b += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
        
        # Compute the eigenvalues and eigenvectors for the scatter matrices
        A = np.linalg.inv(S_w).dot(S_b)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        # Sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(abs(eigenvalues))[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        self.weights = eigenvectors[:, :n_features - 1]
    
    def transform(self, X):
        return X.dot(self.weights)
    
def main():
    """
    The Iris dataset consists of 150 samples of iris flowers from three different species 
    (Iris setosa, Iris versicolor, and Iris virginica). 
    Each sample has four features: Sepal length (cm), Sepal width (cm), Petal length (cm) and 
    Petal width (cm)
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Standardize the data
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # Fit LDA and transform the data
    lda = LDA()
    lda.fit(X_std, y)
    X_lda = lda.transform(X_std)
    
    # Plot the results
    plt.figure()
    colors = ['r', 'g', 'b']
    markers = ['s', 'x', 'o']
    for label, color, marker in zip(np.unique(y), colors, markers):
        plt.scatter(X_lda[y == label, 0], X_lda[y == label, 1], label=iris.target_names[label], color=color, marker=marker)
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.legend(loc='best')
    plt.title('LDA: Iris projection onto the first 2 linear discriminants')
    plt.show()

if __name__ == "__main__":
    main()
