import numpy as np


class LinearRegression:

    '''
    lr - Learning Rate
    n_iters - Number of Iterations
    weights - Weights of the features
    bias - Constant term or Intercept value
    '''

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        #  applying Gradient Descent algorithm
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.sum(np.dot(X.T, (y_pred - y)))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return np.array(y_predicted)
