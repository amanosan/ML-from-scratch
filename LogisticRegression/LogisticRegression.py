import numpy as np


class LogisticRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = 1000
        self.weights = None
        self.bias = None

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # implementing gradient descent
        for _ in range(self.n_iters):
            linearmodel = np.dot(X_train, self.weights) + self.bias
            y_pred = self._sigmoid(linearmodel)

            # updating weights and bias
            dw = (1/n_samples) * (np.dot(X_train.T, (y_pred - y_train)))
            db = (1/n_samples) * np.sum(y_pred - y_train)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X_test):
        linear_model = np.dot(X_test, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        predictions = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(predictions)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
