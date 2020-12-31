import numpy as np


class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # initializing mean, var, priors
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            X_c = X[c == y]
            self._mean[c, :] = np.mean(X_c, axis=0)
            self._var[c, :] = np.var(X_c, axis=0)
            self._priors[c] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predictSample(x) for x in X]
        return y_pred

    # Function to predict one sample
    def _predictSample(self, X):
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditionals = np.sum(np.log(self._pdf(idx, X)))

            posterior = prior + class_conditionals
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    # Returns the Gaussian PDF of sample
    def _pdf(self, class_idx, X):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (X-mean)**2 / (2*var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
