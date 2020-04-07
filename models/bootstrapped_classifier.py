import numpy as np
import copy


class BootstrappedClassifier:
    def __init__(self, classifier, n_bootstraps, bootstrap_size):
        self.classifiers = [copy.copy(classifier) for i in range(n_bootstraps)]
        self.n_bootstraps = n_bootstraps
        self.bootstrap_size = bootstrap_size

    def fit(self, X, y, **kwargs):
        for classifier in self.classifiers:
            idx_sample = np.random.random_integers(low=0, high=len(X) - 1,
                                                   size=self.bootstrap_size)
            X_sample = X[idx_sample]
            y_sample = y[idx_sample]
            classifier.fit(X_sample, y_sample)

    def predict(self, X):
        y_preds = []
        for classifier in self.classifiers:
            y_preds += [np.expand_dims(classifier.predict_proba(X)[:, 1], 1)]
        y_preds_concatenated = np.concatenate(y_preds, axis=1)
        predictions = np.mean(y_preds_concatenated, axis=1)
        return predictions

    def predict_proba(self, X):
        mean_predictions = self.predict(X)
        return np.stack([1 - mean_predictions, mean_predictions], axis=1)

    def get_raw_outputs(self, X):
        y_preds = []
        for classifier in self.classifiers:
            y_preds += [classifier.predict_proba(X)[:, 1]]
        return y_preds
