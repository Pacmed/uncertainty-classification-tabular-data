import numpy as np
import copy


class BootstrappedClassifier:
    """Wrapper class for an ensemble of classifiers, trained on different bootstraps of the data.

    Parameters
    ----------
    classifier:
        A classifier that has the functions 'fit' and 'predict_proba'
    n: int
        The ensemble size
    bootstrap_size: int
        The size of the bootstrapped samples.
    """

    def __init__(self, classifier, n: int, bootstrap_size: int):
        self.classifiers = [copy.copy(classifier) for i in range(n)]
        self.n_bootstraps = n
        self.bootstrap_size = bootstrap_size

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit all the classifiers on the training data.

        Parameters
        ----------
        X: np.ndarray
            The training data
        y: np.ndarray
            The labels corresponding to the training data.
        """
        for classifier in self.classifiers:
            idx_sample = np.random.random_integers(low=0, high=len(X) - 1,
                                                   size=self.bootstrap_size)
            X_sample = X[idx_sample]
            y_sample = y[idx_sample]
            classifier.fit(X_sample, y_sample)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict the probabilities p(y|X) by averaging the predictions of ensemble members.

        Parameters
        ----------
        X: np.ndarray
            The test data.

        Returns
        -------
        type:np.ndarray
            The predicted probabilities.
        """
        y_preds = []
        for classifier in self.classifiers:
            y_preds += [np.expand_dims(classifier.predict_proba(X)[:, 1], 1)]
        y_preds_concatenated = np.concatenate(y_preds, axis=1)
        mean_predictions = np.mean(y_preds_concatenated, axis=1)
        return np.stack([1 - mean_predictions, mean_predictions], axis=1)
