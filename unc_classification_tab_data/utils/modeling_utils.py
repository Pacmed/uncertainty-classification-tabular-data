import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Optional

TINY_NUMBER = 1e-5


def entropy(probabilities: np.ndarray, axis: int) -> np.ndarray:
    """ Calculate the entropy of a probability distribution.

        Parameters
        ----------
        probabilities: np.ndarray
            The probability distribution.
        axis: int
            Over which axis to take the entropy.

        Returns
        -------
        type: np.ndarray
            The calculated entropy.
    """
    return -np.sum(probabilities * np.log2(probabilities + TINY_NUMBER), axis=axis)


def logit(predictions: Optional[np.ndarray, list]) -> np.ndarray:
    """ Calculate the logit (inverse sigmoid) of predicted probabilities

        Parameters
        ----------
        predictions: Optional[np.ndarray, list]
            The predicted probabilities.

        Returns
        -------
        type: np.ndarray
            The logits.
    """
    return np.log(np.array(predictions) / (1 + TINY_NUMBER - np.array(predictions)) + TINY_NUMBER)


def platt_scale(test_predictions: Optional[np.ndarray, list],
                val_predictions: Optional[np.ndarray, list],
                y_val: Optional[np.ndarray, list]) -> np.ndarray:
    """ Scale the predictions with Platt scaling, based on the validation data. Return the
    scaled predictions.

    Parameters
    ----------
    test_predictions: Optional[np.ndarray, list]
        The predicted probabilities on the test set.
    val_predictions: Optional[np.ndarray, list]
        The predicted probabilities on the validation set.
    y_val: Optional[np.ndarray, list]
        The true labels of the validation set
        .
    Returns
    -------
    type: np.ndarray
        The platt scaled predictions.
    """
    lr = LogisticRegression()

    # Obtain the logits
    test_logits = logit(test_predictions)
    val_logits = logit(val_predictions)

    # Fit a logistic regression model (with bias) on the logits.
    lr.fit(val_logits.reshape(-1, 1), y_val)
    calibrated_outputs = lr.predict_proba(
        test_logits.reshape(-1, 1))[:, 1]
    return calibrated_outputs
