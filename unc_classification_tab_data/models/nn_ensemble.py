import torch
from .mlp import MLP
import numpy as np


class NNEnsemble:
    """Wrapper class for an ensemble of neural networks.

    Parameters
    ----------
    n_models: int
        The number of ensemble members.
    model_params: dict
        The model parameters, see class MLP.
    """

    def __init__(self, n_models: int, model_params: dict):
        self.n_models = n_models
        self.model_params = model_params
        self.models = dict()

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
              training_params: dict):
        """Train all MLPs on the training data.

        Parameters
        ----------
        X_train: np.ndarray
            The training data
        y_train: np.ndarray
            The labels corresponding to the training data.
        X_val: np.ndarray
            The validation data
        y_val: np.ndarray
            The labels corresponding to the validation data.
        training_params: dict
            The parameters used for training, see class MLP.
        """
        for i in range(self.n_models):
            mlp = MLP(**self.model_params)
            mlp.train(X_train, y_train, X_val, y_val,
                      **training_params)
            self.models[i] = mlp.model

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Predict the probabilities p(y|X) by averaging the predictions of ensemble members.

        Parameters
        ----------
        X_test: np.ndarray
            The test data.

        Returns
        -------
        type:np.ndarray
            The predicted probabilities.
        """
        predictions = []
        X_test_tensor = torch.tensor(X_test).float()
        for i in range(self.n_models):
            self.models[i].eval()
            predictions.append(torch.sigmoid(
                self.models[i](X_test_tensor)).detach().squeeze().numpy())
        mean_predictions = np.mean(np.array(predictions), axis=0)
        return np.stack([1 - mean_predictions, mean_predictions], axis=1)

    def get_single_predictions(self, X_test: np.ndarray, ensemble_member: int = 0) -> np.ndarray:
        """Return the probabilities p(y|X) as predicted by a single ensemble member.

        Parameters
        ----------
        X_test: np.ndarray
            The test data.
        ensemble_member: int
            The index of the ensemble member.

        Returns
        -------
        type:np.ndarray
            The predicted probabilities.
        """
        X_test_tensor = torch.tensor(X_test).float()
        self.models[ensemble_member].eval()
        predictions = torch.sigmoid(
            self.models[ensemble_member](X_test_tensor)).detach().squeeze().numpy()
        return np.stack([1 - predictions, predictions], axis=1)
