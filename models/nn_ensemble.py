import torch
from mlp import MLP
import numpy as np


class NNEnsemble:
    def __init__(self, n_models, model_params):
        self.n_models = n_models
        self.model_params = model_params
        self.models = dict()

    def train(self, X_train, y_train, X_val, y_val, training_params):
        for i in range(self.n_models):
            mlp = MLP(self.model_params)
            mlp.train(X_train, y_train, X_val, y_val,
                      training_params)
            self.models[i] = mlp.model

    def predict(self, X_test):
        predictions = []
        X_test_tensor = torch.tensor(X_test).float()
        for i in range(self.n_models):
            self.models[i].eval()
            predictions += [torch.sigmoid(
                self.models[i](X_test_tensor)).detach().squeeze().numpy()]
        y_preds_concatenated = np.concatenate(predictions, axis=1)
        predictions = np.mean(y_preds_concatenated, axis=1)
        return predictions

    def predict_proba(self, X_test):
        mean_predictions = self.predict(X_test)
        return np.stack([1 - mean_predictions, mean_predictions], axis=1)
