from typing import Tuple

import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from .constants import *


class MLPModule(nn.Module):
    """Create a new multi-layer perceptron instance with ReLU activations and no non-linearity in
    the output layer.

    Parameters
    ----------
    model_params : dict
        (CONSISTS OF)
        hidden_sizes: list
            The sizes of the hidden layers.
        input_size: int
            The input size.
        output_size: int
            The output size.
        dropout_rate: float
            The dropout rate applied after each layer (except the output layer)
        batch_norm: bool
            Whether to apply batch normalization after each layer.
    """

    def __init__(self, model_params: dict):
        hidden_sizes = model_params.get("hidden_sizes")
        input_size = model_params.get("input_size")
        output_size = model_params.get("output_size", DEFAULT_OUTPUT_SIZE)
        dropout_rate = model_params.get("dropout_rate", DEFAULT_DROPOUT_RATE)
        do_batch_norm = model_params.get("batch_norm", DEFAULT_BATCH_NORM)

        super(MLPModule, self).__init__()
        layers = []

        if len(hidden_sizes) > 0:
            layers.append(nn.Linear(input_size, hidden_sizes[0]))
            if do_batch_norm:
                layers.append(nn.BatchNorm1d(num_features=hidden_sizes[0]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            if do_batch_norm:
                layers.append(nn.BatchNorm1d(num_features=hidden_sizes[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.mlp = nn.Sequential(*layers)

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the MLP.

        Parameters
        ----------
        _input: torch.Tensor
            The input of the model.

        Returns
        -------
        type: torch.Tensor
            The output of the model.
        """
        return self.mlp(_input)


class SimpleDataset(Dataset):
    """Create a new (simple) torch Dataset instance.

    Parameters
    ----------
    X: torch.Tensor
        Predictors
    y: torch.Tensor
        Target
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        """Return the number of items in the dataset.

        Returns
        -------
        type: int
            The number of items in the dataset.
        """
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return X and y at index idx.

        Parameters
        ----------
        idx: int
            Index.

        Returns
        -------
        type: Tuple[torch.Tensor, torch.Tensor]
            X and y at index idx
        """
        return self.X[idx], self.y[idx]


class MLP:
    """Handles training of an MLPModule.

    Parameters
    ----------
    model_params : dict
        (CONSISTS OF)
        hidden_sizes: list
            The sizes of the hidden layers.
        input_size: int
            The input size.
        output_size: int
            The output size.
        dropout_rate: float
            The dropout rate applied after each layer (except the output layer)
        batch_norm: bool
            Whether to apply batch normalization after each layer.
    """

    def __init__(self, model_params: dict):
        self.model = MLPModule(model_params)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.class_weight = model_params.get("class_weight", DEFAULT_CLASS_WEIGHT_SETTING)

    def _initialize_dataloader(self, X_train: np.ndarray, y_train: np.ndarray, batch_size: int):
        """Initialize the dataloader of the train data.

        Parameters
        ----------
        X_train: np.ndarray
            The training data.
        y_train: np.ndarray
            The labels corresponding to the training data.
        batch_size:
            The batch size.
        """
        train_set = SimpleDataset(torch.from_numpy(X_train),
                                  torch.from_numpy(y_train))
        self.train_loader = DataLoader(train_set, batch_size, shuffle=True)

    def get_loss_fn(self, mean_y: float) -> torch.nn.modules.loss.BCEWithLogitsLoss:
        """Obtain the loss function to be used, which is (in case we use class weighting)
        dependent on the class imbalance in the batch.

        Parameters
        ----------
        mean_y: float
            The fraction of positives in the batch.

        Returns
        -------
        type: torch.nn.modules.loss.BCEWithLogitsLoss
            X and y at index idx

        """
        if self.class_weight:
            if mean_y == 0:
                pos_weight = torch.tensor(0.0)
            elif mean_y == 1:
                pos_weight = torch.tensor(1.0)
            else:
                pos_weight = (1 - mean_y) / mean_y

        else:
            # When not using class weighting, the weight is simply 1.
            pos_weight = torch.tensor(1.0)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        return loss_fn

    def validate(self, X_val: np.ndarray, y_val: np.ndarray) -> torch.Tensor:
        """Calculate the validation loss.

        Parameters
        ----------
        X_val: np.ndarray
            The validation data.
        y_val: np.ndarray
            The labels corresponding to the validation data.

        Returns
        -------
        type: torch.Tensor
            The validation loss.
        """
        self.model.eval()
        y_pred = self.model(torch.tensor(X_val).float())
        loss_fn = self.get_loss_fn(torch.tensor(y_val).float().mean().item())
        val_loss = loss_fn(y_pred, torch.tensor(y_val).float().view(-1, 1))
        return val_loss

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
              training_params: dict):
        """Train the MLP.

        Parameters
        ----------
        X_train: np.ndarray
            The training data.
        y_train: np.ndarray
            The labels corresponding to the training data.
        X_val: np.ndarray
            The validation data.
        y_val: np.ndarray
            The labels corresponding to the validation data.
        training_params: dict
            The hyperparameters used for training.
        """
        batch_size = training_params.get('batch_size', DEFAULT_BATCH_SIZE)
        n_epochs = training_params.get('n_epochs', DEFAULT_N_EPOCHS)
        early_stopping = training_params.get('early_stopping', True)
        early_stopping_patience = training_params.get('early_stopping_patience',
                                                      DEFAULT_EARLY_STOPPING_PAT)

        self._initialize_dataloader(X_train, y_train, batch_size)
        prev_val_loss = float('inf')
        n_no_improvement = 0
        for epoch in range(n_epochs):

            self.model.train()
            for batch_X, batch_y in self.train_loader:
                y_pred = self.model(batch_X.float())
                loss_fn = self.get_loss_fn(batch_y.float().mean())

                loss = loss_fn(y_pred.view(-1, 1), batch_y.float().view(-1, 1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if early_stopping:
                val_loss = self.validate(X_val, y_val)
                if val_loss > prev_val_loss:
                    n_no_improvement += 1
                else:
                    n_no_improvement = 0
                    prev_val_loss = val_loss
            if n_no_improvement >= early_stopping_patience:
                break
