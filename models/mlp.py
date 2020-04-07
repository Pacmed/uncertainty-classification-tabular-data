import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader


class MLPModule(nn.Module):
    def __init__(self, model_params):
        hidden_sizes = model_params.get("hidden_sizes", [100])
        input_size = model_params.get("input_size", 1)
        output_size = model_params.get("output_size", 1)
        dropout_rate = model_params.get("dropout_rate", 0.0)
        do_batch_norm = model_params.get("batch_norm", False)

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

    def forward(self, _input):
        return self.mlp(_input)


class SimpleDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class MLP:
    def __init__(self, model_params):
        self.model = MLPModule(model_params)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.class_weight = model_params.get("class_weight", True)

    def get_loss_fn(self, mean_y=None):
        if self.class_weight:
            if mean_y == 0:
                pos_weight = torch.tensor(0.0)
            elif mean_y == 1:
                pos_weight = torch.tensor(1.0)
            else:
                pos_weight = (1 - mean_y) / mean_y

        else:
            pos_weight = torch.tensor(1.0)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        return loss_fn

    def _initialize_dataloader(self, X_train, y_train, batch_size):
        train_set = SimpleDataset(torch.from_numpy(X_train),
                                  torch.from_numpy(y_train))
        self.train_loader = DataLoader(train_set, batch_size, shuffle=True)

    def validate(self, X_val, y_val):
        self.model.eval()
        y_pred = self.model(torch.tensor(X_val.float()))
        loss_fn = self.get_loss_fn(y_val.float().mean())
        val_loss = loss_fn(y_pred, y_val.float().view(-1, 1))
        return val_loss

    def train(self, X_train, y_train, X_val, y_val, training_params):
        batch_size = training_params.get('batch_size', 64)
        n_epochs = training_params.get('n_epochs', 10)
        early_stopping = training_params.get('early_stopping', True)
        early_stopping_patience = training_params.get('early_stopping_patience', 2)

        self._initialize_dataloader(X_train, y_train, batch_size)
        prev_val_loss = 100000
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
                print("Early stopping after %s epochs." % str(epoch))
                break
