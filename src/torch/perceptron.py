import torch
import torch.nn as nn


class CustomBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, pos_weight=None, weight=None, reduction="mean"):
        super(CustomBCEWithLogitsLoss, self).__init__(
            weight=weight, reduction=reduction, pos_weight=pos_weight
        )

    def forward(self, input, target):
        target = target.view(-1, 1)  # Make sure target shape is (n_samples, 1)
        return super().forward(input.to(torch.float32), target.to(torch.float32))


class Perceptron(nn.Module):
    def __init__(self, num_features, dropout_rate=0.0):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(num_features, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X):
        weight_size = self.linear.weight.size()[-1]
        if weight_size != X.shape[1]:
            self.linear = nn.Linear(X.shape[1], 1).to(X.device)

        hidden = self.linear(self.dropout(X))

        return hidden


class MLP(nn.Module):
    def __init__(self, num_features, hidden_units=64, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.linear = nn.Linear(num_features, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.relu(self.linear(x))
        x = self.dropout(x)
        hidden = self.linear2(x)
        return hidden
