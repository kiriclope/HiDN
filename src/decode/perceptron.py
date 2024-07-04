import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier

class Perceptron(nn.Module):
    def __init__(self, num_features, dropout_rate=0.5):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(num_features, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(x)
        hidden = self.linear(x)
        return hidden

class RegularizedNet(NeuralNetClassifier):
    def __init__(self, module, lbd=0.01, alpha=0.01, l1_ratio=0.5, **kwargs):
        self.alpha = alpha  # Regularization strength
        self.l1_ratio = l1_ratio # Balance between L1 and L2 regularization

        super().__init__(module, **kwargs)

    def get_loss(self, y_pred, y_true, X=None, training=False):
        # Call super method to compute primary loss
        loss = super().get_loss(y_pred, y_true, X=X, training=training)

        if self.alpha>0:
            elastic_net_reg = 0
            for param in self.module_.parameters():
                elastic_net_reg += self.alpha * self.l1_ratio * torch.sum(torch.abs(param))
                elastic_net_reg += self.alpha * (1 - self.l1_ratio) * torch.sum(param ** 2) / 2

        # Add the elastic net regularization term to the primary loss
        return loss + elastic_net_reg
