import torch
import torch.nn as nn

class CustomBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, pos_weight=None, weight=None, reduction='mean'):
        super(CustomBCEWithLogitsLoss, self).__init__(weight=weight, reduction=reduction, pos_weight=pos_weight)

    def forward(self, input, target):
        target = target.view(-1, 1)  # Make sure target shape is (n_samples, 1)
        return super().forward(input.to(torch.float32), target.to(torch.float32))
