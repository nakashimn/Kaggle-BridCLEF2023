import torch
from torch import nn

################################################################################
# FocalLoss
################################################################################

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        probas = pred.softmax(dim=1)
        loss = -(target*((1-probas)**self.gamma)*(probas.log())).mean()
        return loss
