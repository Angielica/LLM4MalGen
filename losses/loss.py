import torch
import torch.nn as nn

class DetectionLoss(nn.Module):
    def __init__(self):
        super(DetectionLoss, self).__init__()

        self.criterion = nn.BCELoss()

    def forward(self, y_pred, y_true):
        loss = self.criterion(y_pred, y_true)
        return loss