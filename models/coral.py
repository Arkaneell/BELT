import torch
from torch import nn
import torch.nn.functional as F
from config import NUM_CLASSES

class CoralHead(nn.Module):
    def __init__(self, in_dim, num_classes=NUM_CLASSES):
        super().__init__()
        self.num_thresholds = num_classes - 1
        self.fc = nn.Linear(in_dim, 1, bias=False)
        self.bias0 = nn.Parameter(torch.zeros(1))
        self.deltas = nn.Parameter(torch.ones(max(self.num_thresholds - 1, 0)))

    def forward(self, x):
        base = self.fc(x)
        biases = [self.bias0]
        b = self.bias0
        for i in range(self.num_thresholds - 1):
            b = b - F.softplus(self.deltas[i])
            biases.append(b)
        biases = torch.cat(biases)
        return base + biases.unsqueeze(0)

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        return self.loss_fn(logits.squeeze(-1), targets.float())

def decode_coral(logits, threshold=0.5):
    probabilities = torch.sigmoid(logits[:, 0])
    return (probabilities > threshold).long()
