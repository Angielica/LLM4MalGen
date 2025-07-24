import torch
import torch.nn as nn

class Detector(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.fc = nn.ModuleList()

        prev_dim = self.in_channels
        for h_dim in self.hidden_channels:
            self.fc.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim

        self.out = nn.Linear(prev_dim, self.out_channels)

    def forward(self, emb):
        h = emb
        for layer in self.fc:
            h = layer(h)
            h = torch.relu(h)

        out = self.out(h)
        return torch.sigmoid(out).flatten()



