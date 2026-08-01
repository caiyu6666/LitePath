import torch.nn as nn
import torch.nn.functional as F


class AdaPatchSelector(nn.Module):
    def __init__(self, in_dim=384, out_dim=1):
        super(AdaPatchSelector, self).__init__()
        hidden_dim1 = 512
        hidden_dim2 = 128
        self.score_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(hidden_dim1, hidden_dim2),
            nn.Tanh(),
            nn.Linear(hidden_dim2, out_dim),
        )

    def forward(self, x):
        return self.score_net(x)
