import torch.nn as nn
import torch


class Downstream(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    

class Featurizer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

class Classifier(nn.Module):
    def __init__(self, input_dim=1024, **kwargs):
        super().__init__()
    
        self.net1 = nn.Sequential(
            # nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        
        self.conv_net = nn.Sequential(
            # nn.BatchNorm1d(input_dim),
            nn.Conv2d(1, 1, (3, 1), padding=(1, 0)),
        )

    def forward(self, X):
        if len(X.size()) == 4 and X.size()[1] == 1:
            conv_X = self.conv_net(X)
            return conv_X
        else:
            logits = self.net1(X)
            return logits