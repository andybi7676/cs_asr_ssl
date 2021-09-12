from datasets.LID import SAMPLE_RATE
import torch.nn as nn
import torch

import torch.nn.functional as F

SAMPLE_RATE = 16000
class Downstream(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    

class Featurizer(nn.Module):
    def __init__(self, upstream, device, **kwargs):
        super().__init__()

        upstream.eval()

        paired_wavs = [torch.randn(SAMPLE_RATE).to(device)]
        paired_features = upstream(paired_wavs)

        feature = paired_features['hidden_states']
        if isinstance(feature, (list, tuple)):
            self.layer_num = len(feature)
            print(
                f"[ Featurizer ] - Take a list of {self.layer_num} features and weighted sum them."
            )
        else:
            raise ValueError('Invalid feature!')

        self.weights = nn.Parameter(torch.zeros(self.layer_num))

    def _weighted_sum(self, feature):
        assert self.layer_num == len(feature), f"{self.layer_num} != {len(feature)}"
        stacked_feature = torch.stack(feature, dim=0)

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(self.layer_num, -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature
    
    def forward(self, feature):
        return self._weighted_sum(feature)


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