from numpy import log
from datasets.LID import SAMPLE_RATE
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

import torch.nn.functional as F

SAMPLE_RATE = 16000

class Featurizer(nn.Module):
    def __init__(self, upstream, device, **kwargs):
        super().__init__()

        upstream.eval()

        paired_wavs = [torch.randn(SAMPLE_RATE).to(device)]
        paired_features = upstream(paired_wavs)

        feature = paired_features['hidden_states']
        self.upstream_dim = feature[0].size(-1)
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

class RNNLayer(nn.Module):
    ''' RNN wrapper, includes time-downsampling'''

    def __init__(self, input_dim, module, bidirection, dim, dropout, layer_norm, proj):
        super(RNNLayer, self).__init__()
        # Setup
        rnn_out_dim = 2 * dim if bidirection else dim
        self.out_dim = rnn_out_dim
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.proj = proj

        # Recurrent layer
        self.layer = getattr(nn, module.upper())(
            input_dim, dim, bidirectional=bidirection, num_layers=1, batch_first=True)

        # Regularizations
        if self.layer_norm:
            self.ln = nn.LayerNorm(rnn_out_dim)
        if self.dropout > 0:
            self.dp = nn.Dropout(p=dropout)

        # Additional projection layer
        if self.proj:
            self.pj = nn.Linear(rnn_out_dim, rnn_out_dim)

    def forward(self, input_x, x_len):
        # Forward RNN
        if not self.training:
            self.layer.flatten_parameters()

        input_x = pack_padded_sequence(input_x, x_len, batch_first=True, enforce_sorted=False)
        output, _ = self.layer(input_x)
        output, x_len = pad_packed_sequence(output, batch_first=True)

        # Normalizations
        if self.layer_norm:
            output = self.ln(output)
        if self.dropout > 0:
            output = self.dp(output)

        # Perform Downsampling
        if self.proj:
            output = torch.tanh(self.pj(output))

        return output, x_len


class RNNs(nn.Module):
    def __init__(self,
        input_size,
        dim,
        dropout,
        layer_norm,
        proj,
        output_size=4,
        module='LSTM',
        bidirection=True,
        **kwargs
    ):
        super(RNNs, self).__init__()
        latest_size = input_size

        self.rnns = nn.ModuleList()
        for i in range(len(dim)):
            rnn_layer = RNNLayer(
                latest_size,
                module,
                bidirection,
                dim[i],
                dropout[i],
                layer_norm[i],
                proj[i]
            )
            self.rnns.append(rnn_layer)
            latest_size = rnn_layer.out_dim

        self.linear = nn.Linear(latest_size, output_size)
    
    def forward(self, x, x_len):
        r"""
        Args:
            x (torch.Tensor): Tensor of dimension (batch_size, input_length, num_features).
            x_len (torch.IntTensor): Tensor of dimension (batch_size).
        Returns:
            Tensor: Predictor tensor of dimension (batch_size, input_length, number_of_classes).
        """
        for rnn in self.rnns:
            x, x_len = rnn(x, x_len)

        logits = self.linear(x)
        return logits, x_len 

class Downstream(nn.Module):
    def __init__(self, upstream_dim, model_type='RNNs', **downstream_config):
        super().__init__()
        print(downstream_config)
        self.projector = nn.Linear(upstream_dim, downstream_config['proj_dim'])
        model_cls = eval(model_type)
        model_conf = downstream_config[model_type]

        self.model = model_cls(
            downstream_config['proj_dim'],
            **model_conf
        )

        self.objective = nn.CrossEntropyLoss()
    
    def forward(self, features, labels):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to('cpu')
        labels_len = torch.IntTensor([len(lb) for lb in labels]).to('cpu')
        features = pad_sequence(features, batch_first=True).to(device)
        labels = pad_sequence(labels, batch_first=True).to(device)

        features = self.projector(features)
        logits, log_probs_len = self.model(features, features_len)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        # print(log_probs.size())

        loss = self.objective(
                log_probs.transpose(-1, 1),
                labels,
            )
        
        return loss



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