from numpy import log
import numpy as np
import torch.nn as nn
from typing import Callable, List, Dict, Tuple, Union
from torch import Tensor
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
        self.model_type = kwargs['type']
        self.layer_norm = kwargs.get('layer-norm', False)


        self.upstream_dim = feature[0].size(-1)
        if self.upstream_dim == 240:
            print(f'[ Featurizer ] - upstream is fbank, unpack the features')
        else:
            if isinstance(feature, (list, tuple)):
                self.layer_num = len(feature)
                print(f'[ Featurizer ] - layer-norm: {self.layer_norm}')
                chosen_layers = kwargs.get('choose_layer', False)
                if chosen_layers:
                    start_idx, end_idx = chosen_layers
                    assert start_idx > 0 and end_idx <= self.layer_num
                    self.start_idx, self.end_idx = start_idx, end_idx
                    print(
                        f"[ Featurizer ] - Take a list from layer{self.start_idx} to layer{self.end_idx}(exclusive) of features and do {self.model_type} on them."
                    )
                    self.layer_num = self.end_idx - self.start_idx
                else:
                    print(
                        f"[ Featurizer ] - Take a list of {self.layer_num} features and do {self.model_type} on them."
                    )
                    self.start_idx = False
            else:
                raise ValueError('Invalid feature!')


        self.net_names = []

        if self.layer_norm:
            self.layer_norm = nn.LayerNorm(self.upstream_dim, elementwise_affine=not (kwargs.get('type') == 'weighted-sum'))
            self.net_names.append('layer_norm')
        
        if kwargs.get('type') == 'weighted-sum':
            self.f_type = 'weighted-sum'
            self.weights = nn.Parameter(torch.zeros(self.layer_num))
            self.net_names.append('_weighted_sum')
        
        if kwargs.get('type') == 'DNN':
            self.f_type = 'DNN'
            dnn_config = kwargs['DNN']
            self.dnn = DNN(
                self.upstream_dim,
                self.layer_num,
                **dnn_config
            )
            self.net_names.append('dnn')
        
        print(f'[ Featurizer ] - nets: {self.net_names}')

    def _weighted_sum(self, feature):
            
        _, *origin_shape = feature.shape
        feature = feature.view(self.layer_num, -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature
    
    def forward(self, feature):
        if self.upstream_dim == 240:
            return feature[0]
        if self.start_idx:
            feature = feature[self.start_idx:self.end_idx]
        assert self.layer_num == len(feature), f"{self.layer_num} != {len(feature)}"
        B, T, C = feature[0].size()
        if self.f_type == 'weighted-sum':
            feature = torch.stack(feature, dim=0)
        else:
            feature = torch.stack(feature, dim=2)
        for net_name in self.net_names:
            feature = eval(f'self.{net_name}')(feature)
        # if self.layer_norm:
        #     feature = self.layer_norm(feature)
        # return self._weighted_sum(feature)
        assert tuple(feature.size()[0:2]) == (B, T)
        return feature

# class Conv(nn.Module):
#     def __init__(self, upstream_dim, layer_num, )

class DNN(nn.Module):
    def __init__(self, upstream_dim, layer_num, dims, batch_norm, act, **kwargs):
        super().__init__()

        prev_dim = upstream_dim * layer_num
        self.net = nn.ModuleList()
        for i in range(len(dims)):
            if batch_norm[i]:
                self.net.append(nn.BatchNorm1d(prev_dim))
            self.net.append(nn.Dropout(p=0.1))
            self.net.append(nn.Linear(prev_dim, dims[i]))
            if act[i]:
                self.net.append(eval(f'nn.{act[i]}()'))
            prev_dim = dims[i]
        if prev_dim != upstream_dim:
            self.net.append(nn.Linear(prev_dim, upstream_dim))
        self.out_dim = upstream_dim
    
    def forward(self, X):
        # print(X.shape)
        B, T, L, C = tuple(X.size())
        X = X.view(B*T, L*C)
        for layer in self.net:
            X = layer(X)
            # print(X.size())
        X = X.view(B, T, self.out_dim)
        # print(X.shape)
        # assert 1==2
        return X


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
        output_size,
        module='LSTM',
        bidirection=True,
        **kwargs
    ):
        super(RNNs, self).__init__()
        latest_size = input_size
        self.kwargs = kwargs

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
        if self.kwargs.get('rnn_no_grad', False):
            with torch.no_grad():
                for rnn in self.rnns:
                    x, x_len = rnn(x, x_len) 
        else:
            for rnn in self.rnns:
                x, x_len = rnn(x, x_len)

        logits = self.linear(x)
        return logits, x_len 

class Downstream(nn.Module):
    def __init__(self, feature_dim, model_type='RNNs', **downstream_config):
        super().__init__()
        # print(downstream_config)
        self.model_type = model_type
        self.project_dim = downstream_config['proj_dim']
        self.projector = nn.Linear(feature_dim, downstream_config['proj_dim'])
        if model_type == 'FC': 
            return 
        model_cls = eval(model_type)
        model_conf = downstream_config[model_type]

        self.model = model_cls(
            downstream_config['proj_dim'],
            **model_conf
        )

    def forward(self, features, labels):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to('cpu')
        # print(features_len)
        labels_len = torch.IntTensor([len(lb) for lb in labels]).to('cpu')
        features = pad_sequence(features, batch_first=True).to(device)
        labels = pad_sequence(labels, batch_first=True).to(device)

        if self.model_type == 'FC':
            logits = self.projector(features)
            return logits, labels, features_len, labels_len
        features = self.projector(features)
        logits, logits_len = self.model(features, features_len) # tensor(N, T, C)
        # log_probs = nn.functional.log_softmax(logits, dim=-1)
        # print(log_probs.size())
        return logits, labels, logits_len, labels_len

        # loss = self.objective(
        #         logits.transpose(-1, 1), # tensor(N, C, T)
        #         labels,
        #     )
        
        # # loss = loss / logits.size()[1]
        # pred = logits.transpose(-1, 1).argmax(dim=1) # tensor(N, T)
        # acc = (pred == labels).type(torch.float).sum().item()
        
        # return acc, loss, logits.size()[1], pred.tolist()


class Linear(nn.Module):
    def __init__(self, input_dim, dim, batch_norm, act, output_size, **kwargs):
        super().__init__()

        prev_dim = input_dim
        self.net = nn.ModuleList()
        for i in range(len(dim)):
            if batch_norm[i]:
                self.net.append(nn.BatchNorm1d(prev_dim))
            self.net.append(nn.Linear(prev_dim, dim[i]))
            if act[i]:
                self.net.append(eval(f'nn.{act[i]}()'))
            prev_dim = dim[i]
        # if prev_dim != output_size:
        self.net.append(nn.Linear(prev_dim, output_size))
        self.out_dim = output_size

    def forward(self, X, X_len):
        N, T, C = X.size()[0], X.size()[1], X.size()[2]
        X = X.view(N*T, C)
        for layer in self.net:
            X = layer(X)
        logits = X.view(N, T, self.out_dim)
        return logits, None

class CNN(nn.Module):
    def __init__(self, input_dim, filter_nums, output_size, widths, **kwargs):
        super().__init__()

        self.net = nn.ModuleList()
        filter_total_nums = 0
        for i in range(len(widths)):
            filter_num = filter_nums[i]
            filter_total_nums += filter_num
            width = widths[i]
            self.net.append(nn.Conv2d(1, filter_num, (width, input_dim), padding=(width//2, 0) ) )
        if filter_total_nums != output_size:
            self.linear = nn.Linear(filter_total_nums, output_size)
        else: self.linear = None
        self.out_dim = output_size
    
    def forward(self, X, X_len):
        N, T, C = X.size()[0], X.size()[1], X.size()[2]
        # print(X.size())
        X = X.view(N, 1, T, C)
        Y = [ conv_net(X) for conv_net in self.net ]
        if len(Y) > 1:
            Y = torch.stack(Y, dim=1)
            Y = Y.view(Y.shape[0], Y.shape[1]*Y.shape[2], Y.shape[3], Y.shape[4])
            # print(Y.shape)
        else:
            Y = Y[0]
            # print(X)
        Y = Y.transpose(1, 2).squeeze(dim=-1)
        if self.linear:
            Y = self.linear(Y)
        
        logits = Y.view(N, T, self.out_dim)
        return logits, None

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

class Iven_RNNs(nn.Module):
    def __init__(self,
        input_size,
        output_size,
        lid_output_size,
        module,
        bidirection,
        dim,
        dropout,
        layer_norm,
        proj,
    ):
        super().__init__()
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
                proj[i],
            )
            self.rnns.append(rnn_layer)
            latest_size = rnn_layer.out_dim
        
        self.linear = nn.Linear(latest_size, output_size)
        
        self.lid_rnn = RNNLayer(input_size, module, bidirection, 1024, 0.2, False, False)
        self.lid_linear = nn.Linear(latest_size, lid_output_size)

        
    
    def forward(self, x, l, x_len, l_len):
        r"""
        Args:
            x (torch.Tensor): Tensor of dimension (batch_size, input_length, num_features).
            x_len (torch.IntTensor): Tensor of dimension (batch_size).
        Returns:
            Tensor: Predictor tensor of dimension (batch_size, input_length, number_of_classes).
        """
        # Perform Downsampling
        for rnn in self.rnns:
            x, x_len = rnn(x, x_len)
        logits = self.linear(x)
        
        l, l_len = self.lid_rnn(l, l_len)
        lid_logits = self.lid_linear(l)

        return logits, x_len, lid_logits, l_len        

class Iven_Featurizer(nn.Module):
    def __init__(
        self,
        upstream,
        device,
        feature_selection: str = "hidden_states",
        **kwargs,
    ):
        super().__init__()
        self.feature_selection = "hidden_states"

        # This line is necessary as some models behave differently between train/eval
        # eg. The LayerDrop technique used in wav2vec2
        upstream.eval()

        paired_wavs = [torch.randn(SAMPLE_RATE).to(device)]
        paired_features = upstream(paired_wavs)

        feature = self._select_feature(paired_features)
        self.layer_num = len(feature)
        if isinstance(feature, (list, tuple)):
            self.layer_num = len(feature)
            print(
                f"[ Featurizer ] - Take a list of {self.layer_num} features and weighted sum them."
            )
            self.weights = nn.Parameter(torch.zeros(self.layer_num))
            feature, weights = self._weighted_sum([f.cpu() for f in feature], False)
        else:
            feature = feature.cpu()

        self.output_dim = feature.size(-1)
        self.upstream_dim = feature.size(-1)
        ratio = round(max(len(wav) for wav in paired_wavs) / feature.size(1))
        possible_rate = torch.LongTensor([160, 320])
        self.downsample_rate = int(
            possible_rate[(possible_rate - ratio).abs().argmin(dim=-1)]
        )

    def _select_feature(self, features):
        feature = features.get(self.feature_selection)

        if isinstance(feature, dict):
            feature = list(feature.values())

        if isinstance(feature, (list, tuple)) and len(feature) == 1:
            feature = feature[0]

        if feature is None:
            available_options = [key for key in features.keys() if key[0] != "_"]
            print(
                f"[{self.name}] - feature_selection = {self.feature_selection} is not supported for this upstream.",
                file=sys.stderr,
            )
            print(
                f"[{self.name}] - Supported options: {available_options}",
                file=sys.stderr,
            )
            raise ValueError
        return feature

    def _weighted_sum(self, feature, layer_norm):
        assert self.layer_num == len(feature), f"{self.layer_num} != {len(feature)}"
        stacked_feature = torch.stack(feature, dim=0)

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(self.layer_num, -1)
        
        norm_weights = F.softmax(self.weights, dim=-1)
        if not layer_norm:
            norm_feature = torch.sqrt(torch.sum(stacked_feature ** 2, -1))
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)
        if layer_norm:
            equiv_weights = norm_weights
        else:
            equiv_weights = (norm_weights * norm_feature) / torch.sum(norm_feature)

        return weighted_feature, equiv_weights

    def forward(
        self,
        # paired_wavs: List[Tensor],
        paired_features: Dict[str, Union[Tensor, List[Tensor], Dict[str, Tensor]]],
        layer_norm=False
    ):
        feature = self._select_feature(paired_features)
        weights = None
        if layer_norm:
            f_ = []
            for f in feature:
                f = nn.functional.layer_norm(f, f.size())
                f_.append(f)
            feature = tuple(f_)
        if isinstance(feature, (list, tuple)):
            feature, weights = self._weighted_sum(feature, layer_norm)

        return feature, weights

class Iven_Downstream(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, feature_dim, model_type='iven_RNNs', **downstream_config):

        super().__init__()
        self.upstream_dim = feature_dim

        self.projector = nn.Linear(feature_dim, downstream_config['proj_dim'])
        self.lid_projector = nn.Linear(feature_dim, downstream_config['proj_dim'])
        model_cls = eval(model_type)
        model_conf = downstream_config[model_type]
        self.model = model_cls(
            downstream_config['proj_dim'],
            **model_conf,
        )

    # Interface:
    def forward(self, features, lid_features, labels, **kwargs):
        
        device = features[0].device
        # labels = [torch.IntTensor(l) for l in labels]
        features_len = torch.IntTensor([len(feat) for feat in features]).to('cpu')
        labels_len = torch.IntTensor([len(label) for label in labels]).to('cpu')
        features = pad_sequence(features, batch_first=True).to(device=device)

        lid_features_len = torch.IntTensor([len(feat) for feat in lid_features])
        lid_features = pad_sequence(lid_features, batch_first=True).to(device=device)

        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=0,
        ).to(device=device)

        lid_features = self.lid_projector(lid_features)
        features = self.projector(features)
        logits, log_probs_len, lid_logits, lid_log_probs_len = self.model(features, lid_features, features_len, lid_features_len)
        lid_log_probs = nn.functional.log_softmax(lid_logits, dim=-1)

        _, lid_sil_logits, lid_chi_logits, lid_eng_logits = torch.split(lid_logits, [1, 1, 1, 1], -1)
        sil_logits, eng_logits_1, _logits, eng_logits_2, chi_logits = torch.split(logits, [3, 2248, 1, 30, 2718], -1)

        sil_logits = sil_logits + lid_sil_logits
        chi_logits = chi_logits + lid_chi_logits
        eng_logits_1 = eng_logits_1 + lid_eng_logits
        eng_logits_2 = eng_logits_2 + lid_eng_logits

        adj_logits = torch.cat((sil_logits, eng_logits_1, _logits, eng_logits_2, chi_logits), dim=-1)

        return adj_logits, lid_logits, labels, features_len, labels_len

    # interface