import sys
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import numpy as np
from torch import Tensor
import torch.nn.functional as F

SAMPLE_RATE = 16000

class RNNLayer(nn.Module):
    ''' RNN wrapper, includes time-downsampling'''

    def __init__(self, input_dim, module, bidirection, dim, dropout, layer_norm, sample_rate, proj):
        super(RNNLayer, self).__init__()
        # Setup
        rnn_out_dim = 2 * dim if bidirection else dim
        self.out_dim = rnn_out_dim
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.sample_rate = sample_rate
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
        if self.sample_rate > 1:
            output, x_len = downsample(output, x_len, self.sample_rate, 'drop')

        if self.proj:
            output = torch.tanh(self.pj(output))

        return output, x_len

class iven_RNNs(nn.Module):
    def __init__(self,
        input_size,
        output_size,
        lid_output_size,
        upstream_rate,
        module,
        bidirection,
        dim,
        dropout,
        layer_norm,
        proj,
        sample_rate,
        sample_style,
        total_rate = 320,
    ):
        super(RNNs, self).__init__()
        latest_size = input_size

        self.sample_rate = 1 if total_rate == -1 else round(total_rate / upstream_rate)
        self.sample_style = sample_style
        if sample_style == 'concat':
            latest_size *= self.sample_rate
            input_size *= self.sample_rate

        self.rnns = nn.ModuleList()
        for i in range(len(dim)):
            rnn_layer = RNNLayer(
                latest_size,
                module,
                bidirection,
                dim[i],
                dropout[i],
                layer_norm[i],
                sample_rate[i],
                proj[i],
            )
            self.rnns.append(rnn_layer)
            latest_size = rnn_layer.out_dim
        
        self.linear = nn.Linear(latest_size, output_size)
        
        self.lid_rnn = RNNLayer(input_size, module, bidirection, 1024, 0.2, False, 1, False)
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
        if self.sample_rate > 1:
            x, x_len = downsample(x, x_len, self.sample_rate, self.sample_style)
            l, l_len = downsample(l, l_len, self.sample_rate, self.sample_style)

        for rnn in self.rnns:
            x, x_len = rnn(x, x_len)
        logits = self.linear(x)
        
        l, l_len = self.lid_rnn(l, l_len)
        lid_logits = self.lid_linear(l)

        return logits, x_len, lid_logits, l_len        

class iven_Featurizer(nn.Module):
    def __init__(
        self,
        upstream: UpstreamBase,
        feature_selection: str = "hidden_states",
        upstream_device: str = "cuda",
        **kwargs,
    ):
        super().__init__()
        self.feature_selection = feature_selection
        self.name = f"Featurizer for {upstream.__class__}"

        show(
            f"[{self.name}] - The input upstream is only for initialization and not saved in this nn.Module"
        )

        # This line is necessary as some models behave differently between train/eval
        # eg. The LayerDrop technique used in wav2vec2
        upstream.eval()

        paired_wavs = [torch.randn(SAMPLE_RATE).to(upstream_device)]
        paired_features = upstream(paired_wavs)

        feature = self._select_feature(paired_features)
        if isinstance(feature, (list, tuple)):
            self.layer_num = len(feature)
            show(
                f"[{self.name}] - Take a list of {self.layer_num} features and weighted sum them."
            )
            self.weights = nn.Parameter(torch.zeros(self.layer_num))
            feature, weights = self._weighted_sum([f.cpu() for f in feature], False)
        else:
            feature = feature.cpu()

        self.output_dim = feature.size(-1)
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
            show(
                f"[{self.name}] - feature_selection = {self.feature_selection} is not supported for this upstream.",
                file=sys.stderr,
            )
            show(
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
        paired_wavs: List[Tensor],
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

        return UpstreamBase.tolist(paired_wavs, feature), weights
