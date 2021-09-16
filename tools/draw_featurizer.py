from re import I
import torch
import torchaudio
import os
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import glob
import yaml
from collections import defaultdict
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F

config_pth = './configs/w2v2_base_001.yml'
ckpt_dir = '/home/b07502072/cs_ssl/cs_asr_ssl/results/wav2vec2_base_960/001/LID'
choose = 'lid'
l2_norm_path = './data/l2_norm/wav2vec2_base_960.txt'

def caculate_l2_norm(upstream_name, dataset_config, split='test', outdir='./data/l2_norm'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    upstream = torch.hub.load('s3prl/s3prl', upstream_name).to(device)
    outdir = os.path.join(outdir, upstream_name)
    if not os.path.exists(outdir): os.makedirs(outdir)
    splits = dataset_config[split]

    bucket_path = dataset_config['bucket_path']
    table_list = []
    for item in splits:
        file_path = os.path.join(bucket_path, item + ".csv")
        if os.path.exists(file_path):
            table_list.append(
                pd.read_csv(file_path)
            )
        else:
            print(f'{item} is not found in bucket_path: {bucket_path}, skipping it.')

    table_list = pd.concat(table_list)
    table_list = table_list.sort_values(by=['length'], ascending=False)

    wav_files = table_list['file_path'].tolist()
    upstream.eval()
    records = defaultdict(list)
    for wav_f in tqdm(wav_files, total=len(wav_files), desc='calculating l2 norm ...'):
        wav, sr = torchaudio.load(wav_f)
        wav = [ wav.view(-1).to(device) ]
        with torch.no_grad():
            features = upstream(wav)
            features = features['hidden_states']
            print(f'total layer num: {len(features)}')
            for idx in range(len(features)):
                feature = features[idx][0]
                print(feature.size())
                norm = torch.norm(feature, dim=-1).mean().item()
                print(norm)
                assert 1==2, 'stop here'
                L = feature.size()[0]
                records[f'layer_{idx}_norm'].append()

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

def parse_l2_norm_data(l2_norm_path):
    norms = []
    with open(l2_norm_path, 'r') as f:
        data = f.readlines()
        for line in data:
            line = line.strip()
            if line != '':
                norms.append(float(line))
    return norms

def main():
    with open(config_pth, 'r') as yml_f:
        config = yaml.safe_load(yml_f)['LID']
    
    if l2_norm_path == None:
        caculate_l2_norm(config['UPSTREAM']['name'], config['DATASET'], split='test', outdir='./data/l2_norm')
    else:
        norms = parse_l2_norm_data(l2_norm_path)
        print(norms)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    upstream = torch.hub.load('s3prl/s3prl', upstream_name).to(device)
    featurizer = Featurizer(upstream, device).to(device)
    ckpt_pths = glob.glob(f'{ckpt_dir}/states-*.ckpt')
    ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
    last_ckpt_pth = ckpt_pths[-1]
    loaded_ckpt = torch.load(last_ckpt_pth)

    featurizer_state = torch
    featurizer.load_state_dict(loaded_ckpt['Featurizer_lid'])
    

if __name__ == '__main__':
    main()