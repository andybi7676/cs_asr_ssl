from re import I
import torch
import torchaudio
import os
import matplotlib.pyplot as plt
from ..datasets.LID import LID_Dataset

import pandas as pd
import numpy as np
import glob
import yaml
from collections import defaultdict

config_pth = './results/wav2vec2_base_960/001/LID'
choose = 'lid'
l2_norm_path = None

def caculate_l2_norm(upstream_name, dataset_config, split='test', outdir='./data/l2_norm'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    upstream = torch.hub.load('s3prl/s3prl', upstream_name).to(device)
    outdir = os.path.join(outdir, upstream_name)
    if not os.path.exists(outdir): os.makedirs(outdir)

    bucket_file = os.path.join(dataset_config['bucket_path'], f'{split}.csv')
    df = pd.read_csv(bucket_file)
    wav_files = df['file_path'].tolist()
    upstream.eval()
    records = defaultdict(list)
    for wav_f in wav_files:
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

def main():
    with open(config_pth, 'r') as yml_f:
        config = yaml.safe_load(yml_f)
    
    if l2_norm_path == None:
        caculate_l2_norm(config['UPSTREAM']['name'], config['DATASET'], split='test', out_dir='./data/l2_norm')

if __name__ == '__main__':
    main()