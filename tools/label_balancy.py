import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm

# bucket_path = '/home/b07502072/cs_ssl/iven/hubert_asr/len_for_bucket/SEAME'
valid_path = './data/valid_names/SEAME'

splits = ['train', 'dev-sge', 'dev-man']
class_types = ['<pad>', '<sil>', '<chi>', '<eng>']
distribution = {}
for split in splits:
    # df = pd.read_csv(f'{bucket_path}/{split}.csv')
    distribution[split] = {}
    valid_names = np.load(f'{valid_path}/{split}.npy')
    for c in class_types:
        distribution[split][c] = 0
    for name in tqdm(valid_names, total=len(valid_names), desc=f'counting split {split}'):
        label_fpath = name.replace('SEAME', 'SEAME_LID') + '_lid.pt'
        label = torch.LongTensor(torch.load(label_fpath).squeeze())
        for i in range(4):
            counts_i = (label == i).sum().item()
            distribution[split][class_types[i]] += counts_i
        # print(distribution)
        # assert 1==2
print(distribution)