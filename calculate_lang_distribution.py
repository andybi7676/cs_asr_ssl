import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import numpy as np
from tqdm import tqdm
from datasets.ASR import ASR_Dataset
import os
import math
import glob
import copy
from tools.optim import get_optimizer
from tools.schedulers import get_scheduler
from collections import defaultdict
import matplotlib.pyplot as plt
from time import localtime, strftime
import random

data_dir = '/home/b07502072/cs_ssl/data/splitted-seame'
data_sets = ['train', 'dev', 'dev-man', 'dev-sge']
s_types = ['Man', 'Eng', 'CS', 'Unk']

distribution = {}

def get_type(line):
    line = line.strip().split(' ')[1:]
    zh = False
    en = False
    for word in line:
        if word.isupper():
            en = True
        else:
            zh = True
    if zh and en:
        return 'CS'
    if en:
        return 'Eng'
    if zh:
        return 'Man'
    return 'Unk'

for d_set in data_sets:
    distribution[d_set] = {}
    for t in s_types:
        distribution[d_set][t] = 0
    trans_pths = glob.glob(f'{data_dir}/{d_set}/*/trans.text')
    print(distribution)
    for trans in tqdm(trans_pths, total=len(trans_pths), desc=f'counting {d_set}'):
        with open(trans, 'r') as trans_f:
            lines = trans_f.readlines()
            for line in lines:
                s_type = get_type(line)
                distribution[d_set][s_type] = distribution[d_set][s_type] + 1
    
print(distribution)

    