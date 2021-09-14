import numpy as np
import torch
import glob

exp_path = './results/wav2vec2_base_960-LID-001'
ckpts = glob.glob(f'{exp_path}/states-*.ckpt')
print(ckpts)
for ckpt in ckpts:
    ckpt = torch.load(ckpt)
    print(ckpt)