import numpy as np
import torch
import glob
from text import load_text_encoder
import os
import pandas as pd
from tqdm import tqdm
import torch
import torchaudio

# from textgrid import TextGrid
bucket_path = '/home/b07502072/cs_ssl/cs_asr_ssl/data/len_for_bucket/splitted-seame'
data_path = ''
out_path = './data/valid_names/splitted-seame/'
load_valid = out_path
splits = [ 'train' ]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sample_rate = 16000

def load_wav(wav_path):
    wav, sr = torchaudio.load(wav_path)
    assert sr == sample_rate, f'Sample rate mismatch: real {sr}, config {sample_rate}'
    return wav.view(-1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
upstream = torch.hub.load('s3prl/s3prl', 'wav2vec2_base_960').to(device)
with torch.no_grad():
    for split in splits:
        split_valid_names = np.load(os.path.join(out_path, f'{split}.npy'))
        for valid_name in tqdm(split_valid_names, total=len(split_valid_names), desc=f'ckecking split: {split}'):
            # lid_a = torch.LongTensor(torch.load(f'{valid_name}_lid.pt'))
            # print(lid_a)
            # print(lid_a.size())
            lid_b = torch.load(f'{valid_name}_balanced_lid.pt')
            # print(lid_b)
            # print(lid_b.size())
            # if lid_a.size()[-1] != lid_b.size()[0]:
            wav = load_wav(f'{valid_name}.wav')
            wav = [ torch.FloatTensor(wav).to(device) ]
            features = upstream(wav)['default']
            assert lid_b.size()[0] == features.size()[-2]
                
            


# dict_path = '/home/b07502072/cs_ssl/cs_asr_ssl/dicts/dict_9k.model'
# out_path = './dicts/dict_9k_id_to_text.txt'
# dictionary = load_text_encoder('subword', dict_path)

# print(dictionary.decode([112, 113]))
# # with open(out_path, 'w') as outf:
# #     for i in range(9000):
# #         outf.write(dictionary.decode([i]) + '\n')


# for param in upstream.model.feature_extractor.parameters():
#     param.requires_grad = False
# print(list(upstream.model.feature_extractor.parameters())[0:10])
# print(list(upstream.model.))
# inputs = torch.FloatTensor(np.zeros(3000))
# outs = upstream([ inputs ])['default']
# outs = outs.view(-1).sum() ** 2
# outs.backward()
# print(outs)
