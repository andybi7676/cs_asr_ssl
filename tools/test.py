import numpy as np
import torch
import glob
from text import load_text_encoder
import os
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import yaml
import torchaudio

# from textgrid import TextGrid
# bucket_path = '/home/b07502072/cs_ssl/cs_asr_ssl/data/len_for_bucket/splitted-seame'
# data_path = ''
# out_path = './data/valid_names/splitted-seame/'
# load_valid = out_path
# splits = [ 'train', 'dev', 'dev-man', 'dev-sge' ]

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# sample_rate = 16000

# def load_wav(wav_path):
#     wav, sr = torchaudio.load(wav_path)
#     assert sr == sample_rate, f'Sample rate mismatch: real {sr}, config {sample_rate}'
#     return wav.view(-1)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# upstream = torch.hub.load('s3prl/s3prl', 'wav2vec2_large_960').to(device)
# with torch.no_grad():
#     for split in splits:
#         split_valid_names = np.load(os.path.join(out_path, f'{split}.npy'))
#         lid = []
#         wavs = []
#         for valid_name in tqdm(split_valid_names[0:5], total=len(split_valid_names[0:5]), desc=f'ckecking split: {split}'):
#             # lid_a = torch.LongTensor(torch.load(f'{valid_name}_lid.pt'))
#             # print(lid_a)
#             # print(lid_a.size())
#             lid.append(torch.load(f'{valid_name}_balanced_lid.pt'))
#             # print(lid_b)
#             # print(lid_b.size())
#             # if lid_a.size()[-1] != lid_b.size()[0]:
#             wavs.append(torch.FloatTensor(load_wav(f'{valid_name}.wav')).to(device))
#         # wavs = [ torch.FloatTensor(wav).to(device) ]
#         features = upstream(wavs)['default']
#         for i in range(len(lid)):
#             print( lid[i].size()[0], len(features[i]))
#             if lid[i].size()[0] != len(features[i]):
#                 print(features[i][-1].sum().item(), features[i][100].sum().item())
                
            


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
# config_path = './configs/w2v2_base/w2v2_base_014.yml'
# with open(config_path, 'r') as yml_f:
#     config = yaml.safe_load(yml_f)

# config_asr = config.get('ASR')
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ckpt_path = '/home/b07502072/cs_ssl/cs_asr_ssl/results/wav2vec2_base_960/014/ASR/dev-best.ckpt'
# ckpt = torch.load(ckpt_path)
# upstream_asr = torch.hub.load('s3prl/s3prl', self.config_asr['UPSTREAM']['name']).to(self.device)
# featurizer_asr = Featurizer(upstream_asr, device, **config_asr['FEATURIZER']).to(device)
# downstream_asr = Downstream(featurizer_asr.upstream_dim, **config_asr['DOWNSTREAM']).to(device)
# trainable_params = list(featurizer_asr.parameters()) + list(downstream_asr.parameters())
# print(len(trainable_params))
ckpt_path = '/home/b07502072/cs_ssl/iven/hubert_asr/result/downstream/pseudo_base/dev-clean-best.ckpt'
ckpt = torch.load(ckpt_path)
print(ckpt.keys())

# print(ckpt['Downstream'].keys())
# print(ckpt['CTC_Featurizer'].keys())