import os
import torch 

src = '/home/b07502072/cs_ssl/iven/hubert_asr/result/downstream/base_seame/states-96000.ckpt'
dest = './results/wav2vec2_base_960/002/ASR'
ckpt_name = src.split('/')[-1]
if not os.path.exists(dest): os.makedirs(dest)

ckpt = torch.load(src)
print(ckpt.keys())
new_ckpt = {
    'Optimizer': ckpt['Optimizer'],
    'Step':ckpt['Step'],
    'Epoch': ckpt['Epoch'],
    'Featurizer_asr': ckpt['Featurizer'],
    'Downstream_asr': ckpt['Downstream'],
    'Upstream_asr': 'wav2vec2_base_960'
}

torch.save(new_ckpt, dest+ckpt_name)