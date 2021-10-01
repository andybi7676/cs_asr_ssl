import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torchaudio

# from textgrid import TextGrid
bucket_path = '/home/b07502072/cs_ssl/cs_asr_ssl/data/len_for_bucket/splitted-seame'
data_path = ''
out_path = './data/valid_names/splitted-seame/'
load_valid = out_path
splits = [ 'train', 'dev', 'dev-man', 'dev-sge' ]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sample_rate = 16000

def load_wav(wav_path):
    wav, sr = torchaudio.load(wav_path)
    assert sr == sample_rate, f'Sample rate mismatch: real {sr}, config {sample_rate}'
    return wav.view(-1)

with torch.no_grad():
    upstream = torch.hub.load('s3prl/s3prl', 'fbank').to(device)

    for split in splits:
        split_valid_names = np.load(os.path.join(out_path, f'{split}.npy'))
        for i, valid_name in enumerate(tqdm(split_valid_names, total=len(split_valid_names), desc=f'processing split: {split}')):
            lid_lb = torch.LongTensor(torch.load(f'{valid_name}_lid.pt').squeeze()).to(device)
            wav = [ torch.FloatTensor(load_wav(valid_name+'.wav')).to(device) ]
            features = upstream(wav)
            features = features['hidden_states'][0][0]
            expect_size = features.size()[0]
            # print(expect_size)
            new_lb = lid_lb.unsqueeze(dim=-1)
            T, _ = new_lb.size()
            lid_lb = torch.cat((new_lb, new_lb), 1)
            lid_lb = lid_lb.view(2*T)
            # assert 1==2
            diff = lid_lb.size()[0] - expect_size
            if diff == 0: pass
            elif diff > 0:
                q, r = diff // 2, diff % 2
                if q > 0 :
                    lid_lb = lid_lb[q+r: -q]
                else:
                    lid_lb = lid_lb[r:]
            else:
                lid_lb = torch.cat((lid_lb, torch.LongTensor(np.zeros(-diff, dtype='int'))), 0)
            assert lid_lb.size()[0] == expect_size
            torch.save(lid_lb.to('cpu'), f'{valid_name}_expand_fbank_lid.pt')

# for split in splits:
#     table_list = []
#     df = pd.read_csv(os.path.join(bucket_path, f'{split}.csv'))
#     table_list.append(df)
#     table_list = pd.concat(table_list)

#     f_names = table_list['file_path'].tolist()
#     f_len = table_list['length'].tolist()
#     valid_names = []
#     for f in tqdm(f_names, total=len(f_names), desc='loading dataset'):
#         name = os.path.join(data_path, f.split('.')[0])
#         if os.path.isfile(name + '_lid.pt') and os.path.isfile(name+'.wav'):
#             valid_names.append(name)
#     if not os.path.exists(out_path): os.makedirs(out_path)
#     print(f'valid_num: {len(valid_names)} for {split}')
#     np.save(os.path.join(out_path, f'{split}.npy'), np.array(valid_names))