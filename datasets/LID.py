import logging
import os
import random
#-------------#
import pandas as pd
from tqdm import tqdm
#-------------#
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
#-------------#
import torchaudio
#-------------#
# from .TextGrid import TextGrid

SAMPLE_RATE = 16000
HALF_BATCHSIZE_TIME = 2000

class LID_Dataset(Dataset):
    def __init__(self, splits, bucket_path='./', data_path='', **kwargs) -> None:
        super(LID_Dataset, self).__init__()

        assert os.path.exists(bucket_path), 'Bucket path does not exist.'
        self.bucket_path = bucket_path
        self.splits = splits
        self.sample_rate = SAMPLE_RATE

        table_list = []
        for split in self.splits:
            df = pd.read_csv(os.path.join(bucket_path, f'{split}.csv'))
            table_list.append(df)
        table_list = pd.concat(table_list)

        f_names = table_list['file_path'].tolist()
        f_len = table_list['length'].tolist()

        self.X = [ os.path.join(data_path, f.split('.')[0]) for f in tqdm(f_names, total=len(f_names), desc='loading dataset') ]
        # print(self.X)

        
    def _load_wav(self, wav_path):
        wav, sr = torchaudio.load(wav_path)
        assert sr == self.sample_rate, f'Sample rate mismatch: real {sr}, config {self.sample_rate}'
        return wav.view(-1)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        wav = [ self._load_wav(self.X[index]+'.wav').numpy() ]
        label = [ torch.load(self.X[index]+'_lid.pt').squeeze() ]
        # print(f'wav size: {wav[0].size()}')
        # print(f'label size: {label[0].size()}')
        return (wav, label) # bucketing, return ((wavs, labels))
    
    def collate_fn(self, items):
        assert len(items) == 1
        return items[0][0], items[0][1] # hack bucketing, return (wavs, labels)
