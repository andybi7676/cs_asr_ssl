import logging
import os
import random
#-------------#
import pandas as pd
import numpy as np
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
    def __init__(self, splits, bucket_path='./', data_path='', bucket_size=1, **kwargs) -> None:
        super(LID_Dataset, self).__init__()

        assert os.path.exists(bucket_path), 'Bucket path does not exist.'
        self.bucket_path = bucket_path
        self.splits = splits
        self.sample_rate = SAMPLE_RATE

        table_list = []
        self.valid_names = []
        valid_path = kwargs.get('load_valid', False)
    
        for split in self.splits:
            if os.path.isfile(os.path.join(valid_path, f'{split}.npy')):
                valids_in_split = np.load(os.path.join(valid_path, f'{split}.npy')).tolist()
                # if 'dev' in split: valids_in_split = valids_in_split
                tqdm.write(f'[ LID_dataset ] - loaded valid names of split {split}, {len(valids_in_split)} valid names were found')
                self.valid_names += valids_in_split
            else:
                df = pd.read_csv(os.path.join(bucket_path, f'{split}.csv'))
                # table_list.append(df)
                # table_list = pd.concat(table_list)

                f_names = df['file_path'].tolist()
                f_len = df['length'].tolist()

                for f in tqdm(f_names, total=len(f_names), desc=f'Validating names for split: {split}'):
                    name = os.path.join(data_path, f.split('.')[0])
                    if os.path.isfile(name.replace('SEAME', 'SEAME_LID')+'_lid.pt') and os.path.isfile(name+'.wav'):
                        self.valid_names.append(name)
        tqdm.write(f'[ LID_dataset ] - dataset prepared')
        # print(self.X)

        
    def _load_wav(self, wav_path):
        wav, sr = torchaudio.load(wav_path)
        assert sr == self.sample_rate, f'Sample rate mismatch: real {sr}, config {self.sample_rate}'
        return wav.view(-1)
    
    def __len__(self):
        return len(self.valid_names)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        wav = [ self._load_wav(self.valid_names[index]+'.wav').numpy() ]
        label = [ torch.load(self.valid_names[index].replace('SEAME', 'SEAME_LID')+'_lid.pt').squeeze() - 1 ]
        return (wav, label) # bucketing, return ((wavs, labels))
    
    def collate_fn(self, items):
        assert len(items) == 1
        return items[0][0], items[0][1] # hack bucketing, return (wavs, labels)
