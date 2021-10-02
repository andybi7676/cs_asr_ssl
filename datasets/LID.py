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

SAMPLE_RATE = 16000
HALF_BATCHSIZE_TIME = 2000

class LID_Dataset(Dataset):
    
    def __init__(self, split, bucket_size=1, data_path='', bucket_path='./', **kwargs):
        super().__init__()
        
        self.sample_rate = SAMPLE_RATE
        self.data_path = data_path
        self.split_sets = kwargs[split]
        self.lid_fname = kwargs.get('lid_fname', '_balanced_lid')
        valid_path = kwargs.get('load_valid', False)
        print(f'split_sets: {self.split_sets}')

        # Read table for bucketing
        print(bucket_path)
        assert os.path.isdir(bucket_path), 'Please first run `python3 preprocess/generate_len_for_bucket.py -h` to get bucket file.'

        # Wavs
        table_list = []
        for item in self.split_sets:
            file_path = os.path.join(bucket_path, item + ".csv")
            if os.path.exists(file_path):
                table_list.append(
                    pd.read_csv(file_path)
                )
            else:
                logging.warning(f'{item} is not found in bucket_path: {bucket_path}, skipping it.')

        table_list = pd.concat(table_list)
        table_list = table_list.sort_values(by=['length'], ascending=False)

        lid_valid_names = []
        for split in self.split_sets:
            if os.path.isfile(os.path.join(valid_path, f'{split}.npy')):
                valids_in_split = np.load(os.path.join(valid_path, f'{split}.npy')).tolist()
                if 'dev' in split: valids_in_split = valids_in_split
                tqdm.write(f'[ LID_dataset ] - loaded valid names of split {split}, {len(valids_in_split)} valid names were found')
                lid_valid_names += valids_in_split
            else:
                df = pd.read_csv(os.path.join(bucket_path, f'{split}.csv'))
                # table_list.append(df)
                # table_list = pd.concat(table_list)

                f_names = df['file_path'].tolist()
                f_len = df['length'].tolist()

                for f in tqdm(f_names, total=len(f_names), desc=f'Validating names for split: {split}'):
                    name = os.path.join(data_path, f.split('.')[0])
                    if os.path.isfile(name + f'{self.lid_fname}.pt') and os.path.isfile(name + '.wav'):
                        lid_valid_names.append(name)
        
        tqdm.write(f'[ LID_dataset ] - dataset prepared')

        X = table_list['file_path'].tolist()
        X_lens = table_list['length'].tolist()
        # if split == 'test' or split == 'dev':
        #     X = X[:1000]
        #     X_lens = X_lens[:1000]

        assert len(X) != 0, f"0 data found for {split}"

        # Transcripts
        # Y = self._load_transcript(X)
        Z = { self._parse_x_name(x): x for x in lid_valid_names }
        x_names = set([self._parse_x_name(x) for x in X])
        # y_names = set(Y.keys())
        z_names = set(Z.keys())
        usage_list = list(x_names & z_names)
        self.Z = {key: Z[key] for key in usage_list}
        tqdm.write(f'[ LID_DATASET ] - Found {len(usage_list)} valid names for {split}')

        # Use bucketing to allow different batch sizes at run time
        self.X = []
        batch_x, batch_len = [], []

        for x, x_len in tqdm(zip(X, X_lens), total=len(X), desc=f'Loading LID dataset {split}', dynamic_ncols=True):
            if self._parse_x_name(x) in usage_list:
                batch_x.append(x)
                batch_len.append(x_len)
                
                # Fill in batch_x until batch is full
                if len(batch_x) == bucket_size:
                    # Half the batch size if seq too long
                    if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME):
                        self.X.append(batch_x[:bucket_size//2])
                        self.X.append(batch_x[bucket_size//2:])
                    else:
                        self.X.append(batch_x)
                    batch_x, batch_len = [], []
        
        # Gather the last batch
        if len(batch_x) > 1:
            if self._parse_x_name(x) in usage_list:
                self.X.append(batch_x)

    def _parse_x_name(self, x):
        return x.split('/')[-1].split('.')[0]
    
    def _load_wav(self, wav_path):
        wav, sr = torchaudio.load(os.path.join(self.data_path, wav_path))
        if (sr != self.sample_rate):
            wav = self._resample(wav, sr)
            sr = self.sample_rate

        assert sr == self.sample_rate, f'Sample rate mismatch: real {sr}, config {self.sample_rate}'
        return wav.view(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        wav_batch = [self._load_wav(x_file).numpy() for x_file in self.X[index]]
        lid_batch = [ torch.load(self.Z[self._parse_x_name(x_file)] + f'{self.lid_fname}.pt') for x_file in self.X[index] ]
        return wav_batch, lid_batch # bucketing, return ((wavs, labels))

    def collate_fn(self, items):
        assert len(items) == 1
        return items[0][0], items[0][1] # hack bucketing, return (wavs, labels)