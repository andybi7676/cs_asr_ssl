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
from .text import load_text_encoder

SAMPLE_RATE = 16000
HALF_BATCHSIZE_TIME = 2000

class ASR_Dataset(Dataset):
    
    def __init__(self, split, dictionary, bucket_size=1, data_path='', bucket_path='./', **kwargs):
        super(ASR_Dataset, self).__init__()
        
        self.data_path = data_path
        self.sample_rate = SAMPLE_RATE
        self.split_sets = kwargs[split]
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

        X = table_list['file_path'].tolist()
        X_lens = table_list['length'].tolist()
        # if split == 'test' or split == 'dev':
        #     X = X[:1000]
        #     X_lens = X_lens[:1000]

        assert len(X) != 0, f"0 data found for {split}"

        # Transcripts
        Y = self._load_transcript(X)

        x_names = set([self._parse_x_name(x) for x in X])
        y_names = set(Y.keys())
        usage_list = list(x_names & y_names)
        Y = {key: Y[key] for key in usage_list}
        

        # dictionary, symbol list
        self.dictionary = dictionary

        self.Y = {
            k: self.dictionary.encode(v)
            for k, v in Y.items()
        }

        # Use bucketing to allow different batch sizes at run time
        self.X = []
        batch_x, batch_len = [], []

        for x, x_len in tqdm(zip(X, X_lens), total=len(X), desc=f'Loading ASR dataset {split}', dynamic_ncols=True):
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

    def _resample(self, wav, sr):
        resample = torchaudio.transforms.Resample(
                sr, self.sample_rate, resampling_method='sinc_interpolation'
            )
        return resample(wav)

    def _load_wav(self, wav_path):
        wav, sr = torchaudio.load(os.path.join(self.data_path, wav_path))
        if (sr != self.sample_rate):
            wav = self._resample(wav, sr)
            sr = self.sample_rate

        assert sr == self.sample_rate, f'Sample rate mismatch: real {sr}, config {self.sample_rate}'
        return wav.view(-1)

    def _load_transcript(self, x_list):
        """Load the transcripts for Librispeech"""

        trsp_sequences = {}
        split_spkr_chap_list = list(
            set(
                "/".join(x.split('/')[:-1]) for x in x_list
            )
        )

        for dir in split_spkr_chap_list:
            parts = dir.split('/')
            trans_path = f"trans.text"
            path = os.path.join(self.data_path, dir, trans_path)
            if os.path.exists(path) == False: print(path)
            assert os.path.exists(path)

            with open(path, "r") as trans_f:
                for line in trans_f:
                    lst = line.strip().split()
                    trsp_sequences[lst[0]] = (" ".join(lst[1:]))

        return trsp_sequences

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        wav_batch = [self._load_wav(x_file).numpy() for x_file in self.X[index]]
        label_batch = [np.array(self.Y[self._parse_x_name(x_file)], dtype=np.float32) for x_file in self.X[index]]
        return wav_batch, label_batch # bucketing, return ((wavs, labels))

    def collate_fn(self, items):
        assert len(items) == 1
        return items[0][0], items[0][1] # hack bucketing, return (wavs, labels)
