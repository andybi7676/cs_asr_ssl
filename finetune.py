import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import numpy as np
from tqdm import tqdm
from datasets.LID import LID_Dataset
from datasets.ASR import ASR_Dataset
from datasets.joint import ALL_Dataset
from models.model import Downstream, Featurizer
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
import torch.nn.functional as F
import os
import math
import glob
import copy
import editdistance
import torchaudio
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from datasets.text import load_text_encoder
from tools.optim import get_optimizer
from tools.schedulers import get_scheduler
from collections import defaultdict
import matplotlib.pyplot as plt
from time import localtime, strftime

config_path = './configs/finetune/xlsr/xlsr_001.yml'

class Runner():
    def __init__(self, config):
        self.config = config
        self.id = self.config['id']
        self.exp_name = '/'.join(['finetune', self.config['UPSTREAM']['name'], self.id])
        self.outdir = f'./results/{self.exp_name}'
        if not os.path.exists(self.outdir): os.makedirs(self.outdir)
        time_str = strftime("%Y-%m-%d_%H-%M", localtime())
        with open(self.outdir+f'/{time_str}_config.yml', 'w') as yml_f:
            yaml.dump(self.config, yml_f, default_flow_style=False)
        self.writer = SummaryWriter(log_dir=self.outdir)
        self.dictionary = load_text_encoder(self.config['DATASET']['dict_mode'], self.config['DATASET']['dict_path'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.devices = range(torch.cuda.device_count())
        self.upstream = torch.hub.load('s3prl/s3prl', config['UPSTREAM']['name'])
        randn_wavs = [ torch.FloatTensor(np.random.rand(1000)) ]
        # print(randn_wavs)
        feature = self.upstream(randn_wavs)['default']
        self.upstream_dim = feature.size()[-1]
        self.specaug_asr = None
        if self.config.get('SPECAUG'):
            from tools.specaug import SpecAug
            self.specaug = SpecAug(**self.config["SPECAUG"])
        self.linear = nn.Linear(self.upstream_dim, self.dictionary.vocab_size)
        self.blank = self.dictionary.pad_idx
        self.asr_loss = nn.CTCLoss(
            blank=self.blank,
            zero_infinity = True
        )
        self.records = defaultdict(list)
        self.best_score = 100.
        self.decoder = None
        self.ckpt = None
        if self.config.get('load_ckpt', False):
            if self.config['load_ckpt'] == 'last':
                ckpt_pths = glob.glob(f'{self.outdir}/states-*.ckpt')
                ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
                if len(ckpt_pths) == 0:
                    print(f'No ckpt named as \'states-*.ckpt\' was found in \'{self.outdir}\'')
                else:
                    last_ckpt_pth = ckpt_pths[-1]
                    self.ckpt = torch.load(last_ckpt_pth)
            if self.config['load_ckpt'] == 'best':
                best_ckpt_pths = glob.glob(f'{self.outdir}/best*.ckpt')
                assert len(ckpt_pths) == 1
                self.ckpt = torch.load(best_ckpt_pths[0])
    
    def _get_optimizer(self, trainable_models, config ):
        total_steps = config['runner']['total_steps']
        optimizer_conf = config['optimizer']
        optimizer = get_optimizer(
            trainable_models, 
            total_steps,
            optimizer_conf
        )
        # self._load_weight(optimizer, 'Optimizer')
        return optimizer
    
    def _get_scheduler(self, optimizer, config):
        total_steps = config['runner']['total_steps']
        scheduler_conf = config['scheduler']
        scheduler = get_scheduler(
            optimizer,
            total_steps,
            scheduler_conf
        )
        # self._load_weight(scheduler, 'Scheduler')
        return scheduler

    def train(self):
        self.upstream.to(self.device)
        self.specaug.to(self.device)
        self.linear.to(self.device)
        if self.ckpt:
            self.upstream.load_state_dict(self.ckpt['Upstream'])
            self.linear.load_state_dict(self.ckpt['Linear'])
        
        for param in self.upstream.model.feature_extractor.parameters():
            param.requires_grad = False

        if not hasattr(self, 'train_dataloader'):
            self.train_dataset = ASR_Dataset('dev', self.dictionary, **self.config['DATASET'])
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=1, collate_fn=self.train_dataset.collate_fn, shuffle=True)
        
        if len(self.devices) > 1:
            tqdm.write(f'Using multi gpu, ids: {self.devices}')
            self.upstream = nn.DataParallel(self.upstream)
        
        trainable_models = [self.upstream, self.linear]
        # trainable_params = list(self.featurizer_asr.parameters()) + list(self.downstream_asr.parameters())
        optimizer = self._get_optimizer(trainable_models, self.config)
        if self.ckpt:
            optimizer.load_state_dict(self.ckpt['optimizer'])
        scheduler = None
        if self.config.get('scheduler', False):
            scheduler = self._get_scheduler(optimizer, self.config)
            if self.ckpt:
                scheduler.load_state_dict(self.ckpt['scheduler'])
        
        for batch_id, (wavs, labels) in enumerate(tqdm(self.train_dataloader, dynamic_ncols=True, total=len(self.train_dataloader), desc=f'training')):
            
            wavs, labels = [ torch.FloatTensor(wav).to(self.device) for wav in wavs ], [ torch.LongTensor(label).to(self.device) for label in labels ]
            features = self.upstream(wavs)['default']
            print(features.size())
            features, _ = self.specaug(features)

            features_len = torch.IntTensor([len(feat) for feat in features]).to('cpu')
            labels_len = torch.IntTensor([len(lb) for lb in labels]).to('cpu')
            features = pad_sequence(features, batch_first=True).to(self.device)
            labels = pad_sequence(labels, batch_first=True).to(self.device)

            print(features.size())
            # print(features[1].size())
            logits = self.linear(features)
            print(logits)
            print(logits.size())

            log_probs = nn.functional.log_softmax(logits, dim=-1)
            loss = self.asr_loss(
                log_probs.transpose(0, 1), # (N, T, C) -> (T, N, C)
                padded_labels,
                log_probs_len,
                labels_len,
            )
            print(loss)
            assert 1==2


def main():
    torchaudio.set_audio_backend('sox_io')
    with open(config_path, 'r') as yml_f:
        config = yaml.safe_load(yml_f)
    runner = Runner(config)
    if config['task'] == 'train':
        runner.train()
    


if __name__ == '__main__':
    main()