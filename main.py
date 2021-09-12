import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np
from tqdm import tqdm
from datasets.LID import LID_Dataset
from models.model import Downstream, Featurizer
from torch.utils.tensorboard import SummaryWriter
import os

from tools.optim import get_optimizer

config_path = './configs/w2v2_base.yml'

class Runner():
    def __init__(self, config, args=None):
        
        self.exp_name = '-'.join([config['UPSTREAM']['name'], config['mission'], config['id']])
        # outdir = f'./results/{self.exp_name}'
        # if not os.path.exists(outdir): os.makedirs(outdir)
        # writer = SummaryWriter(log_dir=f'{self.exp_name}')
        self.config = config
        self.args = args

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.upstream = torch.hub.load('s3prl/s3prl', self.config['UPSTREAM']['name']).to(self.device)
        self.featurizer = Featurizer(self.upstream, self.device, **self.config['FEATURIZER']).to(self.device)
        self.downstream = Downstream(self.featurizer.upstream_dim, **self.config['DOWNSTREAM']).to(self.device)
        self.specaug = None
        if self.config.get('SPECAUG'):
            from tools.specaug import SpecAug
            self.specaug = SpecAug(**self.config["SPECAUG"])
            self.specaug.to(self.device)
        self.first_round = True

    def _get_optimizer(self, trainable_models):
        optimizer = get_optimizer(
            trainable_models, 
            self.config['hyperparams']['total_steps'],
            self.config['optimizer']
        )
        # self._load_weight(optimizer, 'Optimizer')
        return optimizer

    def train_LID(self):

        dataset_config = self.config['DATASET']
        splits = dataset_config['train']
        self.train_dataset = LID_Dataset(splits, **dataset_config)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=1, collate_fn=self.train_dataset.collate_fn, shuffle=True)
        
        pbar = tqdm(total=self.config['hyperparams']['total_steps'], dynamic_ncols=True, desc='overall')
        
        self.upstream.eval()
        self.featurizer.train()
        self.downstream.train()
        trainable_models = [self.featurizer, self.downstream]
        trainable_params = [self.featurizer.parameters(), self.downstream.parameters()]

        optimizer = self._get_optimizer(trainable_models)
        print(optimizer)
        if self.config.get('scheduler'):
            scheduler = self._get_scheduler(optimizer)
        
        epoch = 0
        backward_steps = 0
        gradient_accumulate_steps = self.config['hyperparams']['gradient_accumulate_steps']
        # self.train_dataloader.sampler.set_epoch(epoch)
        while pbar.n < pbar.total:
            for batch_id, (wavs, labels) in enumerate(tqdm(self.train_dataloader, dynamic_ncols=True, total=len(self.train_dataloader), desc=f'training')):
                try:
                    if pbar.n >= pbar.total:
                        break
                    global_step = pbar.n + 1
                    # print(wavs)
                    wavs, labels = [torch.FloatTensor(wav).to(self.device) for wav in wavs], [ torch.LongTensor(label).to(self.device) for label in labels ]
                    # wavs => list(tensor(length))
                    
                    with torch.no_grad():
                        features = self.upstream(wavs)
                        features = features['hidden_states'] # features => tuple(tensor_layer1(N,T,C), ...tensor_layer_last(N,T,C))

                    features = self.featurizer(features) # features => tensor(N,T,C)
                    if self.specaug:
                        features, _ = self.specaug(features)  # features => list(tensor_1(T,C), ...tensor_n(T, C))
                    # revise label length
                    assert len(features) == len(labels), 'length of features and labels not consistent'
                    for idx, lb in enumerate(labels):
                        diff = lb.size()[0] - features[idx].size()[0]
                        assert diff >= 0, 'Unexpected event happened, ' 
                        q, r = diff // 2, diff % 2
                        if q > 0 :
                            labels[idx] = lb[q+r: -q]
                        else:
                            labels[idx] = lb[r:]
                    
                    loss = self.downstream(features, labels)

                    (loss / gradient_accumulate_steps).backward()
                    del loss
                    # assert 1==2
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print(f'[Runner] - CUDA out of memory at step {global_step}')
                        # if is_initialized():
                        # raise
                        with torch.cuda.device(self.device):
                            torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        raise
                        # continue
                    else:
                        raise
                # whether to accumulate gradient
                
                backward_steps += 1
                if backward_steps % gradient_accumulate_steps > 0:
                    # pbar.update(1)
                    continue

                # gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, self.config['hyperparams']['gradient_clipping'])

                # optimize
                if math.isnan(grad_norm):
                    print(f'[ Runner ] - grad norm is NaN at step {global_step}')
                else:
                    optimizer.step()
                optimizer.zero_grad()

                # adjust learning rate
                if scheduler:
                    scheduler.step()
                
                print(f'loss: {loss.item()}, global_step = {global_step}')
                # if global_step % self.config['hyperparams'][]

                pbar.update(1)
            epoch += 1


def main():
    with open(config_path, 'r') as yml_f:
        config = yaml.safe_load(yml_f)
    
    runner = Runner(config)
    
    if config['mission'] == 'LID':
        runner.train_LID()

if __name__ == '__main__':
    main()