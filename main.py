import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np
from tqdm import tqdm
from datasets.LID import LID_Dataset
from models.model import Downstream, Featurizer
from torch.utils.tensorboard import SummaryWriter
import os
import math
import glob

from tools.optim import get_optimizer

config_path = './configs/w2v2_base.yml'

class Runner():
    def __init__(self, config, args=None):
        
        self.mission = config['mission']
        self.task = config['task']
        self.exp_name = '/'.join([config['UPSTREAM']['name'], config['mission'], config['id']])
        self.outdir = f'./results/{self.exp_name}'
        self.init_ckpt = {}
        if not os.path.exists(self.outdir): os.makedirs(self.outdir)
        self.writer = SummaryWriter(log_dir=self.outdir)
        self.config = config
        self.config_lid = config.get('LID')
        self.config_asr = config.get('ASR')
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.mission == 'LID':
            self.upstream_lid = torch.hub.load('s3prl/s3prl', self.config_lid['UPSTREAM']['name']).to(self.device)
            self.featurizer_lid = Featurizer(self.upstream_lid, self.device, **self.config_lid['FEATURIZER']).to(self.device)
            self.downstream_lid = Downstream(self.featurizer_lid.upstream_dim, **self.config_lid['DOWNSTREAM']).to(self.device)
            self.specaug_lid = None
            if self.config_lid.get('SPECAUG'):
                from tools.specaug import SpecAug
                self.specaug_lid = SpecAug(**self.config_lid["SPECAUG"])
                self.specaug_lid.to(self.device)
        self.first_round = True
        print('[ RUNNER ] - Initialized')

    def _get_optimizer(self, trainable_models, mission='lid'):
        optimizer = get_optimizer(
            trainable_models, 
            eval(f'self.config_{mission}')['runner']['total_steps'],
            eval(f'self.config_{mission}')['optimizer']
        )
        # self._load_weight(optimizer, 'Optimizer')
        return optimizer

    def train_LID(self):

        dataset_config = self.config_lid['DATASET']
        splits = dataset_config['train']
        self.train_dataset_lid = LID_Dataset(splits, **dataset_config)
        self.train_dataloader_lid = DataLoader(self.train_dataset_lid, batch_size=1, collate_fn=self.train_dataset_lid.collate_fn, shuffle=True)
        
        pbar = tqdm(total=self.config_lid['runner']['total_steps'], dynamic_ncols=True, desc='LID overall')
        
        self.upstream_lid.eval()
        self.featurizer_lid.train()
        self.downstream_lid.train()
        trainable_models = [self.featurizer_lid, self.downstream_lid]
        trainable_params = list(self.featurizer_lid.parameters()) + list(self.downstream_lid.parameters())

        optimizer = self._get_optimizer(trainable_models, mission='lid')
        # print(optimizer)
        if self.config.get('scheduler'):
            scheduler = self._get_scheduler(optimizer)
        else:
            scheduler = None
        
        epoch = 0
        backward_steps = 0
        gradient_accumulate_steps = self.config_lid['runner']['gradient_accumulate_steps']
        # self.train_dataloader.sampler.set_epoch(epoch)
        avg_acc, avg_loss, total_frames = 0., 0., 0
        logs = {'steps_acc': [], 'steps_frames':[], 'steps_loss': [] }
        while pbar.n < pbar.total:
            for batch_id, (wavs, labels) in enumerate(tqdm(self.train_dataloader_lid, dynamic_ncols=True, total=len(self.train_dataloader_lid), desc=f'training')):
                try:
                    if pbar.n >= pbar.total:
                        break
                    global_step = pbar.n + 1
                    # print(wavs)
                    wavs, labels = [torch.FloatTensor(wav).to(self.device) for wav in wavs], [ torch.LongTensor(label).to(self.device) for label in labels ]
                    # wavs => list(tensor(length))
                    
                    with torch.no_grad():
                        features = self.upstream_lid(wavs)
                        features = features['hidden_states'] # features => tuple(tensor_layer1(N,T,C), ...tensor_layer_last(N,T,C))

                    features = self.featurizer_lid(features) # features => tensor(N,T,C)
                    if self.specaug_lid:
                        features, _ = self.specaug_lid(features)  # features => list(tensor_1(T,C), ...tensor_n(T, C))
                    else:
                        features = list(features)
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
                    
                    acc, loss, frames, pred = self.downstream_lid(features, labels)

                    loss = loss / gradient_accumulate_steps
                    loss.backward()

                    avg_acc += acc
                    avg_loss += loss.item()
                    total_frames += frames

                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print(f'[Runner] - CUDA out of memory at step {global_step}')
                        # if self.first_round:
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
                    trainable_params, self.config_lid['runner']['gradient_clipping'])

                # optimize
                if math.isnan(grad_norm):
                    print(f'[ Runner ] - grad norm is NaN at step {global_step}')
                else:
                    optimizer.step()
                optimizer.zero_grad()

                # adjust learning rate
                if scheduler:
                    scheduler.step()
                
                if global_step % self.config_lid['runner']['save_step'] == 0:
                    def check_ckpt_num(directory):
                        max_keep = self.config_lid['runner']['max_keep']
                        ckpt_pths = glob.glob(f'{directory}/states-*.ckpt')
                        if len(ckpt_pths) >= max_keep:
                            ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
                            for ckpt_pth in ckpt_pths[:len(ckpt_pths) - max_keep + 1]:
                                tqdm.write(f'[ SAVE ] - remove ckpt \'{ckpt_pth}\'')
                                os.remove(ckpt_pth)
                    check_ckpt_num(self.outdir)
                    ckpt = {
                        'Upstream_name': self.config_lid['UPSTREAM']['name'],
                        'Downstream_lid': self.downstream_lid.state_dict(),
                        'Featurizer_lid': self.featurizer_lid.state_dict(),
                        'Optimizer': optimizer.state_dict(),
                        'Step': global_step,
                        'Epoch': epoch,
                        'Config': self.config_lid
                    }
                    ckpt_name = f'states-{global_step}.ckpt'
                    out_path = os.path.join(self.outdir, ckpt_name)
                    torch.save(ckpt, out_path)
                    tqdm.write(f'[ SAVE ] - ckpt \'{ckpt_name}\' saved at \'{self.outdir}\'')

                if global_step % self.config_lid['runner']['log_step'] == 0:
                    log_acc = avg_acc / total_frames
                    log_loss = avg_loss / (self.config_lid['runner']['log_step'])
                    tqdm.write(f'[ TRAIN ] - LOSS: {log_loss:8f}, ACC: {log_acc:8f}, STEP={global_step}')
                    self.writer.add_scalar(f'acc/train', log_acc, global_step)
                    self.writer.add_scalar(f'loss/train', log_loss, global_step)
                    avg_acc = 0.
                    avg_loss = 0.
                    total_frames = 0
                
                if global_step % self.config_lid['hyperparams']['eval_step'] == 0:
                    test_acc, test_loss = self.evaluate_LID()
                    tqdm.write(f'[ TEST ] - LOSS: {test_loss:8f}, ACC: {test_acc:8f}, STEP={global_step}')
                    self.writer.add_scalar(f'acc/test', test_acc, global_step)
                    self.writer.add_scalar(f'loss/test', test_loss, global_step)

                pbar.update(1)
            epoch += 1

    def evaluate_LID(self):
        if not hasattr(self, 'test_dataset_lid'):
            eval_name = self.config_lid['hyperparams']['eval_dataloader']
            self.test_dataset_lid = LID_Dataset(self.config_lid['DATASET'][eval_name], **self.config_lid['DATASET'])
            self.test_dataloader_lid = DataLoader(self.test_dataset_lid, batch_size=1, collate_fn=self.test_dataset_lid.collate_fn, shuffle=False)
        
        self.featurizer_lid.eval()
        self.downstream_lid.eval()
        total_acc, total_loss, total_frames = 0., 0., 0
        for batch, (wavs, labels) in enumerate(tqdm(self.test_dataloader_lid, total=len(self.test_dataloader_lid), desc='evaluating...')):
            wavs, labels = [torch.FloatTensor(wav).to(self.device) for wav in wavs], [ torch.LongTensor(label).to(self.device) for label in labels ]
            # wavs => list(tensor(length))
            
            with torch.no_grad():
                features = self.upstream_lid(wavs)
                features = features['hidden_states'] # features => tuple(tensor_layer1(N,T,C), ...tensor_layer_last(N,T,C))

                features = self.featurizer_lid(features) # features => tensor(N,T,C)
                # if self.specaug:
                features = list(features)
                #     features, _ = self.specaug(features)  # features => list(tensor_1(T,C), ...tensor_n(T, C))
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
                
                acc, loss, frames, pred = self.downstream_lid(features, labels)
                total_acc += acc
                total_loss += loss.item()
                total_frames += frames

        avg_acc = total_acc / total_frames
        avg_loss = total_loss / len(self.test_dataloader_lid)
        self.downstream_lid.train()
        self.featurizer_lid.train()

        return avg_acc, avg_loss





def main():
    with open(config_path, 'r') as yml_f:
        config = yaml.safe_load(yml_f)
    
    
    if config['mission'] == 'LID':
        runner = Runner(config)

if __name__ == '__main__':
    main()