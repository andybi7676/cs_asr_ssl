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

config_path = './configs/w2v2_base/w2v2_base_013.yml'

def parse_l2_norm_data(l2_norm_path):
    norms = []
    with open(l2_norm_path, 'r') as f:
        data = f.readlines()
        for line in data:
            line = line.strip()
            if line != '':
                norms.append(float(line))
    return norms

class Runner():
    def __init__(self, config, args=None):
        self.id = config['id']
        self.mission = config['mission']
        self.task = config['task']
        self.init_ckpt = {}
        self.config = config
        self.config_lid = config.get('LID')
        self.config_asr = config.get('ASR')
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.mission == 'LID':
            self.exp_name = '/'.join([self.config_lid['UPSTREAM']['name'], self.id, self.mission])
            self.outdir = f'./results/{self.exp_name}'
            if not os.path.exists(self.outdir): os.makedirs(self.outdir)
            time_str = strftime("%Y-%m-%d_%H-%M", localtime())
            with open(self.outdir+f'/{time_str}_config_lid.yml', 'w') as yml_f:
                yaml.dump(self.config_lid, yml_f, default_flow_style=False)
            self.writer = SummaryWriter(log_dir=self.outdir)
            self.upstream_lid = torch.hub.load('s3prl/s3prl', self.config_lid['UPSTREAM']['name']).to(self.device)
            self.featurizer_lid = Featurizer(self.upstream_lid, self.device, **self.config_lid['FEATURIZER']).to(self.device)
            self.downstream_lid = Downstream(self.featurizer_lid.upstream_dim, **self.config_lid['DOWNSTREAM']).to(self.device)
            self.records = defaultdict(list)
            self.specaug_lid = None
            if self.config_lid.get('SPECAUG'):
                from tools.specaug import SpecAug
                self.specaug_lid = SpecAug(**self.config_lid["SPECAUG"])
                self.specaug_lid.to(self.device)
            if self.config_lid.get('Loss'):
                self.lid_loss = nn.CrossEntropyLoss(
                    weight=torch.FloatTensor(self.config_lid['Loss']['weights']).to(self.device)
                )
            else:
                self.lid_loss = nn.CrossEntropyLoss()
            self.load_ckpt = False
            if self.config_lid['load_ckpt'] == 'last':
                ckpt_pths = glob.glob(f'{self.outdir}/states-*.ckpt')
                ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
                if len(ckpt_pths) == 0:
                    print(f'No ckpt named as \'states-*.ckpt\' was found in \'{self.outdir}\'')
                else:
                    last_ckpt_pth = ckpt_pths[-1]
                    self.load_ckpt = torch.load(last_ckpt_pth)
            if self.config_lid['load_ckpt'] == 'best':
                best_ckpt_pths = glob.glob(f'{self.outdir}/best*.ckpt')
                assert len(ckpt_pths) == 1
                self.load_ckpt = torch.load(best_ckpt_pths[0])
                
        
        if self.mission == 'ASR':
            self.exp_name = '/'.join([self.config_asr['UPSTREAM']['name'], self.id, self.mission])
            self.outdir = f'./results/{self.exp_name}'
            if not os.path.exists(self.outdir): os.makedirs(self.outdir)
            time_str = strftime("%Y-%m-%d_%H-%M", localtime())
            with open(self.outdir+f'/{time_str}_config_asr.yml', 'w') as yml_f:
                yaml.dump(self.config_asr, yml_f, default_flow_style=False)
            self.writer = SummaryWriter(log_dir=self.outdir)
            self.dictionary = load_text_encoder(self.config_asr['DATASET']['dict_mode'], self.config_asr['DATASET']['dict_path'])
            self.config_asr['DOWNSTREAM']['RNNs']['output_size'] = self.dictionary.vocab_size
            self.upstream_asr = torch.hub.load('s3prl/s3prl', self.config_asr['UPSTREAM']['name']).to(self.device)
            self.featurizer_asr = Featurizer(self.upstream_asr, self.device, **self.config_asr['FEATURIZER']).to(self.device)
            self.downstream_asr = Downstream(self.featurizer_asr.upstream_dim, **self.config_asr['DOWNSTREAM']).to(self.device)
            self.specaug_asr = None
            if self.config_asr.get('SPECAUG'):
                from tools.specaug import SpecAug
                self.specaug_asr = SpecAug(**self.config_asr["SPECAUG"])
                self.specaug_asr.to(self.device)
            self.blank = self.dictionary.pad_idx
            self.asr_loss = nn.CTCLoss(
                blank=self.blank,
                zero_infinity = self.config_asr['DATASET']['zero_infinity']
            )
            self.records = defaultdict(list)
            self.best_score = 100.
            self.decoder = None

            self.load_ckpt = False
            if self.config_asr['load_ckpt'] == 'last':
                ckpt_pths = glob.glob(f'{self.outdir}/states-*.ckpt')
                ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
                if len(ckpt_pths) == 0:
                    print(f'No ckpt named as \'states-*.ckpt\' was found in \'{self.outdir}\'')
                else:
                    last_ckpt_pth = ckpt_pths[-1]
                    self.load_ckpt = torch.load(last_ckpt_pth)
            if self.config_asr['load_ckpt'] == 'best':
                best_ckpt_pths = glob.glob(f'{self.outdir}/*best.ckpt')
                assert len(best_ckpt_pths) == 1
                self.load_ckpt = torch.load(best_ckpt_pths[0])
        
        self.config_all = self.config.get('ALL')
        
        if self.mission == 'ALL':
            self.exp_name = '/'.join([self.config_lid['UPSTREAM']['name'], self.id, self.mission])
            self.outdir = f'./results/{self.exp_name}'
            self.writer = SummaryWriter(log_dir=self.outdir)
            self.dictionary = load_text_encoder(self.config_asr['DATASET']['dict_mode'], self.config_asr['DATASET']['dict_path'])
            self.config_asr['DOWNSTREAM']['RNNs']['output_size'] = self.dictionary.vocab_size
            self.upstream_asr = torch.hub.load('s3prl/s3prl', self.config_asr['UPSTREAM']['name']).to(self.device)
            self.featurizer_asr = Featurizer(self.upstream_asr, self.device, **self.config_asr['FEATURIZER']).to(self.device)
            self.downstream_asr = Downstream(self.featurizer_asr.upstream_dim, **self.config_asr['DOWNSTREAM']).to(self.device)
            self.specaug_asr = None
            self.upstream_lid = torch.hub.load('s3prl/s3prl', self.config_lid['UPSTREAM']['name']).to(self.device)
            self.featurizer_lid = Featurizer(self.upstream_lid, self.device, **self.config_lid['FEATURIZER']).to(self.device)
            self.downstream_lid = Downstream(self.featurizer_lid.upstream_dim, **self.config_lid['DOWNSTREAM']).to(self.device)
            self.specaug_lid = None
            self.load_asr = self.config['ALL']['asr_ckpt']
            self.load_lid = self.config['ALL']['lid_ckpt']
            self.records = defaultdict(list)
            self.best_score = 0.
            self.blank = self.dictionary.pad_idx
            self.asr_loss = nn.CTCLoss(
                blank=self.blank,
                zero_infinity = self.config_asr['DATASET']['zero_infinity']
            )
            self.decoder = None
            if self.config_all['load_ckpt'] == 'last':
                ckpt_pths = glob.glob(f'{self.load_lid}/states-*.ckpt')
                ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
                if len(ckpt_pths) == 0:
                    print(f'No ckpt named as \'states-*.ckpt\' was found in \'{self.load_lid}\'')
                else:
                    last_ckpt_pth = ckpt_pths[-1]
                    self.lid_ckpt = torch.load(last_ckpt_pth)
                ckpt_pths = glob.glob(f'{self.load_asr}/states-*.ckpt')
                ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
                if len(ckpt_pths) == 0:
                    print(f'No ckpt named as \'states-*.ckpt\' was found in \'{self.load_asr}\'')
                else:
                    last_ckpt_pth = ckpt_pths[-1]
                    self.asr_ckpt = torch.load(last_ckpt_pth)
            if self.config_lid['load_ckpt'] == 'best':
                best_ckpt_pths = glob.glob(f'{self.load_lid}/best*.ckpt')
                assert len(ckpt_pths) == 1
                self.lid_ckpt = torch.load(best_ckpt_pths[0])
                best_ckpt_pths = glob.glob(f'{self.load_asr}/best*.ckpt')
                assert len(ckpt_pths) == 1
                self.asr_ckpt = torch.load(best_ckpt_pths[0])

        
        self.first_round = True
        print('[ RUNNER ] - Initialized')

    def _get_optimizer(self, trainable_models, mission='lid'):
        total_steps = eval(f'self.config_{mission}')['runner']['total_steps']
        optimizer_conf = eval(f'self.config_{mission}')['optimizer']
        optimizer = get_optimizer(
            trainable_models, 
            total_steps,
            optimizer_conf
        )
        # self._load_weight(optimizer, 'Optimizer')
        return optimizer
    
    def _get_scheduler(self, optimizer, mission='asr'):
        total_steps = eval(f'self.config_{mission}')['runner']['total_steps']
        scheduler_conf = eval(f'self.config_{mission}')['scheduler']
        scheduler = get_scheduler(
            optimizer,
            total_steps,
            scheduler_conf
        )
        # self._load_weight(scheduler, 'Scheduler')
        return scheduler

    def train_LID(self):

        pbar = tqdm(total=self.config_lid['runner']['total_steps'], dynamic_ncols=True, desc='LID overall')
        
        if self.load_ckpt:
            assert self.config_lid['UPSTREAM']['name'] == self.load_ckpt['Upstream_name']
            self.featurizer_lid.load_state_dict(self.load_ckpt['Featurizer_lid'])
            tqdm.write(f'[ LOAD ] - loaded featurizer')
            self.downstream_lid.load_state_dict(self.load_ckpt['Downstream_lid'])
            tqdm.write(f'[ LOAD ] - loaded downstream')

        trainable_models = [self.featurizer_lid, self.downstream_lid]
        trainable_params = list(self.featurizer_lid.parameters()) + list(self.downstream_lid.parameters())

        optimizer = self._get_optimizer(trainable_models, mission='lid')
        # print(optimizer)
        if self.load_ckpt:
            optimizer.load_state_dict(self.load_ckpt['Optimizer'])
            tqdm.write(f'[ LOAD ] - loaded optimizer')
            pbar.update(self.load_ckpt['Step'])
        if self.config.get('scheduler'):
            scheduler = self._get_scheduler(optimizer)
            if self.load_ckpt:
                scheduler = self.load_state_dict(self.load_ckpt['Scheduler'])
        else:
            scheduler = None


        dataset_config = self.config_lid['DATASET']
        splits = dataset_config['train']
        self.train_dataset_lid = LID_Dataset(splits, **dataset_config)
        self.train_dataloader_lid = DataLoader(self.train_dataset_lid, batch_size=1, collate_fn=self.train_dataset_lid.collate_fn, shuffle=True)
        
        self.upstream_lid.eval()
        self.featurizer_lid.train()
        self.downstream_lid.train()
        epoch = 0
        backward_steps = 0
        gradient_accumulate_steps = self.config_lid['runner']['gradient_accumulate_steps']
        # self.train_dataloader.sampler.set_epoch(epoch)
        avg_acc, avg_loss, total_frames = 0., 0., 0
        logs = {'steps_acc': [], 'steps_frames':[], 'steps_loss': [] }
        self.load_ckpt = False
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
                    
                    # acc, loss, frames, pred = self.downstream_lid(features, labels)
                    logits, padded_labels, _, _ = self.downstream_lid(features, labels)

                    loss = self.lid_loss(
                        logits.transpose(-1, 1), # tensor(N, C, T)
                        padded_labels,
                    )
                    # loss = loss / logits.size()[1]
                    pred = logits.transpose(-1, 1).argmax(dim=1) # tensor(N, T)
                    # assert len(pred.squeeze().tolist()) == len(labels[0].tolist())
                    self.records['pred'] += pred.squeeze().tolist()
                    self.records['labels'] += labels[0].tolist()

                    acc = (pred == padded_labels).type(torch.float).sum().item()
                    
                    loss = loss / gradient_accumulate_steps
                    loss.backward()

                    avg_acc += acc
                    avg_loss += loss.item()
                    total_frames += logits.size()[1]

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
                    self.writer.add_scalar(f'acc/train', log_acc, global_step)
                    self.writer.add_scalar(f'loss/train', log_loss, global_step)
                    f1scores = f1_score(self.records['labels'], self.records['pred'], labels=[0, 1, 2], zero_division=0, average=None)
                    class_names = ['silence', 'chinese', 'english']
                    # if len(f1scores)
                    f1scores_str = f'<sil>: {f1scores[0]:.8f}, <chi>: {f1scores[1]:.8f}, <eng>: {f1scores[2]:.8f}'
                    for i, cls_name in enumerate(class_names):
                        self.writer.add_scalar(f'f1score/train/{cls_name}', f1scores[i], global_step)
                    tqdm.write(f'[ TRAIN ] - LOSS: {log_loss:8f}, ACC: {log_acc:8f}, f1_scores: [ {f1scores_str} ], STEP={global_step}')
                    self.records = defaultdict(list)
                    avg_acc = 0.
                    avg_loss = 0.
                    total_frames = 0
                
                if global_step % self.config_lid['runner']['eval_step'] == 0:
                    test_acc, test_loss, f1scores = self.evaluate_LID()
                    self.writer.add_scalar(f'acc/test', test_acc, global_step)
                    self.writer.add_scalar(f'loss/test', test_loss, global_step)
                    class_names = ['slience', 'chinese', 'english']
                    for i, cls_name in enumerate(class_names):
                        self.writer.add_scalar(f'f1score/test/{cls_name}', f1scores[i], global_step)
                    f1scores_str = f'<sil>: {f1scores[0]:.8f}, <chi>: {f1scores[1]:.8f}, <eng>: {f1scores[2]:.8f}'
                    tqdm.write(f'[ TEST ] - LOSS: {test_loss:8f}, ACC: {test_acc:8f}, f1_scores: [ {f1scores_str} ], STEP={global_step}')
                    f_weights = np.array(F.softmax(self.featurizer_lid.weights, dim=-1).tolist())
                    if self.config_lid['FEATURIZER'].get('layer-norm', False):
                        real_weights = f_weights
                    else:
                        upstream_name = self.config_lid['UPSTREAM']['name']
                        l2_norm_path = f'./data/l2_norm/{upstream_name}.txt'
                        norms = np.array(parse_l2_norm_data(l2_norm_path))
                        real_weights = norms * f_weights
                    fig, ax = plt.subplots()
                    ax.plot(real_weights)
                    downstream_name = self.config_lid['DOWNSTREAM']['model_type']
                    ax.set_title(downstream_name)
                    self.writer.add_figure('Featurizer-weights', fig, global_step=global_step)

                    

                pbar.update(1)
            epoch += 1

    def evaluate_LID(self, split='test', load=False, output=False):
        if load:
            assert self.load_ckpt
            assert self.config_lid['UPSTREAM']['name'] == self.load_ckpt['Upstream_name']
            self.featurizer_lid.load_state_dict(self.load_ckpt['Featurizer_lid'])
            tqdm.write(f'[ LOAD ] - loaded featurizer')
            self.downstream_lid.load_state_dict(self.load_ckpt['Downstream_lid'])
            tqdm.write(f'[ LOAD ] - loaded downstream')

        if not hasattr(self, f'test_dataset_lid'):
            eval_name = self.config_lid['runner']['eval_dataloader']
            self.test_dataset_lid = LID_Dataset(self.config_lid['DATASET'][eval_name], **self.config_lid['DATASET'])
            self.test_dataloader_lid = DataLoader(self.test_dataset_lid, batch_size=1, collate_fn=self.test_dataset_lid.collate_fn, shuffle=False)
        
        self.featurizer_lid.eval()
        self.downstream_lid.eval()
        total_acc, total_loss, total_frames = 0., 0., 0
        records = defaultdict(list)
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
                
                logits, padded_labels, _, _ = self.downstream_lid(features, labels)
                
                loss = self.lid_loss(
                        logits.transpose(-1, 1), # tensor(N, C, T)
                        padded_labels,
                    )
                # loss = loss / logits.size()[1]
                pred = logits.transpose(-1, 1).argmax(dim=1) # tensor(N, T)

                records['pred'] += pred.squeeze().tolist()
                records['labels'] += labels[0].tolist()

                acc = (pred == padded_labels).type(torch.float).sum().item()
                pred = pred.tolist()

                total_acc += acc
                total_loss += loss.item()
                total_frames += logits.size()[1]

        avg_acc = total_acc / total_frames
        avg_loss = total_loss / len(self.test_dataloader_lid)
        f1scores = f1_score(records['labels'], records['pred'], labels=[0, 1, 2], zero_division=0, average=None)

        self.downstream_lid.train()
        self.featurizer_lid.train()
        if output:
            print(avg_acc, avg_loss, f1scores)

        return avg_acc, avg_loss, f1scores

    def _decode(self, log_probs, input_lens):
        """Decoder that take log probabilities as input and outputs decoded seq"""
        pred_tokens_batch = []
        pred_words_batch = []

        for log_prob, in_len in zip(log_probs, input_lens):
            log_prob = log_prob[:in_len].unsqueeze(0)
    
            decoded = None
            if self.decoder is not None and not self.training:
                decoded = self.decoder.decode(log_prob)
                if len(decoded) >= 1:
                    decoded = decoded[0]
                    decoded = None if len(decoded) < 1 else decoded[0]
            
            pred_token_ids = log_prob.argmax(dim=-1)
            pred_token_ids = pred_token_ids[pred_token_ids != self.blank].tolist()

            pred_tokens = self.dictionary.decode(pred_token_ids, True)

            if decoded is not None and "words" in decoded:
                pred_words = decoded["words"]
            else:
                pred_words = pred_tokens.split()
            
            pred_tokens_batch.append(pred_tokens)
            pred_words_batch.append(pred_words)

        return pred_tokens_batch, pred_words_batch

    def _compute_metrics(self, pred_tokens_all, pred_words_all, target_tokens_all, target_words_all):
        """Computes WER and UER given the prediction and true transcriptions"""
        unit_error_sum = 0.0
        word_error_sum = 0.0
        unit_length_sum = 0
        word_length_sum = 0

        for pred_tokens, pred_words, target_tokens, target_words in zip(
            pred_tokens_all, pred_words_all, target_tokens_all, target_words_all):

            unit_error_sum += editdistance.eval(pred_tokens, target_tokens)
            unit_length_sum += len(target_tokens)

            word_error_sum += editdistance.eval(pred_words, target_words)
            word_length_sum += len(target_words)

        uer, wer = 100.0, 100.0
        if unit_length_sum > 0:
            uer = 100.0 * unit_error_sum / unit_length_sum
        if word_length_sum > 0:
            wer = 100.0 * word_error_sum / word_length_sum

        return uer, wer

    def train_ASR(self):

        pbar = tqdm(total=self.config_asr['runner']['total_steps'], dynamic_ncols=True, desc='ASR overall')
        if self.load_ckpt:
            # assert self.config_asr['UPSTREAM']['name'] == self.load_ckpt['Upstream_name']
            self.featurizer_asr.load_state_dict(self.load_ckpt['Featurizer_asr'])
            tqdm.write(f'[ LOAD ] - loaded featurizer')
            self.downstream_asr.load_state_dict(self.load_ckpt['Downstream_asr'], strict=False)
            tqdm.write(f'[ LOAD ] - loaded downstream')
        
        

        trainable_models = [self.featurizer_asr, self.downstream_asr]
        trainable_params = list(self.featurizer_asr.parameters()) + list(self.downstream_asr.parameters())

        optimizer = self._get_optimizer(trainable_models, mission='asr')
        if self.load_ckpt:
            optimizer.load_state_dict(self.load_ckpt['Optimizer'])
            tqdm.write(f'[ LOAD ] - loaded optimizer')
            pbar.update(self.load_ckpt['Step'])
        # print(optimizer)
        if self.config_asr.get('scheduler'):
            scheduler = self._get_scheduler(optimizer, mission='asr')
            if self.load_ckpt:
                scheduler.load_state_dict(self.load_ckpt['Scheduler'])
                tqdm.write(f'[ LOAD ] - loaded scheduler')
        else:
            scheduler = None
        
        dataset_config = self.config_asr['DATASET']
        # splits = dataset_config['train']
        bucket_size = dataset_config['bucket_size']
        self.train_dataset_asr = ASR_Dataset('train', self.dictionary, **dataset_config)
        self.train_dataloader_asr = DataLoader(self.train_dataset_asr, batch_size=1, collate_fn=self.train_dataset_asr.collate_fn, shuffle=True)
        self.upstream_asr.eval()
        self.featurizer_asr.train()
        self.downstream_asr.train()
        epoch = 0
        backward_steps = 0
        gradient_accumulate_steps = self.config_asr['runner']['gradient_accumulate_steps']
        # self.train_dataloader.sampler.set_epoch(epoch)
        avg_acc, avg_loss, total_frames = 0., 0., 0
        logs = {'steps_acc': [], 'steps_frames':[], 'steps_loss': [] }
        self.load_ckpt = False
        while pbar.n < pbar.total:
            for batch_id, (wavs, labels) in enumerate(tqdm(self.train_dataloader_asr, dynamic_ncols=True, total=len(self.train_dataloader_asr), desc=f'training')):
                try:
                    if pbar.n >= pbar.total:
                        break
                    global_step = pbar.n + 1
                    # print(wavs)
                    wavs, labels = [torch.FloatTensor(wav).to(self.device) for wav in wavs], [ torch.LongTensor(label).to(self.device) for label in labels ]
                    # wavs => list(tensor(length))
                    
                    with torch.no_grad():
                        features = self.upstream_asr(wavs)
                        features = features['hidden_states'] # features => tuple(tensor_layer1(N,T,C), ...tensor_layer_last(N,T,C))

                    features = self.featurizer_asr(features) # feaes => tensor(N,T,C)
                    if self.specaug_asr:
                        features, _ = self.specaug_asr(features)  # features => list(tensor_1(T,C), ...tensor_n(T, C))
                    else:
                        features = list(features)
                    assert len(features) == len(labels), 'length of features and labels not consistent'
                    
                    logits, padded_labels, log_probs_len, labels_len = self.downstream_asr(features, labels)
                    
                    log_probs = nn.functional.log_softmax(logits, dim=-1)
                    loss = self.asr_loss(
                        log_probs.transpose(0, 1), # (N, T, C) -> (T, N, C)
                        padded_labels,
                        log_probs_len,
                        labels_len,
                    )

                    target_tokens_batch = []
                    target_words_batch = []
                    for label in labels:
                        label_idx = (label != self.dictionary.pad_idx) & (
                            label != self.dictionary.eos_idx
                        )
                        target_token_ids = label[label_idx].tolist()
                        target_tokens = self.dictionary.decode(target_token_ids)
                        target_words = target_tokens.split()

                        target_tokens_batch.append(target_tokens)
                        target_words_batch.append(target_words)
                    with torch.no_grad():
                        pred_tokens_batch, pred_words_batch = self._decode(log_probs.float().contiguous().cpu(), log_probs_len)
                    
                    self.records['loss'].append(loss.item())
                    self.records['target_tokens'] += target_tokens_batch
                    self.records['target_words'] += target_words_batch
                    self.records['pred_tokens'] += pred_tokens_batch
                    self.records['pred_words'] += pred_words_batch

                    loss = loss / gradient_accumulate_steps
                    loss.backward()

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
                    trainable_params, self.config_asr['runner']['gradient_clipping'])

                # optimize
                if math.isnan(grad_norm):
                    print(f'[ Runner ] - grad norm is NaN at step {global_step}')
                else:
                    optimizer.step()
                optimizer.zero_grad()

                # adjust learning rate
                if scheduler:
                    scheduler.step()
                
                save_names = []
                if global_step % self.config_asr['runner']['save_step'] == 0:
                    def check_ckpt_num(directory):
                        max_keep = self.config_asr['runner']['max_keep']
                        ckpt_pths = glob.glob(f'{directory}/states-*.ckpt')
                        if len(ckpt_pths) >= max_keep:
                            ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
                            for ckpt_pth in ckpt_pths[:len(ckpt_pths) - max_keep + 1]:
                                tqdm.write(f'[ SAVE ] - remove ckpt \'{ckpt_pth}\'')
                                os.remove(ckpt_pth)
                    check_ckpt_num(self.outdir)
                    save_names.append(f'states-{global_step}.ckpt')

                if global_step % self.config_asr['runner']['log_step'] == 0:
                    log_save_names = self.log_records('train', global_step)
                    if len(log_save_names) > 0:
                        save_names += (log_save_names)
                if global_step % self.config_asr['runner']['eval_step'] == 0:
                    self.evaluate_ASR()
                    log_save_names = self.log_records('dev', global_step)
                    if len(log_save_names) > 0:
                        save_names += (log_save_names)
                    # tqdm.write(f'[ TEST ] - LOSS: {test_loss:8f}, ACC: {test_acc:8f}, STEP={global_step}')
                    # self.writer.add_scalar(f'acc/test', test_acc, global_step)
                    # self.writer.add_scalar(f'loss/test', test_loss, global_step)

                for save_name in save_names:
                    ckpt = {
                        'Upstream_name': self.config_asr['UPSTREAM']['name'],
                        'Downstream_asr': self.downstream_asr.state_dict(),
                        'Featurizer_asr': self.featurizer_asr.state_dict(),
                        'Optimizer': optimizer.state_dict(),
                        'Step': global_step,
                        'Epoch': epoch,
                        'Config': self.config_asr
                    }
                    if scheduler:
                        ckpt['Scheduler'] = scheduler.state_dict()
                    ckpt_name = save_name
                    out_path = os.path.join(self.outdir, ckpt_name)
                    torch.save(ckpt, out_path)
                    tqdm.write(f'[ SAVE ] - ckpt \'{ckpt_name}\' saved at \'{self.outdir}\'')

                pbar.update(1)
            epoch += 1
    
    def train_ASR_jointly(self):
        alpha = self.config_asr['alpha']

        pbar = tqdm(total=self.config_asr['runner']['total_steps'], dynamic_ncols=True, desc='ASR overall')
        if self.load_ckpt:
            # assert self.config_asr['UPSTREAM']['name'] == self.load_ckpt['Upstream_name']
            self.featurizer_asr.load_state_dict(self.load_ckpt['Featurizer_asr'])
            tqdm.write(f'[ LOAD ] - loaded featurizer')
            self.downstream_asr.load_state_dict(self.load_ckpt['Downstream_asr'], strict=False)
            tqdm.write(f'[ LOAD ] - loaded downstream')
        
        if not hasattr(self, 'featurizer_lid'):
            # self.featurizer_lid = Featurizer(self.upstream_asr, self.device, **self.config_lid['FEATURIZER']).to(self.device)
            self.downstream_lid = Downstream(self.featurizer_asr.upstream_dim, **self.config_lid['DOWNSTREAM']).to(self.device)
            self.specaug_lid = None
            # if self.config_lid.get('SPECAUG'):
            #     from tools.specaug import SpecAug
            #     self.specaug_lid = SpecAug(**self.config_lid["SPECAUG"])
            #     self.specaug_lid.to(self.device)
            if self.config_lid.get('Loss'):
                self.lid_loss = nn.CrossEntropyLoss(
                    weight=torch.FloatTensor(self.config_lid['Loss']['weights']).to(self.device),
                    ignore_index=0
                )
            else:
                self.lid_loss = nn.CrossEntropyLoss()

        trainable_models = [self.featurizer_asr, self.downstream_asr, self.downstream_lid]
        trainable_params = list(self.featurizer_asr.parameters()) + list(self.downstream_asr.parameters()) + list(self.downstream_lid.parameters())

        optimizer = self._get_optimizer(trainable_models, mission='asr')
        if self.load_ckpt:
            optimizer.load_state_dict(self.load_ckpt['Optimizer'])
            tqdm.write(f'[ LOAD ] - loaded optimizer')
            pbar.update(self.load_ckpt['Step'])
        # print(optimizer)
        if self.config_asr.get('scheduler'):
            scheduler = self._get_scheduler(optimizer, mission='asr')
            if self.load_ckpt:
                scheduler.load_state_dict(self.load_ckpt['Scheduler'])
                tqdm.write(f'[ LOAD ] - loaded scheduler')
        else:
            scheduler = None
        
        dataset_config = self.config_asr['DATASET']
        # splits = dataset_config['train']
        bucket_size = dataset_config['bucket_size']
        self.train_dataset_asr = ALL_Dataset('train', self.dictionary, **dataset_config)
        self.train_dataloader_asr = DataLoader(self.train_dataset_asr, batch_size=1, collate_fn=self.train_dataset_asr.collate_fn, shuffle=True)
        self.upstream_asr.eval()
        self.featurizer_asr.train()
        self.downstream_lid.train()
        self.downstream_asr.train()
        epoch = 0
        backward_steps = 0
        gradient_accumulate_steps = self.config_asr['runner']['gradient_accumulate_steps']
        # self.train_dataloader.sampler.set_epoch(epoch)
        avg_acc, avg_loss, total_frames = 0., 0., 0
        logs = {'steps_acc': [], 'steps_frames':[], 'steps_loss': [] }
        self.load_ckpt = False
        while pbar.n < pbar.total:
            for batch_id, (wavs, labels, lid_labels) in enumerate(tqdm(self.train_dataloader_asr, dynamic_ncols=True, total=len(self.train_dataloader_asr), desc=f'training')):
                try:
                    if pbar.n >= pbar.total:
                        break
                    global_step = pbar.n + 1
                    # print(wavs)
                    wavs, labels, lid_labels = [torch.FloatTensor(wav).to(self.device) for wav in wavs], [ torch.LongTensor(label).to(self.device) for label in labels ], [ torch.LongTensor(lid_label).to(self.device) for lid_label in lid_labels ]
                    # wavs => list(tensor(length))
                    
                    with torch.no_grad():
                        features = self.upstream_asr(wavs)
                        features = features['hidden_states'] # features => tuple(tensor_layer1(N,T,C), ...tensor_layer_last(N,T,C))

                    features = self.featurizer_asr(features) # feats => tensor(N,T,C)
                    if self.specaug_asr:
                        features, _ = self.specaug_asr(features)  # features => list(tensor_1(T,C), ...tensor_n(T, C))
                    else:
                        features = list(features)
                    assert len(features) == len(labels), 'length of features and labels not consistent'
                    
                    logits, padded_labels, log_probs_len, labels_len = self.downstream_asr(features, labels)
                    
                    for idx, lb in enumerate(lid_labels):
                        diff = lb.size()[0] - features[idx].size()[0]
                        if diff == 0: continue
                        if diff > 0:
                            q, r = diff // 2, diff % 2
                            if q > 0 :
                                lid_labels[idx] = lb[q+r: -q]
                            else:
                                lid_labels[idx] = lb[r:]
                        else:
                            lid_labels[idx] = torch.cat((lb, torch.LongTensor(np.zeros(-diff, dtype='int')).to(self.device)), 0)

                    logits_lid, padded_labels_lid, _, _ = self.downstream_lid(features, lid_labels)
                    
                    log_probs = nn.functional.log_softmax(logits, dim=-1)
                    asr_loss = self.asr_loss(
                        log_probs.transpose(0, 1), # (N, T, C) -> (T, N, C)
                        padded_labels,
                        log_probs_len,
                        labels_len,
                    )

                    lid_loss = self.lid_loss(
                        logits_lid.transpose(-1, 1), 
                        padded_labels_lid
                    )

                    loss = alpha * asr_loss + (1-alpha) * lid_loss

                    target_tokens_batch = []
                    target_words_batch = []
                    for label in labels:
                        label_idx = (label != self.dictionary.pad_idx) & (
                            label != self.dictionary.eos_idx
                        )
                        target_token_ids = label[label_idx].tolist()
                        target_tokens = self.dictionary.decode(target_token_ids)
                        target_words = target_tokens.split()

                        target_tokens_batch.append(target_tokens)
                        target_words_batch.append(target_words)
                    with torch.no_grad():
                        pred_tokens_batch, pred_words_batch = self._decode(log_probs.float().contiguous().cpu(), log_probs_len)
                    
                    self.records['loss'].append(loss.item())
                    self.records['asr_loss'].append(asr_loss.item())
                    self.records['lid_loss'].append(lid_loss.item())
                    self.records['target_tokens'] += target_tokens_batch
                    self.records['target_words'] += target_words_batch
                    self.records['pred_tokens'] += pred_tokens_batch
                    self.records['pred_words'] += pred_words_batch

                    loss = loss / gradient_accumulate_steps
                    loss.backward()

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
                    trainable_params, self.config_asr['runner']['gradient_clipping'])

                # optimize
                if math.isnan(grad_norm):
                    print(f'[ Runner ] - grad norm is NaN at step {global_step}')
                else:
                    optimizer.step()
                optimizer.zero_grad()

                # adjust learning rate
                if scheduler:
                    scheduler.step()
                
                save_names = []
                if global_step % self.config_asr['runner']['save_step'] == 0:
                    def check_ckpt_num(directory):
                        max_keep = self.config_asr['runner']['max_keep']
                        ckpt_pths = glob.glob(f'{directory}/states-*.ckpt')
                        if len(ckpt_pths) >= max_keep:
                            ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
                            for ckpt_pth in ckpt_pths[:len(ckpt_pths) - max_keep + 1]:
                                tqdm.write(f'[ SAVE ] - remove ckpt \'{ckpt_pth}\'')
                                os.remove(ckpt_pth)
                    check_ckpt_num(self.outdir)
                    save_names.append(f'states-{global_step}.ckpt')

                if global_step % self.config_asr['runner']['log_step'] == 0:
                    log_save_names = self.log_records('train', global_step, jointly=True)
                    if len(log_save_names) > 0:
                        save_names += (log_save_names)
                if global_step % self.config_asr['runner']['eval_step'] == 0:
                    self.evaluate_ASR()
                    log_save_names = self.log_records('dev', global_step)
                    if len(log_save_names) > 0:
                        save_names += (log_save_names)

                for save_name in save_names:
                    ckpt = {
                        'Upstream_name': self.config_asr['UPSTREAM']['name'],
                        'Downstream_asr': self.downstream_asr.state_dict(),
                        'Featurizer_asr': self.featurizer_asr.state_dict(),
                        'Optimizer': optimizer.state_dict(),
                        'Step': global_step,
                        'Epoch': epoch,
                        'Config': self.config_asr
                    }
                    if scheduler:
                        ckpt['Scheduler'] = scheduler.state_dict()
                    ckpt_name = save_name
                    out_path = os.path.join(self.outdir, ckpt_name)
                    torch.save(ckpt, out_path)
                    tqdm.write(f'[ SAVE ] - ckpt \'{ckpt_name}\' saved at \'{self.outdir}\'')

                pbar.update(1)
            epoch += 1

    def evaluate_ASR(self, load=False, mission='dev'):
        if load:
            assert self.load_ckpt, 'No ckpt to be loaded'
            self.featurizer_asr.load_state_dict(self.load_ckpt['Featurizer_asr'])
            tqdm.write(f'[ LOAD ] - loaded featurizer')
            self.downstream_asr.load_state_dict(self.load_ckpt['Downstream_asr'], strict=False)
            tqdm.write(f'[ LOAD ] - loaded downstream')
        
        if not hasattr(self, f'{mission}_dataset_asr'):
            dataset_config = copy.deepcopy(self.config_asr['DATASET'])
            # splits = dataset_config[split]
            dataset_config['bucket_size'] = 1
            eval(f'self.{mission}_dataset_asr') = ASR_Dataset(mission, self.dictionary, **dataset_config)
            eval(f'self.{mission}_dataloader_asr') = DataLoader(eval(f'self.{mission}_dataset_asr'), batch_size=1, collate_fn=self.test_dataset_asr.collate_fn, shuffle=False)
        
        local_dataloader = eval(f'self.{mission}_dataloader_asr')
        self.upstream_asr.eval()
        self.featurizer_asr.eval()
        self.downstream_asr.eval()
        for batch_id, (wavs, labels) in enumerate(tqdm(local_dataloader, dynamic_ncols=True, total=len(local_dataloader), desc=f'{mission} progress...')):
            wavs, labels = [torch.FloatTensor(wav).to(self.device) for wav in wavs], [ torch.LongTensor(label).to(self.device) for label in labels ]
            # wavs => list(tensor(length))
            try:
                with torch.no_grad():
                    features = self.upstream_asr(wavs)
                    features = features['hidden_states'] # features => tuple(tensor_layer1(N,T,C), ...tensor_layer_last(N,T,C))

                    features = self.featurizer_asr(features) # feaes => tensor(N,T,C)
                    features = list(features)
                    assert len(features) == len(labels), 'length of features and labels not consistent'
                
                    logits, padded_labels, log_probs_len, labels_len = self.downstream_asr(features, labels)
                
                    log_probs = nn.functional.log_softmax(logits, dim=-1)
                    loss = self.asr_loss(
                        log_probs.transpose(0, 1), # (N, T, C) -> (T, N, C)
                        padded_labels,
                        log_probs_len,
                        labels_len,
                    )

                    target_tokens_batch = []
                    target_words_batch = []
                    for label in labels:
                        label_idx = (label != self.dictionary.pad_idx) & (
                            label != self.dictionary.eos_idx
                        )
                        target_token_ids = label[label_idx].tolist()
                        target_tokens = self.dictionary.decode(target_token_ids)
                        target_words = target_tokens.split()

                        target_tokens_batch.append(target_tokens)
                        target_words_batch.append(target_words)
                    
                    pred_tokens_batch, pred_words_batch = self._decode(log_probs.float().contiguous().cpu(), log_probs_len)
                    
                    self.records['loss'].append(loss.item())
                    self.records['target_tokens'] += target_tokens_batch
                    self.records['target_words'] += target_words_batch
                    self.records['pred_tokens'] += pred_tokens_batch
                    self.records['pred_words'] += pred_words_batch

            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f'[Runner] - CUDA out of memory during evaluating')
                    # if self.first_round:
                        # raise
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                    raise
                    # continue
                else:
                    raise
        if mission == 'test':
            self.log_records('test', 1)
        
        self.featurizer_asr.train()
        self.downstream_asr.train()
            # whether to accumulate gradient
    def evaluate_jointly(self, split='test'):
        if self.load_asr:
            self.featurizer_asr.load_state_dict(self.asr_ckpt['Featurizer_asr'])
            tqdm.write(f'[ LOAD ] - loaded featurizer_asr')
            self.downstream_asr.load_state_dict(self.asr_ckpt['Downstream_asr'], strict=False)
            tqdm.write(f'[ LOAD ] - loaded downstream_asr')
        
        if self.load_lid:
            self.featurizer_lid.load_state_dict(self.lid_ckpt['Featurizer_lid'])
            tqdm.write(f'[ LOAD ] - loaded featurizer_lid')
            self.downstream_lid.load_state_dict(self.lid_ckpt['Downstream_lid'])
            tqdm.write(f'[ LOAD ] - loaded downstream_lid')

        if not hasattr(self, f'test_dataset_asr'):
            dataset_config = copy.deepcopy(self.config_asr['DATASET'])
            # splits = dataset_config[split]
            dataset_config['bucket_size'] = 1
            self.test_dataset_asr = ASR_Dataset('test', self.dictionary, **dataset_config)
            self.test_dataloader_asr = DataLoader(self.test_dataset_asr, batch_size=1, collate_fn=self.test_dataset_asr.collate_fn, shuffle=False)
        
        self.upstream_asr.eval()
        self.featurizer_asr.eval()
        self.downstream_asr.eval()
        self.featurizer_lid.eval()
        self.downstream_lid.eval()

        for batch_id, (wavs, labels) in enumerate(tqdm(self.test_dataloader_asr, dynamic_ncols=True, total=len(self.test_dataloader_asr), desc=f'testing')):
            wavs, labels = [torch.FloatTensor(wav).to(self.device) for wav in wavs], [ torch.LongTensor(label).to(self.device) for label in labels ]
            # wavs => list(tensor(length))
            try:
                with torch.no_grad():
                    features = self.upstream_asr(wavs)
                    asr_features = features['hidden_states'] # features => tuple(tensor_layer1(N,T,C), ...tensor_layer_last(N,T,C))
                    lid_features = features['hidden_states']
                    asr_features = self.featurizer_asr(asr_features) # feaes => tensor(N,T,C)
                    lid_features = self.featurizer_lid(lid_features)
                    asr_features = list(asr_features)
                    lid_features = list(lid_features)

                    logits_lid, _, _, _ = self.downstream_lid(lid_features, labels)  # tensor(N, T, 3)
                    logits_asr, padded_labels, log_probs_len, labels_len = self.downstream_asr(asr_features, labels) # tensor(N, T, C)
                    # print(logits_asr.size())
                    # pred_asr = logits_asr.argmax(dim=-1)
                    # pred_lid = logits_lid.argmax(dim=-1)
                    # pred_asr = pred_asr[0].tolist()
                    # print(pred_lid[0].tolist()[0:100])
                    # print(pred_asr[0:100])
                    # logits = []
                    # logits_asr = logits_asr[0]
                    # logits_lid = logits_lid[0]
                    # for i, pred in enumerate(pred_asr):
                    #     logit_l = logits_asr[i].cpu().numpy()
                    #     cls_l = logits_lid[i].tolist()
                    #     if pred < 4 or pred == 5164:
                    #         logits.append(logit_l)
                    #     else:
                    #         logit_l = [-float('inf')]*4 + (cls_l[2] + logit_l[4:5164]).tolist() + (cls_l[1]+logit_l[5164:]).tolist()
                    #         logit_l[5164] = -float('inf')
                    #         logits.append(logit_l)

                    # logits = [logits]
                    # logits = torch.FloatTensor(logits).to(self.device)
                    # print(logits.argmax(dim=-1)[0].tolist()[0:100])
                    # assert 1==2
                    logits = logits_asr
                    
                    log_probs = nn.functional.log_softmax(logits, dim=-1)
                    loss = self.asr_loss(
                        log_probs.transpose(0, 1), # (N, T, C) -> (T, N, C)
                        padded_labels,
                        log_probs_len,
                        labels_len,
                    )

                    target_tokens_batch = []
                    target_words_batch = []
                    for label in labels:
                        label_idx = (label != self.dictionary.pad_idx) & (
                            label != self.dictionary.eos_idx
                        )
                        target_token_ids = label[label_idx].tolist()
                        target_tokens = self.dictionary.decode(target_token_ids)
                        target_words = target_tokens.split()

                        target_tokens_batch.append(target_tokens)
                        target_words_batch.append(target_words)
                    
                    pred_tokens_batch, pred_words_batch = self._decode(log_probs.float().contiguous().cpu(), log_probs_len)
                    
                    self.records['loss'].append(loss.item())
                    self.records['target_tokens'] += target_tokens_batch
                    self.records['target_words'] += target_words_batch
                    self.records['pred_tokens'] += pred_tokens_batch
                    self.records['pred_words'] += pred_words_batch

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

        self.log_records('test', 1)

    def log_records(self, split, global_step, jointly=False, **kwargs):
        """
        Return:
            a list of string
                Each string is a filename we wish to use to save the current model
                according to the evaluation result, like the best.ckpt on the dev set
                You can return nothing or an empty list when no need to save the checkpoint
        """
        loss = torch.FloatTensor(self.records['loss']).mean().item()
        tqdm.write(f'[ {split.upper()} ] - LOSS: {loss}')
        if jointly:
            asr_loss = torch.FloatTensor(self.records['asr_loss']).mean().item()
            tqdm.write(f'[ {split.upper()} ] - ASR_LOSS: {asr_loss}')
            self.writer.add_scalar(f'asr/{split}-asr_loss', asr_loss, global_step=global_step)
            lid_loss = torch.FloatTensor(self.records['lid_loss']).mean().item()
            tqdm.write(f'[ {split.upper()} ] - LID_LOSS: {lid_loss}')
            self.writer.add_scalar(f'asr/{split}-lid_loss', lid_loss, global_step=global_step)

        uer, wer = self._compute_metrics(
            self.records['target_tokens'],
            self.records['target_words'],
            self.records['pred_tokens'],
            self.records['pred_words'],
        )

        self.writer.add_scalar(f'asr/{split}-loss', loss, global_step=global_step)
        self.writer.add_scalar(f'asr/{split}-uer', uer, global_step=global_step)
        self.writer.add_scalar(f'asr/{split}-wer', wer, global_step=global_step)
        if self.featurizer_asr != None and hasattr(self.featurizer_asr, 'weights'):
            fig, ax = plt.subplots()
            f_weights = F.softmax(self.featurizer_asr.weights, dim=-1)
            ax.plot(f_weights.detach().cpu().numpy())
            self.writer.add_figure('Featurizer-weights', fig, global_step=global_step)
        tqdm.write(f'[ {split.upper()} ] - UER: {uer:8f}, WER: {wer:8f}')
        # print(f'[ {split.upper()} ] ')

        save_names = []
        if split == 'dev' and wer < self.best_score:
            self.best_score = wer
            save_names.append(f'{split}-best.ckpt')

        if 'test' in split or 'dev' in split:
            hyp_ark = open(os.path.join(self.outdir, f'{split}-hyp.ark'), 'w')
            ref_ark = open(os.path.join(self.outdir, f'{split}-ref.ark'), 'w')
            for idx, (hyp, ref) in enumerate(zip(self.records['pred_words'], self.records['target_words'])):
                hyp = ' '.join(hyp)
                ref = ' '.join(ref)
                hyp_ark.write(f'{hyp}\n')
                ref_ark.write(f'{ref}\n')
            hyp_ark.close()
            ref_ark.close()
        
        self.records = defaultdict(list)
        return save_names

    
    def draw_featurizer(self, mission, loaded=False):
        
        if mission == 'lid':
            if not loaded:
                assert self.config_lid['UPSTREAM']['name'] == self.load_ckpt['Upstream_name']
                self.featurizer_lid.load_state_dict(self.load_ckpt['Featurizer_lid'])
            upstream_name = self.config_lid['UPSTREAM']['name']
            l2_norm_path = f'./data/l2_norm/{upstream_name}.txt'
            norm_weights = np.array(F.softmax(self.featurizer_lid.weights, dim=-1).tolist())
            norms = np.array(parse_l2_norm_data(l2_norm_path))
            real_weights = norms * norm_weights
            print(real_weights)
            fig, ax = plt.subplots()
            ax.plot(real_weights)
            downstream_name = self.config_lid['DOWNSTREAM']['model_type']
            ax.set_title(downstream_name)
            fig.savefig(f'{self.outdir}/featurizer_weights.png')
            

def main():
    torchaudio.set_audio_backend('sox_io')
    with open(config_path, 'r') as yml_f:
        config = yaml.safe_load(yml_f)
    
    
    runner = Runner(config)
    if config['mission'] == 'LID' and config['task'] == 'train':
        runner.train_LID()
    if config['mission'] == 'ASR' and config['task'] == 'train':
        if not config['ASR'].get('jointly-train', False):
            runner.train_ASR()
        else:
            runner.train_ASR_jointly()
    if config['mission'] == 'ALL':
        runner.evaluate_jointly()
    if config['mission'] == 'LID' and config['task'] == 'draw_featurizer':
        runner.draw_featurizer('lid')
    if config['mission'] == 'ASR' and config['task'] == 'evaluate':
        runner.evaluate_ASR(load=True, mission='test')
    if config['mission'] == 'LID' and config['task'] == 'evaluate':
        runner.evaluate_LID(load=True, output=True)

if __name__ == '__main__':
    main()