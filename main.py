import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np
from tqdm import tqdm
from datasets.LID import LID_Dataset
from models.model import Downstream, Featurizer
from torch.utils.tensorboard import SummaryWriter
import os

config_path = './configs/w2v2_base.yml'

# def train(upstream, featurizer, **config):
#     dataset_config = config['DATASET']
#     splits = dataset_config['train']
#     train_dataset = LID_Dataset(splits, **dataset_config)
#     train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     upstream.to(device)
#     upstream.eval()
#     Featurizer.to(device)
#     specaug = None
#     if config.get('specaug'):
#         from tools.specaug import SpecAug
#         specaug = SpecAug(**config["SPECAUG"])
#     specaug.to(device)
#     # pbar = tqdm(total=self.config['hyperparams']['total_steps'], dynamic_ncols=True, desc='overall')
#     epochs = 0

#     for batch_id, (X, Y) in enumerate(tqdm(train_dataloader, dynamic_ncols=True, total=len(train_dataloader), desc=f'training')):
        
#         wav, label = X.to(device), Y.to(device)
#         print(wav.size())
#         print(label.size())

#         with torch.no_grad():
#             feature = upstream(wav)
#         feature = feature['hidden_states']

#         # print(feature)
#         feature = featurizer(feature)
#         print(feature)
#         if specaug:
#             feature, _ = specaug(feature)
#         print(feature)



#         assert 1==2

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
        # self.downstream = Downstream(self.device, **self.config['DOWNSTREAM']).to(self.device)
    
    def train_LID(self):

        dataset_config = self.config['DATASET']
        splits = dataset_config['train']
        self.train_dataset = LID_Dataset(splits, **dataset_config)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=1, collate_fn=self.train_dataset.collate_fn, shuffle=True)
        self.specaug = None
        if self.config.get('SPECAUG'):
            from tools.specaug import SpecAug
            self.specaug = SpecAug(**self.config["SPECAUG"])
            self.specaug.to(self.device)
        # pbar = tqdm(total=self.config['hyperparams']['total_steps'], dynamic_ncols=True, desc='overall')
        epochs = 0
        self.upstream.eval()
        for batch_id, (wavs, labels) in enumerate(tqdm(self.train_dataloader, dynamic_ncols=True, total=len(self.train_dataloader), desc=f'training')):
            print(wavs)
            wavs, labels = [torch.FloatTensor(wav).to(self.device) for wav in wavs], labels
            print(wavs[0].size())
            # print(labels[0].size())

            with torch.no_grad():
                features = self.upstream(wavs)
                features = features['hidden_states'] # features = tuple(tensor_layer1(N,T,C), ...tensor_layer_last(N,T,C))

            features = self.featurizer(features) # features = tensor(N,T,C)
            if self.specaug:
                features, _ = self.specaug(features)  # features = list(tensor_1(T,C), ...tensor_n(T, C))
            

            assert 1==2

def main():
    with open(config_path, 'r') as yml_f:
        config = yaml.safe_load(yml_f)
    
    runner = Runner(config)
    
    if config['mission'] == 'LID':
        runner.train_LID()

if __name__ == '__main__':
    main()