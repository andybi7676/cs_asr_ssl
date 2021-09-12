import torch
from torch.functional import split
from torch.utils.data import DataLoader
import yaml
import numpy as np
from tqdm import tqdm
# from models/model import Classifier, Downstream, Featurizer
from datasets.LID import LID_Dataset
from models.model import Classifier, Downstream, Featurizer
from torch.utils.tensorboard import SummaryWriter
import os

config_path = './configs/xlsr.yml'

def test(upstream, **config):
    splits = config['train']
    test_dataset = LID_Dataset(splits, bucket_path=config['bucket_path'])
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    upstream.to(device)
    upstream.eval()
    with torch.no_grad():
        # print(len(test_dataloader))
        for batch, (X, Y) in enumerate(tqdm(test_dataloader, total=len(test_dataloader), desc='testing')):
            X, Y = X.to(device), Y.to(device)
            print(X.size())
            print(Y.size())
            features = upstream(X)
            print(features['hidden_states'])
            assert 1 == 2, 'stop here'

def train(Upstream, Featurizer, **config):
    dataset_config = config['dataset']
    splits = dataset_config['train']
    train_dataset = LID_Dataset(splits, **dataset_config)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    Upstream.to(device)
    Upstream.eval()
    Featurizer.to(device)
    specaug = None
    if config.get('specaug'):
        from tools.specaug import SpecAug
        specaug = SpecAug(**config["specaug"])
    # pbar = tqdm(total=self.config['hyperparams']['total_steps'], dynamic_ncols=True, desc='overall')
    epochs = 0

    for batch_id, (X, Y) in enumerate(tqdm(train_dataloader, dynamic_ncols=True, total=len(train_dataloader), desc=f'training')):
        
        wav, label = X.to(device), Y.to(device)

        with torch.no_grad():
            feature = Upstream(wav)
        feature = feature['hidden_states']

        print(feature)
        feature = Featurizer(feature)
        print(feature)
        if specaug:
            feature, _ = specaug(feature)
        print(feature)
        assert 1==2

def main():
    with open(config_path, 'r') as yml_f:
        config = yaml.safe_load(yml_f)
    
    exp_name = '-'.join([config['upstream']['name'], config['mission'], config['id']])
    outdir = f'results/{exp_name}'
    if not os.path.exists(outdir): os.makedirs(outdir)
    writer = SummaryWriter(log_dir=f'{exp_name}')

    upstream = torch.hub.load('s3prl/s3prl', config['upstream']['name'])
    featurizer = Featurizer(**config['featurizer'])
    train(upstream, featurizer, **config)
    
    downstream = Downstream(**config['downstream'])
    # classifier = Classifier(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # classifier.to(device)
    np_ary = [[i for i in range(1024)] for j in range(60)]
    np_ary = np.reshape(np_ary, (5, 12, 1024))
    rdn = torch.FloatTensor(np_ary).to(device)
    with torch.no_grad():
        print(rdn.size())
        rdn = rdn.view(5, 2, 6, 1024)
        print(rdn)
        # pred = classifier(rdn)
        print(pred.size())
        # print(pred)

if __name__ == '__main__':
    main()