import torch
from torch.functional import split
from torch.utils.data import dataloader
import yaml
import numpy as np
# from models/model import Classifier, Downstream, Featurizer
from datasets.LID import LID_Dataset
from models.model import Classifier, Downstream, Featurizer

config_path = './configs/xlsr.yml'

def test(upstream, **config):
    test_dataset = LID_Dataset('train', bucket_path=config['bucket_path'])
    test_dataloader = dataloader(LID_Dataset, batch_size=2, shuffle=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for batch, (X, Y) in enumerate(test_dataloader, total=len(test_dataloader), desc='testing'):
            X, Y = X.to(device), Y.to(device)
            print(X.size())
            print(Y.size())
            features = upstream(X)
            print(features)
            assert 1 == 2, 'stop here'

def main():
    with open(config_path, 'r') as yml_f:
        config = yaml.safe_load(yml_f)
    
    upstream = torch.hub.load('s3prl/s3prl', config['upstream']['name'])
    test(upstream , **config['dataset'])
    featurizer = Featurizer(**config['featurizer'])
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