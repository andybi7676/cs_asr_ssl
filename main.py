import torch
import yaml
import numpy as np
# from models/model import Classifier, Downstream, Featurizer
from models.model import Classifier, Downstream, Featurizer

config_path = './configs/xlsr.yml'

def main():
    with open(config_path, 'r') as yml_f:
        config = yaml.safe_load(yml_f)
    # upstream = torch.hub.load('s3prl/s3prl', config['upstream'])
    featurizer = Featurizer(**config)
    downstream = Downstream(**config)
    classifier = Classifier(**config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier.to(device)
    np_ary = [[i for i in range(1024)] for j in range(60)]
    np_ary = np.reshape(np_ary, (5, 12, 1024))
    rdn = torch.FloatTensor(np_ary).to(device)
    with torch.no_grad():
        print(rdn.size())
        rdn = rdn.view(5, 2, 6, 1024)
        print(rdn)
        pred = classifier(rdn)
        print(pred.size())
        # print(pred)

if __name__ == '__main__':
    main()