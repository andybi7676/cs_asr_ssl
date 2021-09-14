import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# from textgrid import TextGrid
bucket_path = '/home/b07502072/cs_ssl/iven/hubert_asr/len_for_bucket/SEAME'
data_path = ''
out_path = './data/valid_names/SEAME/'
split = 'train'

table_list = []
df = pd.read_csv(os.path.join(bucket_path, f'{split}.csv'))
table_list.append(df)
table_list = pd.concat(table_list)

f_names = table_list['file_path'].tolist()
f_len = table_list['length'].tolist()

valid_names = []
for f in tqdm(f_names, total=len(f_names), desc='loading dataset'):
    name = os.path.join(data_path, f.split('.')[0])
    if os.path.isfile(name.replace('SEAME', 'SEAME_LID')+'_lid.pt') and os.path.isfile(name+'.wav'):
        valid_names.append(name)
if not os.path.exists(out_path): os.makedirs(out_path)
np.save(os.path.join(out_path, f'{split}.npy'), np.array(valid_names))