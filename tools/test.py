import numpy as np
import torch
import glob
from text import load_text_encoder

dict_path = '/home/b07502072/cs_ssl/cs_asr_ssl/dicts/dict_9k.model'
out_path = './dicts/dict_9k_id_to_text.txt'
dictionary = load_text_encoder('subword', dict_path)

print(dictionary.decode([112, 113]))
# with open(out_path, 'w') as outf:
#     for i in range(9000):
#         outf.write(dictionary.decode([i]) + '\n')