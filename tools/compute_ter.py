import os
import pdb
import subprocess
import re
import argparse
from ter import ComputeTER

ADVERBIAL = []


def normalize(txt_pth, ignore_noise=False, trans_adverbial=False, rm_minor_token=True):
    txt_ary = []
    with open(txt_pth) as f:
        for line in f:
            # split_idx = line.index('(')
            text = line
            #key = key.strip().replace('(', '').replace(')','')
            norm_text = ''
            #text= '<noise> 啊 对 就 是 那 个 叫老ang 这 个 japanese sub 啦 就 是 要 才 起 啦'
            #text= 'andist  拿   一   个  b ahla  拿   一   个   一  定   会  one  的   好   这   样'
            for word in text.split():
                for char in word:
                    char = char.replace(' ', '')
                    chinese_char = re.findall(r'[\u4e00-\u9fff]+', char)
                    if len(chinese_char) != 0:
                        # Chinese char
                        norm_text += f' {chinese_char[0]} '
                    else:
                        # English char
                        norm_text += f'{char}'
                norm_text += ' '

            norm_text = norm_text.replace(
                '<noise>', '') if ignore_noise else norm_text
            norm_text = norm_text.replace(
                ' unk>', ' <unk>').replace('<unk ', '<unk> ')
            norm_text = norm_text.replace(
                '-', ' ') if rm_minor_token else norm_text
            norm_text = norm_text.replace('<noise>', ' <noise> ')

            if trans_adverbial:
                raise NotImplementedError('Not implemented yet')
            sentence = ''
            for token in norm_text.split():
                sentence += f'{token} '
            sentence = sentence[:-1]
            sentence = re.sub(' +', ' ', norm_text).strip()
            sentence = sentence.replace('  ', ' ')

            txt_ary.append(sentence)
    return txt_ary


if __name__ == '__main__':
    hyp_pth = './results/wav2vec2_base_960/001/ASR/dev-all/test-hyp.ark'
    ref_pth = './results/wav2vec2_base_960/001/ASR/dev-all/test-ref.ark'
    hyp_out = './results/wav2vec2_base_960/001/ASR/dev-all/test-hyp-norm.ark'
    ref_out = './results/wav2vec2_base_960/001/ASR/dev-all/test-ref-norm.ark'
    # normalized_hyp = normalize(hyp_pth)
    # normalized_ref = normalize(ref_pth)

    # with open(hyp_out, 'w') as fw:
    #     for line in normalized_hyp:
    #         fw.write(line + '\n')
    # with open(ref_out, 'w') as fw:
    #     for line in normalized_ref:
    #         fw.write(line + '\n')
    ComputeTER(ref_out, hyp_out, 'eng', False)
    print('=' * 20)




    # parser = argparse.ArgumentParser(description='Compute TER.')
    # parser.add_argument('--rec', '-r', default='all', type=str,
    #                     help='specified decoding result')
    # parser.add_argument('--mode', '-m', default='all', type=str,
    #                     choices=['all', 'eng', 'man'],
    #                     help='mode')
    # parser.add_argument('--pyin', '-p', action='store_true',
    #                     help='whether to transform to pinyin')
    # args = parser.parse_args()

    # # CTC baseline
    # exp_pth = 'exp/train_pytorch_conformer_wo_decoder_bpe5000'

    # # Mask-CTC baseline
    # # exp_pth = 'exp/train_pytorch_w_pretrain_nar_bpe5000_baseline_all'

    # # M2M
    # # exp_pth = 'exp/train_pytorch_w_pretrain_nar_bpe5000_baseline_dup_all'

    # # P2M
    # # exp_pth = 'exp/train_pytorch_w_pretrain_nar_bpe5000_pyin_wo_tok_all'

    # # P2M + Reg
    # # exp_pth = 'exp/train_pytorch_w_pretrain_nar_bpe5000_pyin_wo_tok_w_connect10e4_embreg_all'

    # # AR baseline
    # # exp_pth = 'exp/train_pytorch_pretrain_at_bpe5000'

    # recog_folders = [os.path.join(exp_pth, n) for n in os.listdir(
    #     exp_pth) if n.startswith('decode')]

    # if args.rec != 'all':
    #     recog_folders = [f for f in recog_folders
    #                      if args.rec in f.split('.') or args.rec in f.split('_')]

    # recog_folders = sorted(recog_folders)

    # for pth in recog_folders:

    #     # if not os.path.isfile(os.path.join(pth, 'ref.wrd.trn')) and not os.path.isfile(os.path.join(pth, 'hyp.wrd.trn')):
    #     #     print(f'Cannot find decoding results under {pth}. Skip it...')
    #     #     continue

    #     ref_w_pth = os.path.join(pth, 'ref.wrd.trn.norm')
    #     hyp_w_pth = os.path.join(pth, 'hyp.wrd.trn.norm')

    #     if not os.path.exists(ref_w_pth) or not os.path.exists(ref_w_pth):
    #         ref = normalize(os.path.join(pth, 'ref.wrd.trn'))
    #         hyp = normalize(os.path.join(pth, 'hyp.wrd.trn'))
    #         assert(len(ref) == len(hyp))
    #         # Write
    #         with open(ref_w_pth, 'w') as ref_w, open(hyp_w_pth, 'w') as hyp_w:
    #             for key in ref.keys():
    #                 ref_w.write(f'{ref[key]}\n')
    #                 hyp_w.write(f'{hyp[key]}\n')

    #     # New eval
        # ComputeTER(ref_w_pth, hyp_w_pth, args.mode, args.pyin)
        # print('=' * 20)
