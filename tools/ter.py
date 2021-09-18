import edit_distance
import json
from pypinyin import lazy_pinyin as pinyin
from tqdm import tqdm


def check_english(c):
    return (c[0] != '-') and (ord(c[0].upper()) >= ord('A') and ord(c[0].upper()) <= ord('Z'))


def check_mandarin(c):
    return (c[0] == '-') or (not check_english(c))


def count_man(text):
    return sum([1 if check_mandarin(t) else 0 for t in text])


def count_eng(text):
    return sum([1 if t[0] != '<' and check_english(t) else 0 for t in text])


def sub_count_m2m(ref, hyp, i1, i2, j1, j2):
    ''' ref is Mandarin / hyp is Mandarin '''
    assert (i1 - i2) == (j1 - j2)
    count = 0
    for i in range(i2 - i1):
        if check_mandarin(ref[i1 + i]) and check_mandarin(hyp[j1 + i]):
            count += 1

    return count


def sub_count_e2e(ref, hyp, i1, i2, j1, j2):
    ''' ref is English / hyp is English '''
    assert (i1 - i2) == (j1 - j2)
    count = 0
    for i in range(i2 - i1):
        if check_english(ref[i1 + i]) and check_english(hyp[j1 + i]):
            count += 1

    return count


def sub_count_m2e(ref, hyp, i1, i2, j1, j2):
    ''' ref is Mandarin / hyp is English '''
    assert (i1 - i2) == (j1 - j2)
    count = 0
    for i in range(i2 - i1):
        if check_mandarin(ref[i1 + i]) and check_english(hyp[j1 + i]):
            count += 1

    return count


def sub_count_e2m(ref, hyp, i1, i2, j1, j2):
    ''' ref is English / hyp is Mandarin '''
    assert (i1 - i2) == (j1 - j2)
    count = 0
    for i in range(i2 - i1):
        if check_english(ref[i1 + i]) and check_mandarin(hyp[j1 + i]):
            count += 1

    return count


def del_count_man(ref, i1, i2):
    count = 0
    for i in range(i1, i2):
        if check_mandarin(ref[i]):
            count += 1
    return count


def del_count_eng(ref, i1, i2):
    count = 0
    for i in range(i1, i2):
        if ref[i][0] != '<' and check_english(ref[i]):
            count += 1
    return count


def ins_count_man(hyp, j1, j2):
    count = 0
    for j in range(j1, j2):
        if check_mandarin(hyp[j]):
            count += 1
    return count


def ins_count_eng(hyp, j1, j2):
    count = 0
    for j in range(j1, j2):
        if hyp[j][0] != '<' and check_english(hyp[j]):
            count += 1
    return count


def cal_errors(ref: list, hyp: list):
    if len(ref) == 0:
        return {
            'Dis': len(hyp),
            'Dis_Man': count_man(hyp),
            'Dis_Eng': count_eng(hyp),
            'Sub': 0,
            'Sub_M2M': 0,
            'Sub_E2E': 0,
            'Sub_M2E': 0,
            'Sub_E2M': 0,
            'Del': 0,
            'Del_Man': 0,
            'Del_Eng': 0,
            'Ins': len(hyp),
            'Ins_Man': count_man(hyp),
            'Ins_Eng': count_eng(hyp),
            'Len': 0,
            'Len_Err': len(hyp)
        }

    sm = edit_distance.SequenceMatcher(a=ref, b=hyp)
    opcodes = sm.get_opcodes()

    # Substitution
    S = sum([(max(x[2] - x[1], x[4] - x[3]) if x[0] == 'replace' else 0)
             for x in opcodes])
    S_M2M = sum([(sub_count_m2m(ref, hyp, x[1], x[2], x[3], x[4])
                  if (x[0] == 'replace') else 0) for x in opcodes])
    S_E2E = sum([(sub_count_e2e(ref, hyp, x[1], x[2], x[3], x[4])
                  if (x[0] == 'replace') else 0) for x in opcodes])
    S_M2E = sum([(sub_count_m2e(ref, hyp, x[1], x[2], x[3], x[4])
                  if (x[0] == 'replace') else 0) for x in opcodes])
    S_E2M = sum([(sub_count_e2m(ref, hyp, x[1], x[2], x[3], x[4])
                  if (x[0] == 'replace') else 0) for x in opcodes])

    # Deletion
    D = sum([(max(x[2] - x[1], x[4] - x[3]) if x[0] == 'delete' else 0)
             for x in opcodes])
    D_M = sum([del_count_man(ref, x[1], x[2])
               if x[0] == 'delete' else 0 for x in opcodes])
    D_E = sum([del_count_eng(ref, x[1], x[2])
               if x[0] == 'delete' else 0 for x in opcodes])

    # Insertion
    I = sum([(max(x[2] - x[1], x[4] - x[3]) if x[0] == 'insert' else 0)
             for x in opcodes])
    I_M = sum([ins_count_man(hyp, x[3], x[4])
               if x[0] == 'insert' else 0 for x in opcodes])
    I_E = sum([ins_count_eng(hyp, x[3], x[4])
               if x[0] == 'insert' else 0 for x in opcodes])

    N = len(ref)
    # N = len(hyp)

    result = {
        'Dis': S + D + I,
        'Dis_Man': S_M2M + S_M2E + D_M + I_M,
        'Dis_Eng': S_E2E + S_E2M + D_E + I_E,
        'Sub': S,
        'Sub_M2M': S_M2M,
        'Sub_E2E': S_E2E,
        'Sub_M2E': S_M2E,
        'Sub_E2M': S_E2M,
        'Del': D,
        'Del_Man': D_M,
        'Del_Eng': D_E,
        'Ins': I,
        'Ins_Man': I_M,
        'Ins_Eng': I_E,
        'Len': N,
        'Len_Err': abs(len(ref) - len(hyp))
    }

    return result


def read_trn_norm(path: str):
    with open(path, 'r') as fp:
        data = []
        for line in fp:
            line = line.strip().split(' ')
            line = [l for l in line if l != '']
            data.append(line)
        return data


def remove_chinese(text: list):
    text_new = []
    for w in text:
        if check_mandarin(w):
            continue
        text_new.append(w)
    return text_new


def remove_english(text: list):
    text_new = []
    for w in text:
        if (w[0] == '<') or (check_mandarin(w)):
            text_new.append(w)
    return text_new


def remove_noise(text: list):
    text_new = []
    for w in text:
        if w[0] != '<':
            text_new.append(w)
    return text_new


def to_pinyin(text: list):
    text_new = []
    for w in text:
        if check_mandarin(w):
            text_new.append('-' + pinyin(w)[0])
        else:
            text_new.append(w)
    return text_new


def write_utt_ter(utt_ter: list, path: str):
    with open(path, 'w') as fp:
        for t in utt_ter:
            fp.write('{:.3f}\n'.format(t))


def ComputeTER(ref_path: str, hyp_path: str, mode: str = 'all', trans_pyin: bool = False):
    '''
        Compute TER
    '''

    ref_data = read_trn_norm(ref_path)
    hyp_data = read_trn_norm(hyp_path)

    assert len(ref_data) == len(hyp_data), (len(ref_data), len(hyp_data))

    # ref_data = [remove_noise(d) for d in ref_data]
    # hyp_data = [remove_noise(d) for d in hyp_data]

    if mode == 'eng':
        ref_data = [remove_chinese(d) for d in ref_data]
        hyp_data = [remove_chinese(d) for d in hyp_data]
    elif mode == 'man':
        ref_data = [remove_english(d) for d in ref_data]
        hyp_data = [remove_english(d) for d in hyp_data]

    if trans_pyin:
        ref_data = [to_pinyin(d) for d in ref_data]
        hyp_data = [to_pinyin(d) for d in hyp_data]

    results = {}
    utt_ter = []
    record = []

    for r, h in tqdm(zip(ref_data, hyp_data)):
        res = cal_errors(r, h)
        res_pyin = cal_errors(to_pinyin(r), to_pinyin(h))

        if len(results) == 0:
            results = res
            results['Dis_Man_Pyin'] = res_pyin['Dis_Man']
        else:
            for k in res.keys():
                results[k] += res[k]
            results['Dis_Man_Pyin'] += res_pyin['Dis_Man']

        utt_ter.append(1. if res['Len'] == 0 else res['Dis'] / res['Len'])
        res_rec = {
            'TER': 1. if res['Len'] == 0 else res['Dis'] / res['Len'],
            'Man_CER': 1. if res['Len'] == 0 else res['Dis_Man'] / res['Len'],
            'Man_PER': 1. if res['Len'] == 0 else res_pyin['Dis_Man'] / res['Len'],
            'Eng_WER': 1. if res['Len'] == 0 else res['Dis_Eng'] / res['Len'],
            'Sub': 0. if res['Len'] == 0 else res['Sub'] / res['Len'],
            'Del': 0. if res['Len'] == 0 else res['Del'] / res['Len'],
            'Del_Man': 0. if res['Len'] == 0 else res['Del_Man'] / res['Len'],
            'Del_Eng': 0. if res['Len'] == 0 else res['Del_Eng'] / res['Len'],
            'Ins': 1. if res['Len'] == 0 else res['Ins'] / res['Len'],
            'Len_Err': 1. if res['Len'] == 0 else res['Len_Err'] / res['Len']
        }
        record.append(res_rec)

    results['WER'] = results['Dis'] / results['Len']
    results['WER_M'] = results['Dis_Man'] / results['Len']
    results['WER_MP'] = results['Dis_Man_Pyin'] / results['Len']
    results['WER_E'] = results['Dis_Eng'] / results['Len']
    ignore_list = ['WER', 'WER_M', 'WER_MP',
                   'WER_E', 'Dis', 'Dis_Man', 'Dis_Eng', 'Len']
    for k in results.keys():
        if k not in ignore_list:
            results[k] /= results['Len']

    print('Ref: {}'.format(ref_path))
    print('Hyp: {}'.format(hyp_path))
    print('TER: {:.3f} , Man CER: {:.3f} , Man PER: {:.3f} , Eng WER: {:.3f}'
          .format(results['WER'] * 100., results['WER_M'] * 100., results['WER_MP'] * 100., results['WER_E'] * 100.))
    print('Sub: {:.3f} , Sub M2M: {:.3f} , Sub M2E: {:.3f} , Sub E2E: {:.3f} , Sub E2M: {:.3f}'
          .format(results['Sub'] * 100.,
                  results['Sub_M2M'] * 100., results['Sub_M2E'] * 100.,
                  results['Sub_E2E'] * 100., results['Sub_E2M'] * 100.))
    print('Del: {:.3f} , Del Man: {:.3f} , Del Eng: {:.3f}'
          .format(results['Del'] * 100., results['Del_Man'] * 100., results['Del_Eng'] * 100.))
    print('Ins: {:.3f} , Ins Man: {:.3f} , Ins Eng: {:.3f}'
          .format(results['Ins'] * 100., results['Ins_Man'] * 100., results['Ins_Eng'] * 100.))
    print('Len Diff: {:.3f}'.format(results['Len_Err'] * 100.))

    print()

    write_utt_ter(utt_ter, hyp_path + '.{}.ters'.format(mode))

    with open(hyp_path + '.{}.res.json'.format(mode), 'w') as fp:
        json.dump(record, fp, indent=4)
