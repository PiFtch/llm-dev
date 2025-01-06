import re
import opencc
import random

print("IMPORT")

pattern = r"CC-BY"
converter = opencc.OpenCC('t2s')

create_dataset = False
if create_dataset:
    with open('translate-dataset.txt', 'w', encoding='utf-8') as target_file:
        with open('cmn.txt', 'r', encoding='utf-8') as file:
            for line in file:
                # print(line)
                _line = re.sub(pattern + '.*', '', line)
                _line = _line.rstrip()
                _line = converter.convert(_line)
                # print(_line)
                target_file.write(_line + '\n')

split_dataset = False
if split_dataset:
    train_set = []
    test_set = []
    with open('translate-dataset.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        random.shuffle(lines)
        split_point = int(len(lines) * 0.8)

        train_set = lines[:split_point]
        test_set = lines[split_point:]

        with open('train_data.txt', 'w', encoding='utf-8') as target_file:
            target_file.writelines(train_set)

        with open('test_data.txt', 'w', encoding='utf-8') as target_file:
            target_file.writelines(test_set)

generate_vocab = False
def get_vocab() -> (dict, dict):
    tgt_vocab = {'P':0, 'S':1, 'E':2} # english
    src_vocab = {'P'.encode('utf8'):0} # chn

    max_src_len = 1
    max_tgt_len = 1
    with open('translate-dataset.txt', 'r', encoding='utf-8') as file:
        
        lines = file.readlines()
        for line in lines:
            line = line.strip().split('\t')
            # print(line)
            # tgt = line[0].split(' ,.!?')
            tgt = re.findall(r'\w+|[^\w\s]', line[0], re.UNICODE)
            tgt_idx = len(tgt_vocab)

            # src = re.findall(r'\w+|[^\w\s]', line[1], re.UNICODE)
            src = list(line[1])
            max_src_len = max(max_src_len, len(src))
            max_tgt_len = max(max_tgt_len, len(tgt))
            src_idx = len(src_vocab)
            for tok in tgt:
                if tok not in tgt_vocab:
                    tgt_vocab[tok] = tgt_idx
                    tgt_idx += 1
            for tok in src:
                if tok not in src_vocab:
                    src_vocab[tok] = src_idx
                    src_idx += 1
        # print(tgt_vocab)
        # print(src_vocab)
        print(max_src_len)
        print(max_tgt_len)
    return src_vocab, tgt_vocab


padding_train_set = True
import copy
chn_maxlen = 64
eng_maxlen = 64
# if padding_train_set:
def padding_data():
    padding_enc_input_chn = []
    padding_dec_input_eng = []
    padding_dec_output_eng = []

    with open('train_data.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split('\t')
            src_chn = list(line[1]) # chinese
            tgt_eng = re.findall(r'\w+|[^\w\s]', line[0], re.UNICODE)
            dec_output = copy.deepcopy(tgt_eng)
            
            src_padding_cnt = chn_maxlen - len(src_chn)
            src_chn.extend(['P'.encode('utf8')] * src_padding_cnt)
            tgt_eng.insert(0, 'S')
            tgt_padding_cnt = eng_maxlen - len(tgt_eng)
            tgt_eng.extend(['P'] * tgt_padding_cnt)
            dec_output_padding_cnt = eng_maxlen - 1 - len(dec_output)
            dec_output.extend(['P'] * dec_output_padding_cnt)
            dec_output.extend(['E'])
            # print(src_chn, len(src_chn))
            # print(tgt_eng, len(tgt_eng))
            # print(dec_output, len(dec_output))
            # exit()

            padding_enc_input_chn.append(src_chn)
            padding_dec_input_eng.append(tgt_eng)
            padding_dec_output_eng.append(dec_output)
    
    return padding_enc_input_chn, padding_dec_input_eng, padding_dec_output_eng

# padding_data()
# with open('translate-dataset2.txt', 'w', encoding='utf-8') as target_file:
#     with open('translate-dataset.txt', 'r', encoding='utf-8') as file:
#         for line in file:
#             _line = converter.convert(line)
#             target_file.write(_line)