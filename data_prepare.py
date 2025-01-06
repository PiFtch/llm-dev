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

generate_vocab = True
def get_vocab() -> (dict, dict):
    tgt_vocab = {'P':0, 'S':1, 'E':2} # english
    src_vocab = {'P':0} # chn

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
                tgt_vocab[tok] = tgt_idx
                tgt_idx += 1
            for tok in src:
                src_vocab[tok] = src_idx
                src_idx += 1
        # print(tgt_vocab)
        # print(src_vocab)
        print(max_src_len)
        print(max_tgt_len)
    return src_vocab, tgt_vocab


def padding_data():
    a = 1

padding_train_set = True
if padding_train_set:
    
    with open('train_data.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
