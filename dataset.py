import torch
import data_prepare
             # Encoder_input    Decoder_input        Decoder_output
sentences = [['我 是 学 生 P' , 'S I am a student'   , 'I am a student E'],         # S: 开始符号
             ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],      # E: 结束符号
             ['我 是 男 生 P' , 'S I am a boy'       , 'I am a boy E']]             # P: 占位符号，如果当前句子不足固定长度用P占位

# src_vocab = {'P':0, '我':1, '是':2, '学':3, '生':4, '喜':5, '欢':6,'习':7,'男':8}   # 词源字典  字：索引
# src_idx2word = {src_vocab[key]: key for key in src_vocab}   # {idx: word}
# src_vocab_size = len(src_vocab)                                                     # 字典字的个数
# tgt_vocab = {'P':0, 'S':1, 'E':2, 'I':3, 'am':4, 'a':5, 'student':6, 'like':7, 'learning':8, 'boy':9}   # target dict
# idx2word = {tgt_vocab[key]: key for key in tgt_vocab}                               # 把目标字典转换成 索引：字的形式
# tgt_vocab_size = len(tgt_vocab)                                                     # 目标字典尺寸
# src_len = len(sentences[0][0].split(" "))                                           # Encoder输入的最大长度
# tgt_len = len(sentences[0][1].split(" "))                                           # Decoder输入输出最大长度

# print(idx2word)

# 把sentences 转换成字典索引
# def make_data(sentences):
#     enc_inputs, dec_inputs, dec_outputs = [], [], []
#     for i in range(len(sentences)):
#         enc_input = [[src_vocab[n] for n in sentences[i][0].split()]] 
#         dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]] 
#         dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]
#         # print(enc_input)
#         enc_inputs.extend(enc_input)
#         dec_inputs.extend(dec_input)
#         dec_outputs.extend(dec_output)
#     return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)



def make_data(train_set_file):
    src_vocab, tgt_vocab = data_prepare.get_vocab()
    src_idx2word = {src_vocab[key]: key for key in src_vocab}
    tgt_idx2word = {tgt_vocab[key]: key for key in tgt_vocab}

    enc_inputs, dec_inputs, dec_outputs = [], [], []
    with open(train_set_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        for i in range(len(lines)):
            # print(lines[i])
            chn_enc = list(lines[i].strip().split('\t')[1])
            eng_dec_input
            enc_input = [[src_vocab[n] for n in chn_enc]]
            dec_input = [[tgt_vocab[n] for n in ]]

    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]] 
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]] 
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]
        # print(enc_input)
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs), src_idx2word, tgt_idx2word

enc_inputs, dec_inputs, dec_outputs, src_idx2word, tgt_idx2word = make_data("train_data.txt")

print(enc_inputs, dec_inputs, dec_outputs)

#自定义数据集函数
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs
        x, y = data_prepare.get_vocab()
  
    def __len__(self):
        return self.enc_inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]
    
# x, y = data_prepare.get_vocab()
# print(x)