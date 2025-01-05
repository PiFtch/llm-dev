import torch
import torch.nn as nn
import numpy as np

dev = torch.device("cuda:0" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu")
print(dev)
NEG_INF = -1e10 

# 定义一个普通函数
def my_function(x):
    return x * 2 + 3

def flash_attn(Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, Q_BLOCKSIZE:int, KV_BLOCKSIZE:int):
    O = torch.zeros(Q.shape)

    QBLOCKS = torch.split(Q, Q_BLOCKSIZE, dim=1)
    KBLOCKS = torch.split(K, KV_BLOCKSIZE, dim=1)
    VBLOCKS = torch.split(V, KV_BLOCKSIZE, dim=1)

    padding_value = NEG_INF
    padding_mask = (Q != padding_value)
    print("padding_mask", padding_mask)

    causal_mask = torch.tril(torch.ones((Q.shape[0], Q.shape[1], K.shape[1])))
    print("causal_mask",causal_mask)
    mask = padding_mask & causal_mask
    print("mask", mask)

    Tr = Q.shape[1] // Q_BLOCKSIZE
    Tc = K.shape[1] // KV_BLOCKSIZE

    for j in range(Tc):
        Kj = KBLOCKS[j]
        Vj = VBLOCKS[j]
        for i in range(Tr):
            Qi = QBLOCKS[i]

            Sij = torch.einsum("...id,...jd->ij", Qi, Kj)

            

            # Sij = torch.masked_fill(Sij, )

    return QBLOCKS, KBLOCKS, VBLOCKS, O

# 封装在一个 nn.Module 中
class MultiHeadSelfAttn(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttn, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.query_linear = torch.nn.Linear(embed_size, embed_size)
        self.key_linear = torch.nn.Linear(embed_size, embed_size)
        self.value_linear = torch.nn.Linear(embed_size, embed_size)

        self.fc_out = torch.nn.Linear(embed_size, embed_size)

    # QKV shape [batch_size, seq_len, embdded_len]
    def forward(self, input_queries, input_keys, input_values, mask=None):

        batch_size, seq_len, embed_size = input_queries.shape

        queries = self.query_linear(input_queries)
        keys = self.key_linear(input_keys)
        values = self.value_linear(input_values)

        queries = queries.view(batch_size, seq_len, self.heads, self.head_dim)
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.view(batch_size, seq_len, self.heads, self.head_dim)
        keys = keys.permute(0, 2, 1, 3)
        values = values.view(batch_size, seq_len, self.heads, self.head_dim)
        values = values.permute(0, 2, 1, 3)

        # einsum计算
        # (batch_size, heads, seq_len, head_dim)
        # out (batch_size, heads, seq_len, seq_len)
        # for b:
        #     for h:
        #         for q:
        #             for k:
        #                 // sum the d-dim
        #                 // dot product
        #                 for d:
        #                     QKt[b,h,q,k] += Q[b,h,q,d] * K[b,h,k,d]
        QKt = torch.einsum("bhqd,bhkd->bhqk", queries, keys)

        # scale
        scaling_factor = self.head_dim ** 0.5
        QKt = QKt / scaling_factor

        # apply mask
        if mask is not None:
            QKt = QKt.masked_fill(mask==0, float(NEG_INF))

        # softmax
        attn = torch.softmax(QKt, dim=3)

        # attn(batch_size, heads, seq_len, seq_len)
        # values(batch_size, heads, seq_len, head_dim)
        # O(batch_size, heads, seq_len, head_dim)
        # 使用嵌套的 for 循环计算注意力输出
        # for b in range(batch_size):
        #     for h in range(heads):
        #         for q in range(seq_len):
        #             for d in range(head_dim):
        #                 sum_value = 0
        #                 for k in range(seq_len):
        #                     sum_value += attention_weights[b, h, q, k] * V[b, h, k, d]
        #                 attention_output[b, h, q, d] = sum_value
        out = torch.einsum("bhqk,bhkd->bhqd", attn, values).reshape(batch_size, seq_len, self.heads * self.head_dim)

        out = self.fc_out(out)

        return out

class AddNorm(torch.nn.Module):
    def __init__(self, size, dropout=0.1):
        super(AddNorm, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x, prev_output):
        return x + self.dropout(self.layer_norm(prev_output))
    
class FeedForward(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForward, self).__init__()
        # self.fc1 = torch.nn.Linear(input_shape[0] * input_shape[1] * input_shape[2], hidden_size)
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x:torch.Tensor):  # input [batch_size, seq_len, embed_size]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class EncoderLayer(torch.nn.Module):
    def __init__(self, embed_size, heads):
        super(EncoderLayer, self).__init__()
        self.multi_head_self_attn = MultiHeadSelfAttn(embed_size=embed_size, heads=heads)
        self.add_norm1 = AddNorm(embed_size)
        self.feed_forward = FeedForward(input_size=embed_size, hidden_size=2048, output_size=embed_size)
        self.add_norm2 = AddNorm(embed_size)

    def forward(self, enc_inputs):  # enc_inputs [batch_size, seq_len, embed_size]
        x = self.multi_head_self_attn(enc_inputs, enc_inputs, enc_inputs)
        x = self.add_norm1(enc_inputs, x)
        y = self.feed_forward(x)
        y = self.add_norm2(x, y)

        return y

class DecoderLayer(torch.nn.Module):
    def __init__(self, embed_size, heads):
        super(DecoderLayer, self).__init__()
        self.multi_head_self_attn_masked = MultiHeadSelfAttn(embed_size=embed_size, heads = heads)
        self.add_norm1 = AddNorm(embed_size)
        self.multi_head_self_attn = MultiHeadSelfAttn(embed_size=embed_size, heads=heads)
        self.add_norm2 = AddNorm(embed_size)
        self.feed_forward = FeedForward(input_size=embed_size, hidden_size=2048, output_size=embed_size)
        self.add_norm3 = AddNorm(embed_size)

    # input shape [batch_size, seq_len, embed_size]
    def forward(self, dec_input:torch.Tensor, mask:torch.Tensor, enc_output:torch.Tensor):
        x = self.multi_head_self_attn_masked(dec_input, dec_input, dec_input, mask=mask)
        add_norm1_output = self.add_norm1(dec_input, x)

        x = self.multi_head_self_attn(add_norm1_output, enc_output, enc_output)
        add_norm2_output = self.add_norm2(add_norm1_output, x)

        x = self.feed_forward(add_norm2_output)
        output = self.add_norm3(add_norm2_output, x)
        return output

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout) 
        pos_table = np.array([[pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
                              if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])                  # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])                  # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table)          # enc_inputs: [seq_len, d_model]

    def forward(self, enc_inputs):                                         # enc_inputs: [batch_size, seq_len, d_model]
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs)

class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, num_layers, embed_size, heads):
        super(Encoder, self).__init__()
        self.src_emb = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.pos_enc = PositionalEncoding(d_model=embed_size)
        self.layers = torch.nn.ModuleList([EncoderLayer(embed_size=embed_size, heads=heads) for _ in range(num_layers)])
    
    def forward(self, inputs):
        # 1. encoding
        enc_inputs = self.src_emb(inputs)
        enc_inputs = self.pos_enc(enc_inputs)

        # 2. encoder_layers
        enc_outputs = enc_inputs
        for layer in self.layers:
            enc_outputs = layer(enc_outputs)
        

# 创建模型实例
# model = MyModel()
# 创建一个示例输入张量
# example_input = torch.tensor([1.0, 2.0, 3.0])
# 导出模型为 ONNX 格式
# torch.onnx.export(model, example_input, "my_model.onnx", input_names=['input'], output_names=['output'])
# print("Model exported to my_model.onnx")

batch_size = 3
seq_len = 192
embed_size = 1024
heads = 4

d = 256

multi_head_attn = MultiHeadSelfAttn(embed_size=embed_size, heads=heads).to(device=dev)
# add_norm = AddNorm((batch_size, seq_len, embed_size)).to(device=dev)
add_norm = AddNorm(size=embed_size).to(device=dev)
# ff = FeedForward((batch_size, seq_len, embed_size), embed_size, embed_size).to(device=dev)
ff = FeedForward(input_size=embed_size, hidden_size=2048, output_size=embed_size).to(device=dev)

# input [batch_size, seq_len, embed_size]
X = torch.randn(batch_size, seq_len, embed_size).to(device=dev)
print("input", X.shape)
# Encoder
Y = multi_head_attn(X, X, X)
print("multi-head self attention", Y.shape)
Z = add_norm(X, Y)
print("add&norm", Z.shape)
W = ff(Z)
print("FFN", W.shape)

encoder = EncoderLayer(embed_size=embed_size, heads=heads).to(device=dev)
# encoder = Encoder(vocab_size=)
RES = encoder(X)
print("RES", RES.shape)

torch.onnx.export(encoder, X, "transformer-encoder.onnx", input_names=['input'], output_names=['output'])

decoder = DecoderLayer(embed_size=embed_size, heads=heads).to(device=dev)
decoder_res = decoder(X, None, RES)
torch.onnx.export(decoder, (X, None, RES), "transformer-decoder.onnx", input_names=['input'], output_names=['output'])

             # Encoder_input    Decoder_input        Decoder_output
sentences = [['我 是 学 生 P' , 'S I am a student'   , 'I am a student E'],         # S: 开始符号
             ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],      # E: 结束符号
             ['我 是 男 生 P' , 'S I am a boy'       , 'I am a boy E']]             # P: 占位符号，如果当前句子不足固定长度用P占位

src_vocab = {'P':0, '我':1, '是':2, '学':3, '生':4, '喜':5, '欢':6,'习':7,'男':8}   # 词源字典  字：索引
src_idx2word = {src_vocab[key]: key for key in src_vocab}   # {idx: word}
src_vocab_size = len(src_vocab)                                                     # 字典字的个数
tgt_vocab = {'P':0, 'S':1, 'E':2, 'I':3, 'am':4, 'a':5, 'student':6, 'like':7, 'learning':8, 'boy':9}   # target dict
idx2word = {tgt_vocab[key]: key for key in tgt_vocab}                               # 把目标字典转换成 索引：字的形式
tgt_vocab_size = len(tgt_vocab)                                                     # 目标字典尺寸
src_len = len(sentences[0][0].split(" "))                                           # Encoder输入的最大长度
tgt_len = len(sentences[0][1].split(" "))                                           # Decoder输入输出最大长度

# 把sentences 转换成字典索引
def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]] 
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]] 
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]] 
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)
enc_inputs, dec_inputs, dec_outputs = make_data(sentences)

print(enc_inputs, dec_inputs, dec_outputs)

#自定义数据集函数
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs
  
    def __len__(self):
        return self.enc_inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

loader = torch.utils.data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True) 


d_model = 512   # 字 Embedding 的维度
d_ff = 2048     # 前向传播隐藏层维度
d_k = d_v = 64  # K(=Q), V的维度 
n_layers = 6    # 有多少个encoder和decoder
n_heads = 8     # Multi-Head Attention设置为8

# print(W)

# print(X)
# print(model(X))
# torch.onnx.export(model, X, "multi-head-self-attn.onnx", input_names=['input'], output_names=['output'])

# trace model
# traced_model = torch.jit.trace(model, X)
# print(traced_model.graph)

# traced_symbolic = torch.fx.symbolic_trace(model)
# print(f"{type(traced_model)=} {type(traced_symbolic)=}")
# print(dir(traced_symbolic.__class__))
# print("=" * 20 )  
# print(model)
# torch.fx.graph_module._print_readable(traced_symbolic, "MultiHeadSelfAttn")
# print("=" * 20 )  
# print(traced_symbolic)