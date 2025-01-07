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

def get_attn_pad_mask(seq_q, seq_k):    # seq_q: [batch_size, seq_len], seq_k: [batch_size, seq_len]
    # print(seq_q.size())
    # print(seq_q.shape)
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)

    return pad_attn_mask.expand(batch_size, len_q, len_k)

def get_attn_subsequence_mask(seq): # seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = torch.triu(torch.ones(size=attn_shape), diagonal=1)
    return subsequence_mask # [batch_size, tgt_len, tgt_len]

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

class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, num_layers, embed_size, heads):
        super(Decoder, self).__init__()
        self.tgt_emb = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.pos_enc = PositionalEncoding(embed_size)
        self.layers = torch.nn.ModuleList([DecoderLayer(embed_size=embed_size, heads=heads) for _ in range(num_layers)])
    
    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_enc(dec_outputs)

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)
        dec_self_attn_mask = torch.gt(dec_self_attn_pad_mask + dec_self_attn_subsequence_mask, 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(torch.nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.Encoder = Encoder()
        self.Decoder = DecoderLayer()
        # self.projection = torch.nn.Linear()

    def forward(self, enc_inputs, dec_inputs):
        a = 1

import dataset, data_prepare
import torch.utils.data as Data
loader = Data.DataLoader(dataset.MyDataSet(), batch_size=2, shuffle=True)


def train():
    src_vocab, tgt_vocab = data_prepare.get_vocab()
    
    decoder = Decoder(vocab_size=len(tgt_vocab), num_layers=2, embed_size=1024, heads=4)
    for epoch in range(5):
        for enc_inputs, dec_inputs, dec_outputs in loader:
                        
            decoder(dec_inputs, enc_inputs, enc_inputs)
            print(dec_inputs)
            print(enc_inputs)
            exit()
train()

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

import dataset
loader = torch.utils.data.DataLoader(dataset.MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True) 


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