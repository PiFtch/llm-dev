import torch
import torch.nn as nn
import numpy as np

import dataset, data_prepare
import torch.utils.data as Data

dev = torch.device("cuda:0" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu")
# print(dev)
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
        # mask: [batch_size, 1, 1, seq_len]
        if mask is not None:
            # assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"
            # print("mask shape", mask.shape, "QKt shape", QKt.shape)
            # print("mask", mask)
            QKt = QKt.masked_fill(mask==True, float(NEG_INF))
            # QKt = QKt.masked_fill(mask==True, torch.tensor(-60000, dtype=torch.float16))
            # mask = torch.where(mask, 1.0, 0.0)
            # QKt = QKt.masked_fill(mask==1.0, float(NEG_INF))
            # QKt = torch.where(mask, float('-inf'), QKt)
            # print("masked QKt", QKt)

        # print("after masked")
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

    def forward(self, enc_inputs, mask=None):  # enc_inputs [batch_size, seq_len, embed_size]
        x = self.multi_head_self_attn(enc_inputs, enc_inputs, enc_inputs, mask=mask)
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
        # self.dropout = nn.Dropout(p=dropout) 
        pos_table = np.array([[pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
                              if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])                  # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])                  # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table).to(device=dev)          # enc_inputs: [seq_len, d_model]

    def forward(self, enc_inputs):                                         # enc_inputs: [batch_size, seq_len, d_model]
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        # return self.dropout(enc_inputs)
        return enc_inputs

# def get_attn_pad_mask(seq_q, seq_k):    # seq_q: [batch_size, seq_len], seq_k: [batch_size, seq_len]
#     # print(seq_q.size())
#     # print(seq_q.shape)
#     batch_size, len_q = seq_q.size()
#     batch_size, len_k = seq_k.size()
#     # pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
#     pad_mask = torch.eq(seq_k, 0)

#     return pad_attn_mask.expand(batch_size, len_q, len_k)

# def get_attn_subsequence_mask(seq): # seq: [batch_size, tgt_len]
#     attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
#     subsequence_mask = torch.triu(torch.ones(size=attn_shape), diagonal=1)
#     return subsequence_mask # [batch_size, tgt_len, tgt_len]

# 将padding位置设为True，在之后对QKt进行mask操作时，QKt对应mask上为True的位置的值置为NEG_INF，使之无穷小而难以影响softmax计算
def get_pad_mask(seq:torch.Tensor, vocab_pad_value=0):
    # seq: [batch_size, seq_len]
    # pad_mask: [batch_size, 1, 1, seq_len]
    batch_size, seq_len = seq.shape
    # pad_mask = seq.data.eq(vocab_pad_value)
    pad_mask = torch.eq(seq, float(vocab_pad_value))
    pad_mask = pad_mask.unsqueeze(1).unsqueeze(1)
    # pad_mask = pad_mask.expand(batch_size, 1, seq_len, seq_len)
    # print("pad_mask shape", pad_mask.shape)
    return pad_mask

def get_subseq_mask(seq:torch.Tensor):
    # seq: [batch_size, seq_len]
    # subseq_mask: [batch_size, 1, seq_len, seq_len]
    batch_size, seq_len = seq.shape
    subseq_mask = torch.triu(torch.ones([batch_size, 1, seq_len, seq_len]), diagonal=1).bool()
    # print("subseq_mask shape", subseq_mask.shape)
    return subseq_mask

class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, num_layers, embed_size, heads):
        super(Encoder, self).__init__()
        self.src_emb = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.pos_enc = PositionalEncoding(d_model=embed_size)
        self.layers = torch.nn.ModuleList([EncoderLayer(embed_size=embed_size, heads=heads) for _ in range(num_layers)])
    
    def forward(self, inputs, mask):
        # 1. encoding
        enc_inputs = self.src_emb(inputs)
        enc_inputs = self.pos_enc(enc_inputs)

        # 2. encoder_layers
        enc_outputs = enc_inputs
        for layer in self.layers:
            enc_outputs = layer(enc_outputs, mask=mask)
        
        return enc_outputs

class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, num_layers, embed_size, heads):
        super(Decoder, self).__init__()
        self.tgt_emb = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.pos_enc = PositionalEncoding(embed_size)
        self.layers = torch.nn.ModuleList([DecoderLayer(embed_size=embed_size, heads=heads) for _ in range(num_layers)])
    
    # def forward(self, dec_inputs, enc_inputs, enc_outputs):
    #     dec_outputs = self.tgt_emb(dec_inputs)
    #     dec_outputs = self.pos_enc(dec_outputs)

    #     dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
    #     dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)
    #     dec_self_attn_mask = torch.gt(dec_self_attn_pad_mask + dec_self_attn_subsequence_mask, 0)

    #     dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
    #     dec_self_attns, dec_enc_attns = [], []
    #     for layer in self.layers:
    #         dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
    #         dec_self_attns.append(dec_self_attn)
    #         dec_enc_attns.append(dec_enc_attn)
    #     return dec_outputs, dec_self_attns, dec_enc_attns
    def forward(self, dec_inputs, enc_outputs, mask=None):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_enc(dec_outputs)
        
        for layer in self.layers:
            # dec_input:torch.Tensor, mask:torch.Tensor, enc_output:torch.Tensor
            dec_outputs = layer(dec_input=dec_outputs, mask=mask, enc_output=enc_outputs)

        return dec_outputs


class Transformer(torch.nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, num_layers, embed_size, heads):
        super(Transformer, self).__init__()
        self.Encoder = Encoder(vocab_size=src_vocab_size, num_layers=num_layers, embed_size=embed_size, heads=heads)
        self.Decoder = Decoder(vocab_size=tgt_vocab_size, num_layers=num_layers, embed_size=embed_size, heads=heads)
        self.projection = torch.nn.Linear(embed_size, tgt_vocab_size)

    # enc_inputs: [batch_size, src_len]
    # dec_inputs: [batch_size, tgt_len]
    def forward(self, enc_inputs, dec_inputs):
        # Get encoder padding mask
        enc_pad_mask = get_pad_mask(enc_inputs, vocab_pad_value=0).to(device=dev)

        # Get decoder padding & subseq masks
        dec_pad_mask = get_pad_mask(dec_inputs, vocab_pad_value=0).to(device=dev)
        dec_subseq_mask = get_subseq_mask(dec_inputs).to(device=dev)
        dec_mask = dec_pad_mask | dec_subseq_mask

        # enc_outputs: [batch_size, src_len, embed_size]
        enc_outputs = self.Encoder(enc_inputs, mask=enc_pad_mask)

        # dec_outputs: [batch_size, tgt_len, embed_size]
        dec_outputs = self.Decoder(dec_inputs, enc_outputs, mask=dec_mask)

        dec_logits = self.projection(dec_outputs) # [batch_size, seq_len, vocab_size]

        return dec_logits.view(-1, dec_logits.size(-1)) # [batch_size * seq_len, vocab_size]

# batch_size = 3
# seq_len = 192
# embed_size = 1024
# heads = 4

# d = 256

def train(model:Transformer, loader:Data.DataLoader, epochs=50):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device=dev)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    k = 500
    for epoch in range(epochs):
        step = 0
        losses = []
        for enc_inputs, dec_inputs, dec_outputs in loader:      # enc_inputs : [batch_size, src_len]
                                                                # dec_inputs : [batch_size, tgt_len]
                                                                # dec_outputs: [batch_size, tgt_len]
        
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(dev), dec_inputs.to(dev), dec_outputs.to(dev)
            outputs = model(enc_inputs, dec_inputs)             # outputs: [batch_size * tgt_len, tgt_vocab_size]
            # print(outputs.shape, dec_outputs.shape)
            loss = criterion(outputs, dec_outputs.view(-1))
            # print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            losses.append(loss.item())

            if step % k == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'step', step, 'loss =', '{:.6f}'.format(np.mean(losses)))
                losses = []

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

    return model

def greedy_decode(model: Transformer, enc_input, start_symbol, end_symbol, max_len):
    # 1. Encode
    enc_mask = get_pad_mask(seq=enc_input, vocab_pad_value=0).to(device=dev)
    enc_outputs = model.Encoder(enc_input, mask=enc_mask)

    # 2. Init decoder input
    dec_input = torch.zeros(1, max_len).type_as(enc_input.data).to(device=dev)
    next_symbol = start_symbol

    # 3. greedy decoding
    for i in range(max_len):
        dec_input[0][i] = next_symbol
        dec_pad_mask = get_pad_mask(seq=dec_input, vocab_pad_value=0).to(device=dev)
        dec_subseq_mask = get_subseq_mask(seq=dec_input).to(device=dev)
        dec_mask = dec_pad_mask | dec_subseq_mask
        # print("===========")
        # print("dec_input shape", dec_input.shape, "enc_output shape", enc_outputs.shape)
        # print("===========")
        dec_outputs = model.Decoder(dec_input, enc_outputs, mask=dec_mask)
        projected = model.projection(dec_outputs)
        next_symbol = projected.squeeze(0).max(dim=-1)[1][i].item()

        if next_symbol == end_symbol:
            break
    
    # 4. collect predictions
    return dec_input

def inference(model:Transformer, loader:Data.DataLoader):
    src_vocab, tgt_vocab = data_prepare.get_vocab()
    enc_inputs, _, _ = next(iter(loader))
    # print("enc_inputs ", enc_inputs.shape)
    enc_inputs = enc_inputs.to(dev)

    start_symbol = tgt_vocab['S']
    end_symbol = tgt_vocab['E']
    max_len = 64

    # Get the decoder input using greedy decoding
    # print("greedy input", enc_inputs)
    # print("start idx", start_symbol, "end idx", end_symbol)
    predict_dec_input = greedy_decode(model, enc_inputs, start_symbol, end_symbol, max_len)

    # Run the model to get the final predictions
    predict = model(enc_inputs, predict_dec_input)

    # Extract the most probable predictions
    final_predictions = predict.data.max(-1)[1]

    print("Input:", enc_inputs)
    print([loader.dataset.src_idx2word[int(i)] for i in enc_inputs[0]])
    print("Final Predictions:", final_predictions)
    print([loader.dataset.tgt_idx2word[int(i)] for i in final_predictions])

if __name__ == '__main__':

    # data loaders
    train_dataset = dataset.MyDataSet(filename='train_data.txt')
    train_loader = Data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    test_dataset = dataset.MyDataSet(filename='test_data.txt')
    test_loader = Data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    src_vocab, tgt_vocab = data_prepare.get_vocab()
    model = Transformer(src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab), num_layers=6, embed_size=2048, heads=4)
    model.to(device=dev)

    model = train(model=model, loader=train_loader, epochs=2)
    torch.save(model, 'transformer_fp16.pth')

    exit()

    # model.eval()

    # trace model
    enc_inputs, dec_inputs, dec_outputs = next(iter(test_loader))
    enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device=dev), dec_inputs.to(device=dev), dec_outputs.to(device=dev)
    model(enc_inputs, dec_inputs)
    # traced_model = torch.jit.trace(model, [enc_inputs, dec_inputs])
    # print(traced_model.graph)

    # torch.onnx.export(model, enc_inputs, dec_inputs, "transformer.onnx", input_names=['enc_input', 'dec_input'], output_names=['dec_output'])

    input_dict = {"enc_inputs": enc_inputs, "dec_inputs": dec_inputs}
    torch.onnx.export(model, input_dict, "transformer_fp16.onnx")

    exit()

model = torch.load("transformer.pth", weights_only=False).to(dev)
model = train(model=model, loader=train_loader, epochs=5)

torch.save(model, 'transformer_fp32.pth')

inference(model, test_loader)

exit()

# input [batch_size, seq_len, embed_size]
X = torch.randn(batch_size, seq_len, embed_size).to(device=dev)
print("input", X.shape)
# Encoder
# pad_mask = torch.ones([batch_size, 1, 1, seq_len])
# pad_mask = get_attn_pad_mask()
Y = multi_head_attn(X, X, X, mask=pad_mask)
print("multi-head self attention", Y.shape)
exit()
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


# d_model = 512   # 字 Embedding 的维度
# d_ff = 2048     # 前向传播隐藏层维度
# d_k = d_v = 64  # K(=Q), V的维度 
# n_layers = 6    # 有多少个encoder和decoder
# n_heads = 8     # Multi-Head Attention设置为8

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