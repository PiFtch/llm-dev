import torch
import torch.nn as nn

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    def __init__(self,embed_size, heads):
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

    def forward(self, x, mask=None):   # X shape [batch_size, seq_len, embdded_len]
        batch_size, seq_len, embed_size = x.shape

        queries = self.query_linear(x)
        keys = self.key_linear(x)
        values = self.value_linear(x)

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
    def __init__(self, input_shape, hidden_size, output_size):
        super(FeedForward, self).__init__()
        self.fc1 = torch.nn.Linear(input_shape[0] * input_shape[1] * input_shape[2], hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x:torch.Tensor):
        x = self.fc1(x.flatten())
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
# class Encoder(torch.nn.Module):
#     def __init__(self, input_shape, hidden_size, output_size):
#         super(Encoder, self).__init__()

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
add_norm = AddNorm((batch_size, seq_len, embed_size)).to(device=dev)
ff = FeedForward((batch_size, seq_len, embed_size), embed_size, embed_size).to(device=dev)

# input [batch_size, seq_len, embed_size]
X = torch.randn(batch_size, seq_len, embed_size).to(device=dev)

# Encoder
Y = multi_head_attn(X)
Z = add_norm(X, Y)
W = ff(Z)

print(W)

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