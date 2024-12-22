import torch

device = 'cuda'

def softmax(input: torch.Tensor):
    # print(input.shape[dim])
    output = torch.Tensor(input.shape)
    
    for i in range(input.shape[0]):
        row_sum = 0.0
        for j in range(input.shape[1]):
            temp = torch.exp(input[i][j])
            row_sum += temp
            output[i][j] = temp
        for j in range(input.shape[1]):
            output[i][j] /= row_sum
    
    return output

def safe_softmax(input: torch.Tensor):
    output = torch.Tensor(input.shape)

    for i in range(input.shape[0]):
        row_sum = 0.0
        # 1. m(i)
        row_max = 0.0
        for j in range(input.shape[1]):
            row_max = max(row_max, input[i][j])
        # 2. softmax
        for j in range(input.shape[1]):
            temp = torch.exp(input[i][j] - row_max)
            row_sum += temp
            output[i][j] = temp
        for j in range(input.shape[1]):
            output[i][j] /= row_sum
    
    return output

def test():
    N = 16
    d = 8

    Q = torch.rand((N, d))
    K = torch.rand((N, d))
    V = torch.rand((N, d))

    print(Q)
    print(K)
    print(V)

    # standard attention
    std_softmax = torch.softmax(Q @ K.T, dim=1)
    print(std_softmax)

def flash_attention(input_Q, input_K, input_V):
    NEG_INF = -1e10  # -infinity
    EPSILON = 1e-10

    Q_LEN = 6
    K_LEN = 6
    Q_BLOCK_SIZE = 3
    KV_BLOCK_SIZE = 3
    P_DROP = 0.2

    Tr = Q_LEN // Q_BLOCK_SIZE
    Tc = K_LEN // KV_BLOCK_SIZE

    Q = torch.randn(1, 1, Q_LEN, 4, requires_grad=True).to(device='cpu')
    K = torch.randn(1, 1, K_LEN, 4, requires_grad=True).to(device='cpu')
    V = torch.randn(1, 1, K_LEN, 4, requires_grad=True).to(device='cpu')

    O = torch.zeros_like(Q, requires_grad=True)
    l = torch.zeros(Q.shape[:-1])[..., None]
    m = torch.ones(Q.shape[:-1])[..., None] * NEG_INF

    # step 4
    Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
    K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
    V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)

    # step 5
    O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
    l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
    m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

    # step 6
    for j in range(Tc):
        # step 7
        Kj = K_BLOCKS[j]
        Vj = V_BLOCKS[j]
        # step 8
        for i in range(Tr):
            # step 9
            Qi = Q_BLOCKS[i]
            Oi = O_BLOCKS[i]
            li = l_BLOCKS[i]
            mi = m_BLOCKS[i]

            # step 10
            S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi, Kj)

            # step 11
            mask = S_ij.ge(0.5)
            S_ij = torch.masked_fill(S_ij, mask, value=0)
            
            # step 12
            m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
            P_ij = torch.exp(S_ij - m_block_ij)
            l_block_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON
            P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)

            # step 13
            mi_new = torch.maximum(m_block_ij, mi)

            li_new = torch.exp(mi - mi_new) * li + \
                    torch.exp(m_block_ij - mi_new) * l_block_ij

            # step 14
            m = torch.nn.Dropout(p=P_DROP)
            P_ij_Vj = m(P_ij_Vj)

            # Step 15
            O_BLOCKS[i] = (li / li_new) * torch.exp(mi - mi_new) * Oi \
                        + (torch.exp(m_block_ij - mi_new) / li_new) * P_ij_Vj
            print(f'-----------Attention : Q{i}xK{j}---------')
            print(O_BLOCKS[i].shape)
            print(O_BLOCKS[0])
            print(O_BLOCKS[1])
            print('\n')

            # step 16
            l_BLOCKS[i] = li_new
            m_BLOCKS[i] = mi_new

    O = torch.cat(O_BLOCKS, dim=2)
    l = torch.cat(l_BLOCKS, dim=2)
    m = torch.cat(m_BLOCKS, dim=2)

if __name__ == '__main__':
    # print("flag")
    # test()

    M = torch.rand((4, 4))

    print("softmax")
    res = softmax(M)
    print('input=', M)
    print('output=', res)

    res = torch.softmax(M, 1)
    print('torch output=', res)

    print("safe softmax")
    res = safe_softmax(M)
    print('input=', M)
    print('output=', res)

    res = torch._safe_softmax(M, dim=1)
    print('torch output=', res)

    flash_attention()
