import math
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_model, n_heads):
        super().__init__()
        self.d_k = d_k
        self.d_model = d_model
        self.n_heads = n_heads

        self.key = nn.Linear(d_model, d_k * n_heads)
        self.value = nn.Linear(d_model, d_k * n_heads)
        self.query = nn.Linear(d_model, d_k * n_heads)

        self.alpha_weights = nn.Parameter(torch.randn(d_k))
        self.beta_weights = nn.Parameter(torch.randn(d_k))
        self.weight_r = nn.Linear(d_k, d_k)
        self.scale_factor = d_k ** -0.5

        self.n_heads = n_heads
        self.fc = nn.Linear(d_k * n_heads, d_model)

    def forward(self, x_k, x_v, x_q, mask=None): 
        k = self.key(x_k) # shape N x T x h*d_k
        v = self.value(x_v) # shape N x T x h*d_v
        q = self.query(x_q) # shape N x T x h*d_q

        k = rearrange(k, 'n t (h d) -> n h t d', h=self.n_heads)
        v = rearrange(v, 'n t (h d) -> n h t d', h=self.n_heads)
        q = rearrange(q, 'n t (h d) -> n h t d', h=self.n_heads)

        n, h, t, d = q.shape # batch_size, n_heads, sequence_lenght, dim

        # global query vector
        alpha = torch.einsum('n h t d, d -> n h t', q, self.alpha_weights) * self.scale_factor
        if mask is not None:
            mask = rearrange(mask, 'n t -> n () t')
            alpha = alpha.masked_fill(
                mask==0, float('-inf')
            )
        alpha = torch.softmax(alpha, dim=-1)
        alpha = repeat(alpha, 'n h t -> n h t copy', copy=d)
        global_query = torch.einsum('n h t d -> n h d', alpha * q)

        # interaction between global query vector and the key vector
        repeat_global_query = repeat(global_query, 'n h d -> n h copy d', copy=t)
        p = repeat_global_query * k

        # global key vector
        beta = torch.einsum('b h t d, d -> b h t', p, self.beta_weights) * self.scale_factor
        if mask is not None:
            beta = beta.masked_fill(
                mask==0, float('-inf')
            )
        beta = torch.softmax(beta, dim=-1)
        beta = repeat(beta, 'n h t -> n h t copy', copy=d)
        global_key = torch.einsum('n h t d -> n h d', beta * p)

        # interaction between global key vector and the value vector
        repeat_global_key = repeat(global_key, 'n h d -> n h copy d', copy=t)
        u = repeat_global_key * v

        # output fastformer
        u_r = self.weight_r(u)
        A = u_r + q
        assert A.shape == (n, h, t, d)
        A = rearrange(A, 'n h t d -> n t (h d)')
        A = self.fc(A.contiguous())
        return A


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024, dropout_prob=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)

        pos = torch.arange(max_len)
        pos = rearrange(pos, 'n -> n ()')
        exp_term = torch.arange(0, d_model, 2)
        div_term = torch.exp(exp_term * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(torch.einsum('n i, d -> n d', pos, div_term))
        pe[0, :, 1::2] = torch.cos(torch.einsum('n i, d -> n d', pos, div_term))
        self.register_buffer('pe', pe) # able to save and load model correctly

    def forward(self, x):
        # x shape (N, T, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_k, d_model, n_heads, dropout_prob=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_k, d_model, n_heads)
        self.ann = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout_prob)
        )
        self.drop_out = nn.Dropout(dropout_prob)

    def forward(self, x, mask=None):
        x = self.ln1(x + self.mha(x, x, x, mask))
        x = self.ln2(x + self.ann(x))
        x = self.drop_out(x)
        return x
        

class FastformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        d_k,
        d_model,
        n_heads,
        n_layers,
        n_classes,
        dropout_prob):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
        transformer_blocks = [
            TransformerBlock(d_k, d_model, n_heads, dropout_prob) for _ in range(n_layers)]

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 20)
        self.classifier = nn.Linear(20, n_classes)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x, mask)
        x = self.ln(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        x = self.classifier(x)
        return x


# if __name__ == '__main__':
#     model = TransformerEncoder(
#         vocab_size=20_000, 
#         max_len=1024, 
#         d_k=16, 
#         d_model=64, 
#         n_heads=4, 
#         n_layers=2, 
#         n_classes=5, 
#         dropout_prob=0.1)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(device)
#     model.to(device)

#     x = np.random.randint(0, 20_000, size=(8, 512))
#     x_t = torch.tensor(x).to(device)

#     mask = np.ones((8, 512))
#     mask[:, 256:] = 0
#     mask_t = torch.tensor(mask).to(device)

#     y = model(x_t, mask_t)
#     print(y.shape)