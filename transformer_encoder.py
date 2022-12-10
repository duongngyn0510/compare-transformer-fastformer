import math
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_model, n_heads):
        super().__init__()
        self.d_k = d_k
        self.d_model = d_model

        self.key = nn.Linear(d_model, d_k * n_heads)
        self.value = nn.Linear(d_model, d_k * n_heads)
        self.query = nn.Linear(d_model, d_k * n_heads)

        self.n_heads = n_heads
        self.fc = nn.Linear(d_k * n_heads, d_model)

    def forward(self, x_k, x_v, x_q, mask=None): # W_q shape d_model x h*d_q
        k = self.key(x_k) # shape N x T x h*d_k
        v = self.value(x_v) # shape N x T x h*d_v
        q = self.query(x_q) # shape N x T x h*d_q

        N = k.shape[0]
        T = k.shape[1]

        # tính toán cho từng head attention nên reshape lại thành shape N x T x h x d_q
        # change shape (N, T, h, d_k)  -->  (N, h, T, d_k) 
        k = rearrange(k, 'n t (h d) -> n h t d', h=self.n_heads)
        v = rearrange(v, 'n t (h d) -> n h t d', h=self.n_heads)
        q = rearrange(q, 'n t (h d) -> n h t d', h=self.n_heads)

        # compute attention weights 
        # (N, h, T, d_k) x (N, h, d_k, T) --> (N, h, T, T)
        attention_scores = torch.einsum('n h i d, n h j d -> n h i j', q, v) / math.sqrt(self.d_k)
        if mask is not None:
            mask = rearrange(mask, 'n t -> n () () t')
            attention_scores = attention_scores.masked_fill(
                mask == 0, float('-inf')
            )
        
        attention_weights = F.softmax(attention_scores, dim=-1)

        # compute attention weights value
        # (N, h, T, T) x (N, h, T, d_k) --> (N, h, T, d_k)
        A = torch.einsum('n h i j, n h j d -> n h i d', attention_weights, v)
        A = rearrange(A, 'n h t d -> n t (h d)')
        A = A.contiguous()

        # projection
        A = self.fc(A) # shape (N, T, d_model)
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


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        x = self.pos_emb(x)
        return x


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
        

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        d_k,
        d_model,
        n_heads,
        n_layers,
        n_classes,
        dropout_prob, positional_embedding=False):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = positional_embedding
        if not positional_embedding:
            self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
        else:
            self.pos_embedding = PositionalEmbedding(d_model, max_len)

        transformer_blocks = [
            TransformerBlock(d_k, d_model, n_heads, dropout_prob) for _ in range(n_layers)]

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 20)
        self.classifier = nn.Linear(20, n_classes)

    def forward(self, x, mask=None):
        n = x.shape[1]
        x = self.embedding(x)
        if not self.positional_embedding:
            x = self.pos_encoding(x)
        else:
            pos_emb = self.pos_embedding(torch.arange(n, device=device))
            pos_emb = rearrange(pos_emb, 'n d -> () n d')
            x = x + pos_emb

        for block in self.transformer_blocks:
            x = block(x, mask)
        x = self.ln(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        x = self.classifier(x)
        return x


# if __name__ == '__main__':
# model = TransformerEncoder(20_000, 1024, 16, 64, 4, 2, 5, 0.1)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
# model.to(device)

# x = np.random.randint(0, 20_000, size=(8, 512))
# x_t = torch.tensor(x).to(device)

# mask = np.ones((8, 512))
# mask[:, 256:] = 0
# mask_t = torch.tensor(mask).to(device)

# y = model(x_t, mask_t)
# print(y.shape)