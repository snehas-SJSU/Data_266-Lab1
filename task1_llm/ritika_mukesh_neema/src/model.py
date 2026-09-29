"""
Task 1.2 - GPT model implemented from scratch.

No `torch.nn.MultiheadAttention`, `torch.nn.Transformer*`, or any prebuilt
attention/transformer module is used anywhere below. The only nn.* building
blocks used are `nn.Linear`, `nn.Embedding`, `nn.Dropout` and `nn.GELU`
(plain layers, not attention/transformer modules) — LayerNorm itself is also
implemented by hand to make the "from scratch" requirement unambiguous.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormScratch(nn.Module):
    """Manual LayerNorm: normalize over the last dim, then learnable affine."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention built from raw linear projections + matmuls.
    Q, K, V are produced by a single fused Linear, split into heads by hand,
    scaled dot-product attention is computed explicitly, and a causal mask
    (lower-triangular) is applied via masked_fill before the softmax.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # causal mask, cached as a buffer (not a parameter)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)

        # split into heads: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # scaled dot-product attention, computed explicitly (no F.scaled_dot_product_attention)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # (B, nh, T, T)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, nh, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # merge heads
        y = self.resid_dropout(self.out_proj(y))
        return y


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, mult * n_embd),
            nn.GELU(),
            nn.Linear(mult * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """LN -> Self-Attention -> residual -> LN -> Feed-Forward -> residual (pre-norm GPT-2 style)."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = LayerNormScratch(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = LayerNormScratch(n_embd)
        self.ff = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTScratch(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_layer: int = 4,
                 n_head: int = 4, n_embd: int = 128, dropout: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)  # learnable positional embedding
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = LayerNormScratch(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"sequence length {T} > block_size {self.block_size}"
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)

        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, greedy: bool = False, top_k: int = None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            if greedy:
                next_id = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
