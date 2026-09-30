"""
Task 2.2(1) - Three architecturally distinct models, all with embeddings
learned from scratch (nn.Embedding trained jointly with the classifier;
no pretrained word2vec/GloVe/transformer embeddings anywhere).

1) Baseline        : Embedding -> mean-pool -> Linear (bag-of-embeddings
                      logistic regression). No sequence order used at all.
2) Experimental #1  : BiLSTM   -> concat(final fwd, final bwd hidden) -> Linear.
                      Adds sequential / recurrent context and direction.
3) Experimental #2  : TextCNN  -> parallel 1D conv filters (kernel sizes
                      3/4/5) -> global max-pool -> concat -> Linear.
                      Adds local n-gram feature detectors instead of
                      recurrence -- a genuinely different inductive bias
                      from both the baseline and the BiLSTM.
"""
import torch
import torch.nn as nn


class BaselineMeanEmbed(nn.Module):
    """Baseline: embeddings learned from scratch, mean-pooled, linear head."""

    def __init__(self, vocab_size, embed_dim=100, pad_idx=0, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x, lengths=None):
        emb = self.embed(x)                      # (B, T, E)
        mask = (x != 0).unsqueeze(-1).float()     # (B, T, 1)
        summed = (emb * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        mean = summed / counts
        return self.fc(self.dropout(mean)).squeeze(-1)  # (B,) logits


class BiLSTMClassifier(nn.Module):
    """Experimental #1: bidirectional LSTM over the embedded sequence."""

    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, pad_idx=0,
                 num_layers=1, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, lengths):
        emb = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers*2, B, hidden_dim) -> take last layer's fwd/bwd
        h_fwd, h_bwd = h_n[-2], h_n[-1]
        h = torch.cat([h_fwd, h_bwd], dim=-1)
        return self.fc(self.dropout(h)).squeeze(-1)


class TextCNNClassifier(nn.Module):
    """Experimental #2: multi-kernel 1D CNN + global max pooling (Kim, 2014 style)."""

    def __init__(self, vocab_size, embed_dim=100, num_filters=100,
                 kernel_sizes=(3, 4, 5), pad_idx=0, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, k, padding=k // 2) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x, lengths=None):
        emb = self.embed(x).transpose(1, 2)  # (B, E, T) for Conv1d
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(emb))          # (B, F, T')
            p, _ = c.max(dim=-1)                # global max-pool
            pooled.append(p)
        feat = torch.cat(pooled, dim=-1)
        return self.fc(self.dropout(feat)).squeeze(-1)


MODEL_REGISTRY = {
    "baseline": BaselineMeanEmbed,
    "bilstm": BiLSTMClassifier,
    "textcnn": TextCNNClassifier,
}
