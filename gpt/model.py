import torch
import torch.nn.functional as F
from torch import nn

from gpt.config import GPTConfig


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_embedder = nn.Embedding(config.vocab_size,
                                           config.embed_dim)

        self.position_embedder = nn.Parameter(torch.zeros(1,
                                                          config.max_len,
                                                          config.embed_dim))

        self.dropout = nn.Dropout(config.embed_dropout)

        self.blocks = nn.Sequential(*[
            Block(config) for _ in range(config.num_blocks)
        ])

        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.fully_connected = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        sequence_length = x.size(1)
        if sequence_length > self.config.max_len:
            raise ValueError(f"Input sequence is too long: {sequence_length} > {self.config.max_len}")

        token_embeddings = self.token_embedder(x)
        position_embeddings = self.position_embedder[:, :sequence_length, :]

        x = self.dropout(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.layer_norm(x)
        x = self.fully_connected(x)
        # x.shape == (batch_size, seq_len, vocab_size)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.layer_norm2 = nn.LayerNorm(config.embed_dim)

        self.attention = MultiHeadAttention(config)

        self.feed_forward = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 4),
            nn.GELU(),
            nn.Linear(config.embed_dim * 4, config.embed_dim),
            nn.Dropout(config.feed_forward_dropout)
        )

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.layer_norm(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        embed_dim = config.embed_dim
        if config.embed_dim % config.num_heads != 0:
            raise ValueError("Invalid heads and embedding dimension configuration:\n"
                             "embed_dim must be divisible by num_heads.\n"
                             f"{config.embed_dim} % {config.num_heads} != 0")

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.projection = nn.Linear(embed_dim, embed_dim)

        self.attention_dropout = nn.Dropout(config.attn_dropout)
        self.projection_dropout = nn.Dropout(config.feed_forward_dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(self.config.max_len, self.config.max_len))
            .unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x: torch.Tensor):
        batch_size, sequence_length = x.shape[:2]
        # x.shape == (batch_size, seq_len, embed_dim)
        k = (self.key(x)
             .reshape(batch_size, sequence_length, self.config.num_heads, -1)
             .permute(0, 2, 3, 1))
        q = (self.query(x)
             .reshape(batch_size, sequence_length, self.config.num_heads, -1)
             .transpose(1, 2))
        v = (self.value(x)
             .reshape(batch_size, sequence_length, self.config.num_heads, -1)
             .transpose(1, 2))
        # shape == (batch_size, num_heads, seq_len, head_dim)
        attention = (q @ k) / torch.sqrt(torch.tensor(q.size(-1), dtype=torch.float32))
        # attn.shape == (batch_size, num_heads, seq_len, seq_len)
        mask = self.mask[:, :, :sequence_length, :sequence_length]
        attention.masked_fill_(mask == 0, float('-inf'))
        attention = self.attention_dropout(attention)
        # attn.shape == (batch_size, num_heads, seq_len, seq_len)
        attention = F.softmax(attention, dim=-1)
        output = ((attention @ v)
                  # output.shape == (batch_size, num_heads, seq_len, head_dim)
                  .transpose(1, 2)
                  # output.shape == (batch_size, seq_len, num_heads, head_dim)
                  .reshape(batch_size, sequence_length, -1))
        # output.shape == (batch_size, seq_len, embed_dim)
        output = self.projection_dropout(self.projection(output))
        return output
