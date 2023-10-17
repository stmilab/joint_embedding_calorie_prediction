import torch
import torch.nn as nn
import math


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        n_features,
        embed_dim,
        num_heads,
        num_classes,
        dropout=0,
        num_layers=6,
        use_pos_emb=False,
        activation=nn.GELU(),
    ):
        super().__init__()

        self.use_pos_emb = use_pos_emb

        self.conv = nn.Conv1d(n_features, embed_dim, 1, 1)
        self.attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                embed_dim,
                num_heads,
                batch_first=True,
                dropout=dropout,
                activation=activation,
            ),
            num_layers,
        )

        self.register_buffer(
            "position_vec",
            torch.tensor(
                [
                    math.pow(10000.0, 2.0 * (i // 2) / embed_dim)
                    for i in range(embed_dim)
                ],
            ),
        )

        self.linear = nn.Linear(embed_dim, num_classes)
        self.sig = nn.Sigmoid()

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time[:, None] / self.position_vec[None, :, None]
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask[:, None]

    def forward(self, x, lens=0, t=0):
        mask = (x == float("inf"))[:, :, 0]
        x[mask] = 0

        z = self.conv(x.permute(0, 2, 1))

        if self.use_pos_emb:
            tem_enc = self.temporal_enc(t, mask)
            z = z + tem_enc

        z = z.permute(0, 2, 1).float()

        z = self.attn(z, src_key_padding_mask=mask)

        return self.linear(z.mean(1))
