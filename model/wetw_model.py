import torch.nn as nn
import torch 
import torch.nn.functional as F
import math


class PairConvModel(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 num_tokens=16,
                 token_dim=8,
                 num_layers=2,
                 conv_channels=128,
                 kernel_size=3,
                 output_dim=128,
                 use_positional_encoding=True,
                 num_heads=4,
                 softmax_tau=0.7,
                 routing_alpha=10.0):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.use_positional_encoding = use_positional_encoding
        self.num_heads = num_heads
        self.softmax_tau = softmax_tau
        self.routing_alpha = routing_alpha

        # --- Token projections ---
        self.token_proj_i = nn.Linear(token_dim, embed_dim)
        self.token_proj_j = nn.Linear(token_dim, embed_dim)

        # --- Positional encoding (for Ei, Ej, |Δ|, Ei*Ej) ---
        if use_positional_encoding:
            self.positional_encoding = self.create_positional_encoding(num_tokens * 4, embed_dim)

        # --- Convolutional stack ---
        conv_layers = []
        in_channels = embed_dim
        for _ in range(num_layers):
            conv_layers += [
                nn.Conv1d(in_channels, conv_channels, kernel_size, padding=kernel_size // 2),
                nn.ReLU(inplace=True)
            ]
            in_channels = conv_channels
        self.conv_net = nn.Sequential(*conv_layers)

        # --- Feature fusion ---
        self.fuse_proj = nn.Linear(conv_channels + 2 * embed_dim, conv_channels)
        self.fuse_act = nn.ReLU(inplace=True)

        # --- Multi-head generator & routing ---
        self.heads_proj = nn.Linear(conv_channels, num_heads * output_dim)
        self.route_proj = nn.Linear(conv_channels, num_heads)

        # ✅ Projection to align global_diff with output_dim (fix for shape mismatch)
        self.global_proj = nn.Linear(embed_dim, output_dim)

    def create_positional_encoding(self, length, d_model):
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, E_i, E_j, **kwargs):
        """
        Drop-in compatible with your existing code.
        (E_i, E_j, wandb=None, i=None, j=None, mask=None, ...)
        Returns: (B, output_dim)
        """
        B = E_i.size(0)
        T, D = self.num_tokens, self.token_dim

        # --- 1. reshape & project ---
        tokens_i = E_i.view(B, T, D)
        tokens_j = E_j.view(B, T, D)
        tokens_i = self.token_proj_i(tokens_i)
        tokens_j = self.token_proj_j(tokens_j)

        # --- 2. relational features ---
        diff = torch.abs(tokens_i - tokens_j)
        prod = tokens_i * tokens_j
        combined = torch.cat([tokens_i, tokens_j, diff, prod], dim=1)  # (B, 4T, embed_dim)

        # --- 3. positional encoding ---
        if self.use_positional_encoding:
            pe = self.positional_encoding.to(combined.device)
            combined = combined + pe[:, :combined.size(1), :]

        # --- 4. convolution ---
        conv_input = combined.permute(0, 2, 1)
        conv_output = self.conv_net(conv_input)
        pooled = conv_output.mean(dim=2)  # (B, conv_channels)

        # --- 5. global relational summary ---
        global_diff = diff.mean(dim=1)   # (B, embed_dim)
        global_prod = prod.mean(dim=1)   # (B, embed_dim)
        fused = torch.cat([pooled, global_diff, global_prod], dim=1)
        fused = self.fuse_act(self.fuse_proj(fused))  # (B, conv_channels)

        # --- 6. multi-head weight generation ---
        heads = self.heads_proj(fused).view(B, self.num_heads, self.output_dim)
        w_heads = F.softmax(heads / self.softmax_tau, dim=-1)  # (B,K,D)

        # --- 7. routing (softmin over heads) ---
        gdiff = self.global_proj(global_diff)  # ✅ project to output_dim
        proxy = torch.sqrt(((w_heads * gdiff.unsqueeze(1)) ** 2).sum(-1) + 1e-8)  # (B,K)
        scores = F.softmax(-self.routing_alpha * proxy, dim=1)  # (B,K)

        out = (scores.unsqueeze(-1) * w_heads).sum(1)  # (B,D)
        out = out / (out.sum(dim=1, keepdim=True) + 1e-8)  # normalize per sample

        return out