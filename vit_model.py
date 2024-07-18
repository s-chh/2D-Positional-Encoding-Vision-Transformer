import torch
import torch.nn as nn
from positional_embeddings.pos_embed_none import EmbedLayerWithNone
from positional_embeddings.pos_embed_learn import EmbedLayerWithLearn
from positional_embeddings.pos_embed_sinusoidal import EmbedLayerWithSinusoidal
from positional_embeddings.pos_embed_relative import SelfAttentionWithRelative
from positional_embeddings.pos_embed_rope import SelfAttentionWithRope


# B -> Batch Size
# C -> Number of Input Channels
# IH -> Image Height
# IW -> Image Width
# P -> Patch Size
# E -> Embedding Dimension
# N -> Number of Patches = IH/P * IW/P
# S -> Sequence Length   = IH/P * IW/P + 1 or N + 1 (extra 1 is of Classification Token)
# Q -> Query Sequence length (equal to S for self-attention)
# K -> Key Sequence length   (equal to S for self-attention)
# V -> Value Sequence length (equal to S for self-attention)
# H -> Number of heads
# HE -> Head Embedding Dimension = E/H


class OriginalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_attention_heads):
        super().__init__()
        self.embed_dim         = embed_dim
        self.n_attention_heads = n_attention_heads
        self.head_embed_dim    = embed_dim // n_attention_heads

        self.queries           = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)   # Queries projection
        self.keys              = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)   # Keys projection
        self.values            = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)   # Values projection
        self.out_projection    = nn.Linear(self.head_embed_dim * self.n_attention_heads, self.embed_dim)   # Out projection

    def forward(self, x):
        b, s, e = x.shape  # Note: In case of self-attention Q, K and V are all equal to S

        xq = self.queries(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)     # B, Q, E      ->  B, Q, (H*HE)  ->  B, Q, H, HE
        xq = xq.permute(0, 2, 1, 3)                                                         # B, Q, H, HE  ->  B, H, Q, HE
        xk = self.keys(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)        # B, K, E      ->  B, K, (H*HE)  ->  B, K, H, HE
        xk = xk.permute(0, 2, 1, 3)                                                         # B, K, H, HE  ->  B, H, K, HE
        xv = self.values(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)      # B, V, E      ->  B, V, (H*HE)  ->  B, V, H, HE
        xv = xv.permute(0, 2, 1, 3)                                                         # B, V, H, HE  ->  B, H, V, HE


        # Compute Attention presoftmax values
        xk = xk.permute(0, 1, 3, 2)                                                         # B, H, K, HE  ->  B, H, HE, K
        x_attention = torch.matmul(xq, xk)                                                  # B, H, Q, HE  *   B, H, HE, K   ->  B, H, Q, K    (Matmul tutorial eg: A, B, C, D  *  A, B, E, F  ->  A, B, C, F   if D==E)

        x_attention /= float(self.head_embed_dim) ** 0.5                                    # Scale presoftmax values for stability

        x_attention = torch.softmax(x_attention, dim=-1)                                    # Compute Attention Matrix

        x = torch.matmul(x_attention, xv)                                                   # B, H, Q, K  *  B, H, V, HE  ->  B, H, Q, HE     Compute Attention product with Values

        # Format the output
        x = x.permute(0, 2, 1, 3)                                                           # B, H, Q, HE -> B, Q, H, HE
        x = x.reshape(b, s, e)                                                              # B, Q, H, HE -> B, Q, (H*HE)

        x = self.out_projection(x)                                                          # B, Q,(H*HE) -> B, Q, E
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim, n_attention_heads, forward_mul, seq_len, dropout=0.0, pos_embed='learn', max_relative_distance=2):
        super().__init__()
        self.norm1      = nn.LayerNorm(embed_dim)
        self.dropout1   = nn.Dropout(dropout)

        if pos_embed == 'relative':
            self.attention = SelfAttentionWithRelative(embed_dim, n_attention_heads, seq_len=seq_len, max_relative_dist=max_relative_distance)
        elif pos_embed == 'rope':
            self.attention = SelfAttentionWithRope(embed_dim, n_attention_heads, seq_len=seq_len)
        else:
            self.attention = OriginalSelfAttention(embed_dim, n_attention_heads)

        self.norm2      = nn.LayerNorm(embed_dim)
        self.fc1        = nn.Linear(embed_dim, embed_dim * forward_mul)
        self.activation = nn.GELU()
        self.fc2        = nn.Linear(embed_dim * forward_mul, embed_dim)
        self.dropout2   = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.attention(self.norm1(x)))                                # Skip connections
        x = x + self.dropout2(self.fc2(self.activation(self.fc1(self.norm2(x)))))           # Skip connections
        return x


class Classifier(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = x[:, 0, :]              # Get CLS token
        x = self.fc(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, n_channels, embed_dim, n_layers, n_attention_heads, forward_mul, image_size, patch_size, n_classes, dropout=0.1, pos_embed='learn', max_relative_distance=2):
        super().__init__()

        # 
        if pos_embed == 'learn':
            self.embedding = EmbedLayerWithLearn(n_channels, embed_dim, image_size, patch_size, dropout=dropout)
        elif pos_embed == 'sinusoidal':
            self.embedding = EmbedLayerWithSinusoidal(n_channels, embed_dim, image_size, patch_size, dropout=dropout)
        else:
            self.embedding = EmbedLayerWithNone(n_channels, embed_dim, image_size, patch_size, dropout=dropout)

        # 
        self.encoder    = nn.ModuleList([Encoder(embed_dim, n_attention_heads, forward_mul, 
                                                 seq_len=(image_size//patch_size)**2+1, 
                                                 dropout=dropout, 
                                                 pos_embed=pos_embed, 
                                                 max_relative_distance=max_relative_distance) for _ in range(n_layers)])

        self.norm       = nn.LayerNorm(embed_dim)                                       # Final normalization layer after the last block
        self.classifier = Classifier(embed_dim, n_classes)

        self.apply(vit_init_weights)                                                    # Weight initalization

    def forward(self, x):
        x = self.embedding(x)
        for block in self.encoder:
            x = block(x)
        x = self.norm(x)
        x = self.classifier(x)
        return x


def vit_init_weights(m): 
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
