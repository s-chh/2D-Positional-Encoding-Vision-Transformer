import torch
import torch.nn as nn


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


class EmbedLayerWithNone(nn.Module):
    def __init__(self, n_channels, embed_dim, image_size, patch_size, dropout=0.0):
        super().__init__()
        self.conv1         = nn.Conv2d(n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)                       # Patch Encoding
        self.cls_token     = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)                                    # Classification Token
        self.dropout       = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.conv1.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.trunc_normal_(self.cls_token, mean=0.0, std=0.02)
        

    def forward(self, x):
        B = x.shape[0]
        x = self.conv1(x)                                                         # B, C, IH, IW     -> B, E, IH/P, IW/P                Split image into the patches and embed patches
        x = x.reshape([B, x.shape[1], -1])                                        # B, E, IH/P, IW/P -> B, E, (IH/P*IW/P) -> B, E, N    Flattening the patches
        x = x.permute(0, 2, 1)                                                    # B, E, N          -> B, N, E                         Rearrange to put sequence dimension in the middle
        x = torch.cat((torch.repeat_interleave(self.cls_token, B, 0), x), dim=1)  # B, N, E          -> B, (N+1), E       -> B, S, E    Add classification token at the start of every sequence
        x = self.dropout(x)
        return x
