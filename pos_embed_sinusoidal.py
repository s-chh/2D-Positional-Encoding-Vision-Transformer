import torch
import torch.nn as nn
from utils import get_x_positions, get_y_positions

# B -> Batch Size
# C -> Number of Input Channels
# IH -> Image Height
# IW -> Image Width
# P -> Patch Size
# E -> Embedding Dimension
# N_ -> Number of Patches in 1 dimension = IH/P = IW/P
# N -> Number of Patches = IH/P * IW/P
# S -> Sequence Length   = IH/P * IW/P + 1 or N + 1 (extra 1 is of Classification Token)
# Q -> Query Sequence length (equal to S for self-attention)
# K -> Key Sequence length   (equal to S for self-attention)
# V -> Value Sequence length (equal to S for self-attention)
# H -> Number of heads
# HE -> Head Embedding Dimension = E/H


class SinusoidalPositionEmbedding2D(nn.Module):
    def __init__(self, n_patches, embed_dim):
        super().__init__()

        self.embed_dim  = embed_dim//2                                              # Split the dimensions into two parts. We will use 1 part for x-axis and the other part for y-axis

        # X-axis specific values
        x_positions = get_x_positions(n_patches).reshape(-1, 1)                     # N  ->  N, 1
        x_pos_embedding = self.generate_sinusoidal1D(x_positions)                   # 1, N, E//2

        # Y-axis specific values
        y_positions = get_y_positions(n_patches).reshape(-1, 1)                     # N  ->  N, 1
        y_pos_embedding = self.generate_sinusoidal1D(y_positions)                   # 1, N, E//2

        # Combine x-axis and y-axis positional encodings
        pos_embedding = torch.cat((x_pos_embedding, y_pos_embedding), -1)           # 1, N, E//2  concat  1, N, E//2  ->  1, N, E
        self.register_buffer("pos_embedding", pos_embedding)                        # Register_buffer for easy switching of device

    def generate_sinusoidal1D(self, sequence):
        # Denominator
        denominator = torch.pow(10000, torch.arange(0, self.embed_dim, 2) / self.embed_dim)   # E//4                     Denominator used to produce sinusoidal equation

        # Create an empty tensor and fill with sin and cos values as per sinusoidal embedding equation
        pos_embedding = torch.zeros(1, sequence.shape[0], self.embed_dim)                     # 1, N, E//2               Used to store positional embedding for x-axis variations
        denominator = sequence / denominator                                                  # N, 1 / (E//4)  ->  N, E//4
        pos_embedding[:, :, ::2]  = torch.sin(denominator)                                    # Fill positional embedding's even dimensions with sin values
        pos_embedding[:, :, 1::2] = torch.cos(denominator)                                    # Fill positional embedding's odd dimensions with cos values
        return pos_embedding                                                                  # return shape 1, N, E//2


class EmbedLayerWithSinusoidal(nn.Module):
    def __init__(self, n_channels, embed_dim, image_size, patch_size, dropout=0.0):
        super().__init__()
        self.conv1         = nn.Conv2d(n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)        # Patch Encoding
        n_patches          = (image_size//patch_size) ** 2
        self.pos_embedding = SinusoidalPositionEmbedding2D(n_patches, embed_dim)                                # 2D Sinusoidal Positional Embedding
        self.cls_token     = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)                     # Classification Token
        self.dropout       = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.conv1.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.trunc_normal_(self.cls_token, mean=0.0, std=0.02)


    def forward(self, x):
        B = x.shape[0]
        x = self.conv1(x)                                                         # B, C, IH, IW     -> B, E, IH/P, IW/P                Split image into the patches and embed patches
        x = x.reshape([B, x.shape[1], -1])                                        # B, E, IH/P, IW/P -> B, E, (IH/P*IW/P) -> B, E, N    Flattening the patches
        x = x.permute(0, 2, 1)                                                    # B, E, N          -> B, N, E                         Rearrange to put sequence dimension in the middle
        x = x + self.pos_embedding.pos_embedding                                  # B, N, E          -> B, N, E                         Add positional embedding
        x = torch.cat((torch.repeat_interleave(self.cls_token, B, 0), x), dim=1)  # B, N, E          -> B, (N+1), E       -> B, S, E    Add classification token at the start of every sequence
        x = self.dropout(x)
        return x
