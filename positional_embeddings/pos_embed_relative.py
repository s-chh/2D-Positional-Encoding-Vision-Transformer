import torch
import torch.nn as nn
import torch.nn.functional as F
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

'''
A big challenge with 2D Relative Positional Embedding is handling classification token in Vision Transformers. 
One solution is to not update the classification token.
Instead, I add a value (0) in the embedding lookup tables for classification token to determine its distance. 
That's why embedding table distance contains an extra item (additional to itself). 
This makes embedding table contain 2 * max_relative_distance + 2 items.
0 is used to calculate distance between an image patch token and the classification token.
'''

class RelativePositionEmbedding2D(nn.Module):
    def __init__(self, embed_dim, seq_len, max_relative_dist):                                                         
        super().__init__()

        self.max_relative_dist = max_relative_dist                                                      # Max relative distance to clamp distance. Distance grether than this will be set to max relative distance

        # Embedding tables contain only half the embedding to split the dimensions into two parts. We will use 1 part for x-axis and the other part for y-axis
        self.x_relative_embedding_table = nn.Embedding(max_relative_dist*2+1+1, embed_dim//2)           # Embedding table to store embeddings for different relative distances for x-axis positions
        self.y_relative_embedding_table = nn.Embedding(max_relative_dist*2+1+1, embed_dim//2)           # Embedding table to store embeddings for different relative distances for y-axis positions

        # X-axis specific values
        x_positions = get_x_positions(seq_len-1)                                                        # N  ->  N, 1
        x_distances = self.generate_relative1D_distances(x_positions)                                   # N, N                Precompute and Store distances for faster processing
        self.register_buffer("x_distances", x_distances)                                                # Register_buffer for easy switching of device

        # Y-axis specific values
        y_positions = get_y_positions(seq_len-1)                                                        # N  ->  N, 1
        y_distances = self.generate_relative1D_distances(y_positions)                                   # N, N                Precompute and Store distances for faster processing
        self.register_buffer("y_distances", y_distances)                                                # Register_buffer for easy switching of device


    def generate_relative1D_distances(self, positions):
        distances = positions.unsqueeze(0) - positions.unsqueeze(1)                                     # 1, N  -  N, 1  ->  N, N          Compute distance matrix along x axis only
        distances = torch.clamp(distances, -self.max_relative_dist, self.max_relative_dist)             # N, N  ->  N, N                   Clamp the distances
        distances = distances + self.max_relative_dist + 1                                              # N, N  ->  N, N                   Shift the values to be between (1 and max_relative_dist + 1)
        distances = F.pad(input=distances, pad=(1, 0, 1, 0), mode='constant', value=0)                  # N, N  ->  (N+1), (N+1) = S, S    Add zeros for distances to classification token
        return distances


    def forward(self):
        x_pos_embedding = self.x_relative_embedding_table(self.x_distances)                                   # S, S  ->  S, S, E//2             Generate embedding
        y_pos_embedding = self.y_relative_embedding_table(self.y_distances)                                   # S, S  ->  S, S, E//2             Generate embedding

        # Combine x-axis and y-axis positional encodings
        pos_embedding = torch.cat((x_pos_embedding, y_pos_embedding), -1)                                     # S, S, E//2  concat  S, S, E//2  ->  S, S, E
        return pos_embedding


class SelfAttentionWithRelative(nn.Module):
    def __init__(self, embed_dim, n_attention_heads, seq_len, max_relative_dist):
        super().__init__()
        self.embed_dim           = embed_dim
        self.n_attention_heads   = n_attention_heads
        self.head_embed_dim      = embed_dim // n_attention_heads

        self.queries             = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)          # Queries projection
        self.keys                = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)          # Keys projection
        self.values              = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)          # Values projection
        self.out_projection      = nn.Linear(self.head_embed_dim * self.n_attention_heads, self.embed_dim)          # Out projection

        self.query_pos_embedding = RelativePositionEmbedding2D(self.head_embed_dim, seq_len, max_relative_dist)     # Relative Positional Embedding for Queries
        self.value_pos_embedding = RelativePositionEmbedding2D(self.head_embed_dim, seq_len, max_relative_dist)     # Relative Positional Embedding for Values

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

        # Add positional embedding to presoftmax values
        q_pos = self.query_pos_embedding()                                                  # Q, K, E
        q_pos = q_pos.permute(0, 2, 1)                                                      # Q, K, E -> Q, E, K

        xq = xq.permute(2, 0, 1, 3)                                                         # B, H, Q, HE   ->    Q, B, H, E
        xq = xq.reshape(s, -1, self.head_embed_dim)                                         # Q, B, H, E    ->    Q, (BH), E

        relative_xq = torch.matmul(xq, q_pos)                                               # Q, (BH), E    @     Q, E, K -> Q, (BH), K
        relative_xq = relative_xq.reshape(s, b, -1, s)                                      # Q, (BH), K   ->     Q, B, H, K
        relative_xq = relative_xq.permute(1, 2, 0, 3)                                       # Q, B, H, K   ->     B, H, Q, K

        x_attention = x_attention + relative_xq                                             # B, H, Q, K   +      B, H, Q, K  ->  B, H, Q, K

        # Continue attention calculations
        x_attention /= float(self.head_embed_dim) ** 0.5                                    # Scale presoftmax values for stability

        x_attention = torch.softmax(x_attention, dim=-1)                                    # Compute Attention Matrix

        x = torch.matmul(x_attention, xv)                                                   # B, H, Q, K  *  B, H, V, HE  ->  B, H, Q, HE     Compute Attention product with Values

        # Compute relative xv
        v_pos = self.value_pos_embedding()                                                  # Q, K, E
        x_attention = x_attention.permute(2, 0, 1, 3)                                       # B, H, Q, K   ->   Q, B, H, K
        x_attention = x_attention.reshape(s, -1, s)                                         # Q, B, H, K   ->   Q, (BH), K
        v_relative = torch.matmul(x_attention, v_pos)                                       # Q, (BH), K   @    Q, K, E   ->   Q, (BH), E
        v_relative = v_relative.reshape(s, b, -1, self.head_embed_dim)                      # Q, (BH), E   ->   Q, B, H, E
        v_relative = v_relative.permute(1, 2, 0, 3)                                         # Q, B, H, E   ->   B, H, Q, E
        x = x + v_relative                                                                  # B, H, Q, E   +    B, H, Q, E   ->   B, H, Q, E

        # Format the output
        x = x.permute(0, 2, 1, 3)                                                           # B, H, Q, HE -> B, Q, H, HE
        x = x.reshape(b, s, e)                                                              # B, Q, H, HE -> B, Q, (H*HE)

        x = self.out_projection(x)                                                          # B, Q, (H*HE) -> B, Q, E
        return x
