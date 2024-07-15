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
Rotational Positional Embedding was originally designed for 1-D data.
To adapt it for Images, I used a simple trick of splitting the dimensions into two parts. 
I use the first half of the dimensions for x-axis and the second half for y-axis.
Another challenge is handling the classification token for which I set m=0 for classification token i.e. position=0 (generally starts at 1).
This results in no rotations (no change) on the classification token and others tokens are handled just like a regular transformer.
'''

class RotatoryPositionEmbedding2D(nn.Module):
	def __init__(self, seq_len, embed_dim):
		super().__init__()
		self.embed_dim = embed_dim // 2                                           # Split the dimensions into two parts. We will use 1 part for x-axis and the other part for y-axis

		n_patches = seq_len - 1

		x_positions  = get_x_positions(n_patches, start_idx=1).reshape(-1, 1)     # N  ->  N, 1
		x_sin, x_cos = self.generate_rope1D(x_positions)						  # 1, 1, N, E//2    ,    1, 1, N, E//2
		self.register_buffer("x_cos", x_cos)                        			  # Register_buffer for easy switching of device
		self.register_buffer("x_sin", x_sin)                        			  # Register_buffer for easy switching of device

		y_positions  = get_y_positions(n_patches, start_idx=1).reshape(-1, 1)     # N  ->  N, 1
		y_sin, y_cos = self.generate_rope1D(y_positions)						  # 1, 1, N, E//2    ,    1, 1, N, E//2
		self.register_buffer("y_cos", y_cos)                        			  # Register_buffer for easy switching of device
		self.register_buffer("y_sin", y_sin)                        			  # Register_buffer for easy switching of device


	def generate_rope1D(self, sequence):
		'''
		Create theta as per the equation in the RoPe paper: theta = 10000 ^ -2(i-1)/d for i belongs to [1, 2, ... d/2].  
		Note this d/2 is different from previous x/y axis split.
		'''
		sequence   = F.pad(sequence, (0, 0, 1, 0))                           					# N, 1        ->  N + 1, 1 = S      Pad with 0 to account for classification token		
		thetas 	   = -2 * torch.arange(start=1, end=self.embed_dim//2+1) / self.embed_dim       # E//4
		thetas 	   = torch.repeat_interleave(thetas, 2, 0)                                      # E//2
		thetas 	   = torch.pow(10000, thetas)                                                   # E//2
		values 	   = sequence * thetas                                                     		# S, 1 * E//2 -> S, E//2
		cos_values = torch.cos(values).unsqueeze(0).unsqueeze(0)		                   		# N, E//2     -> 1, 1, N, E//2      Precompute and store cos values
		sin_values = torch.sin(values).unsqueeze(0).unsqueeze(0)		                   		# N, E//2     -> 1, 1, N, E//2      Precompute and store sin values
		return sin_values, cos_values		


	def forward(self, x):
		x_x = x[:, :, :, :self.embed_dim]                                            # B, H, S, E//2                                            Split half of the embeddings of x for x-axis
		x_y = x[:, :, :, self.embed_dim:]                                            # B, H, S, E//2                                            Split half of the embeddings of x for y-axis

		x_x1 = x_x * self.x_cos                                                   	 # B, H, S, E//2  *  1, 1, S, E//2   ->  B, H, S, E//2      Multiply x-axis part of input with its cos factor as per the eq in RoPe
		x_x_shifted = torch.stack((-x_x[:, :, :, 1::2], x_x[:, :, :, ::2]), -1)      # B, H, S, E//2                     ->  B, H, S, E//4, 2   Shift values for sin multiplications are per the eq in RoPe
		x_x_shifted = x_x_shifted.reshape(x_x.shape)                                 # B, H, S, E//4, 2                  ->  B, H, S, E//2
		x_x2 = x_x_shifted * self.x_sin                                           	 # B, H, S, E//2  *  1, 1, S, E//2   ->  B, S, E//2         Multiply x-axis part of x with its sin factor as per the eq in RoPe
		x_x = x_x1 + x_x2                                                            # Add sin and cosine value
		
		x_y1 = x_y * self.y_cos                                                   	 # B, H, S, E//2  *  1, 1, S, E//2   ->  B, H, S, E//2      Multiply y-axis part of input with its cos factor as per the eq in RoPe
		x_y_shifted = torch.stack((-x_y[:, :, :, 1::2], x_y[:, :, :, ::2]), -1)    	 # B, H, S, E//2                     ->  B, H, S, E//4, 2   Shift values for sin multiplications are per the eq in RoPe
		x_y_shifted = x_y_shifted.reshape(x_y.shape)                               	 # B, H, S, E//4, 2                  ->  B, H, S, E//2
		x_y2 = x_y_shifted * self.y_sin                                           	 # B, H, S, E//2  *  1, 1, S, E//2   ->  B, H, S, E//2      Multiply y-axis part of x with its sin factor as per the eq in RoPe
		x_y = x_y1 + x_y2                                                            # Add sin and cosine value

		x = torch.cat((x_x, x_y), -1)                                                # B, H, S, E//2  cat  B, H, S, E//2 -> B, H, S, E          Combine x and y rotational projections
		return x

		
class SelfAttentionWithRope(nn.Module):
	def __init__(self, embed_dim, n_attention_heads, seq_len):
		super().__init__()
		self.embed_dim         = embed_dim
		self.n_attention_heads = n_attention_heads
		self.head_embed_dim    = embed_dim // n_attention_heads

		self.queries           = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)   # Queries projection
		self.keys              = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)   # Keys projection
		self.values            = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)   # Values projection
		self.out_projection    = nn.Linear(self.head_embed_dim * self.n_attention_heads, self.embed_dim)   # Out projection

		self.rotary_embedding  = RotatoryPositionEmbedding2D(seq_len=seq_len, embed_dim=self.head_embed_dim)  # Rotation for Queries and keys (Used for both because it applies the same funtions).

	def forward(self, x):
		b, s, e = x.shape  # Note: In case of self-attention Q, K and V are all equal to S

		xq = self.queries(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)     # B, Q, E      ->  B, Q, (H*HE)  ->  B, Q, H, HE
		xq = xq.permute(0, 2, 1, 3)                                                         # B, Q, H, HE  ->  B, H, Q, HE
		xk = self.keys(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)        # B, K, E      ->  B, K, (H*HE)  ->  B, K, H, HE
		xk = xk.permute(0, 2, 1, 3)                                                         # B, K, H, HE  ->  B, H, K, HE
		xv = self.values(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)      # B, V, E      ->  B, V, (H*HE)  ->  B, V, H, HE
		xv = xv.permute(0, 2, 1, 3)                                                         # B, V, H, HE  ->  B, H, V, HE

		# Rotate Queries and Keys only
		xq = self.rotary_embedding(xq)                                                      # B, Q, H, HE  ->  B, Q, V, HE
		xk = self.rotary_embedding(xk)                                                      # B, K, H, HE  ->  B, K, V, HE  

		# Compute Attention presoftmax values
		xk = xk.permute(0, 1, 3, 2)                                                         # B, H, K, HE  ->  B, H, HE, K
		x_attention = torch.matmul(xq, xk)                                                  # B, H, Q, HE  *   B, H, HE, K   ->  B, H, Q, K    (Matmul tutorial eg: A, B, C, D  *  A, B, E, F  ->  A, B, C, F   if D==E)

		x_attention /= float(self.head_embed_dim) ** 0.5                                    # Scale presoftmax values for stability

		x_attention = torch.softmax(x_attention, dim=-1)                                    # Compute Attention Matrix

		x = torch.matmul(x_attention, xv)                                                   # B, H, Q, K  *  B, H, V, HE  ->  B, H, Q, HE     Compute Attention product with Values

		# Format the output
		x = x.permute(0, 2, 1, 3)                                                           # B, H, Q, HE -> B, Q, H, HE
		x = x.reshape(b, s, e)                                                              # B, Q, H, HE -> B, Q, (H*HE)

		x = self.out_projection(x)                                                          # B, Q, (H*HE)-> B, Q, E
		return x
