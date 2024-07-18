import torch

# IH -> Image Height
# IW -> Image Width
# P -> Patch Size
# N_ -> Number of Patches in 1 dimension = IH/P = IW/P
# N -> Number of Patches = IH/P * IW/P


# X-axis specific values
def get_x_positions(n_patches, start_idx=0):
    n_patches_ = int(n_patches ** 0.5)                                    # Number of patches along 1 dimension

    x_positions = torch.arange(start_idx, n_patches_ + start_idx)         # N_
    x_positions = x_positions.unsqueeze(0)                                # 1, N_
    x_positions = torch.repeat_interleave(x_positions, n_patches_, 0)     # N_ , N_                         Matrix to replicate positions of patches on x-axis
    x_positions = x_positions.reshape(-1)                                 # N_ , N_  ->  N_ ** 2  =  N

    return x_positions


# Y-axis specific values
def get_y_positions(n_patches, start_idx=0):
    n_patches_ = int(n_patches ** 0.5)                                    # Number of patches along 1 dimension

    y_positions = torch.arange(start_idx, n_patches_+start_idx)           # N_
    y_positions = torch.repeat_interleave(y_positions, n_patches_, 0)     # N_ , N_  ->  N_ ** 2  =  N                  Matrix to replicate positions of patches on y-axis

    return y_positions


# Print arguments
def print_args(args):
    for k in dict(sorted(vars(args).items())).items():
        print(k)
    print()


