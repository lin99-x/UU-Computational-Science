import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size, dtype=np.float32) # generate an array of values starting from 0 and ending at grid_size-1, with a step size of 1
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h) # a list of two 2-dimensional arrays, weidht goes first and then height
    grid = np.stack(grid, axis=0)   # shape is (2, grid_size, grid_size)
    
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    grid: [2, 1, grid_size, grid_size]
    return:
    pos_embed: [grid_size*grid_size, embed_dim]
    """
    assert embed_dim % 2 == 0
    
    # use half of the dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim//2, grid[0]) # (H*W, d/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim//2, grid[1]) # (H*W, d/2)
    
    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, d)
    return emb
    
    
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension of each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, d)
    """
    assert embed_dim % 2 ==0    # embed_dim must be even, since we need half for sin and half for cos
    omega = np.arange(embed_dim//2, dtype = np.float)
    omega /= embed_dim / 2.     # range from 0 to 1
    omega = 1. / 10000**omega   # (d/2,) range from 1 to 1/10000
    
    pos = pos.reshape(-1)       # (M,)
    out = np.einsum('m,d->md', pos, omega) # (M, d/2)
    
    emb_sin = np.sin(out) # (M, d/2)
    emb_cos = np.cos(out) # (M, d/2)
    
    emb = np.concatenate([emb_sin, emb_cos], axis = 1) # (M, d)
    return emb


# -------visualization of positional embedding--------
P = get_2d_sincos_pos_embed(768, 14, cls_token=False)
cax = plt.matshow(P)
plt.gcf().colorbar(cax)
plt.savefig('pos_embed.png')
