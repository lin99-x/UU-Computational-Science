import os
from enum import Enum
import torch
from torch import _assert
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from functools import partial
import collections.abc
from itertools import repeat

# from visualizer import get_local

_HAS_FUSED_ATTN = hasattr(torch.nn.functional, 'scale_dot_product_attention')
if 'TIMM_FUSED_ATTN' in os.environ:
    _USE_FUSED_ATTN = int(os.environ['TIMM_FUSED_ATTN'])
else:
    _USE_FUSED_ATTN = 1

# set to True if want to export a model with same padding via ONNX
_EXPORTABLE = False

def _ntuple(n):        
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

def use_fused_attn(experimental: bool = False) -> bool:
    if not _HAS_FUSED_ATTN or _EXPORTABLE:
        return False
    if experimental:
        return _USE_FUSED_ATTN > 1
    return _USE_FUSED_ATTN > 0      # 0 == off, 1 == on, 2 == on (for experimental use)
    
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob > 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) # work with diff dim tensor, not just 2D ConvNets
    random_tensor = x.new_empty(shape).beroulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class Format(str, Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'
    
def nchw_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NHWC:
        x = x.permute(0, 2, 3, 1)
    elif fmt == Format.NLC:
        x = x.flatten(2).transpose(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(2)
    return x

class Mlp(nn.Module):
    """ MLP as used in vit, MLP-Mixer and related networks"""
    def __init__(self,
                 in_features,
                 hidden_features = None,
                 out_features = None,
                 act_layer = nn.GELU,
                 norm_layer = None,
                 bias = True,
                 drop = 0.,
                 use_conv = False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)   # ensures that bias is a tuple of length 2
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size = 1) if use_conv else nn.Linear
        
        self.fc1 = linear_layer(in_features, hidden_features, bias = bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias = bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        
        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    # @get_local('attn')
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if self.fused_attn:
            x = F.scale_dot_product_attention(q, k, v, dropout_p = self.attn_drop.p if self.training else 0.)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class LayerScale(nn.Module):
    def __init__(self,
                 dim: int,
                 init_values: float = 1e-5,
                 inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    
class DropPath(nn.Module):
    """ Drop paths (stochastic depth) per sample (when applied in main path of residual blocks)"""
    def __init__(self,
                 drop_prob: float = 0., 
                 scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    def extra_repr(self):
        return f'drop_prob = {round(self.drop_prob, 3):0.3f}'
    
class Block(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_norm: bool = False,
                 proj_drop: float = 0.,
                 attn_drop: float = 0.,
                 init_values: Optional[float] = None,
                 drop_path: float = 0.,
                 act_layer: nn.Module = nn.Sigmoid,
                 norm_layer: nn.Module = nn.LayerNorm,
                 mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads = num_heads,
            qkv_bias = qkv_bias,
            qk_norm = qk_norm,
            attn_drop = attn_drop,
            proj_drop = proj_drop,
            norm_layer = norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values = init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(in_features = dim, hidden_features = int(dim * mlp_ratio), act_layer = act_layer, drop = proj_drop)
        self.ls2 = LayerScale(dim, init_values = init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
    
class PatchEmbed(nn.Module):
    """ Turns a 2D input image into a 1D sequence learnable embedding vector. """
    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]
    
    def __init__(self,
                 img_size: Optional[int] = 224,
                 patch_size: int = 16,
                 in_chans: int = 1,
                 embed_dim: int = 768,
                 norm_layer: Optional[Callable] = None,
                 flatten: bool = True,
                 output_fmt: Optional[str] = None,
                 bias: bool = True,
                 strict_img_size: bool = True,
                 dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)    
        if img_size is not None:
            self.img_size = to_2tuple(img_size)     
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None
        
        if output_fmt is not None:                  # don't flatten if output_fmt is specified
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            self.flatten = flatten
            self.output_fmt = Format.NCHW           # number of samples, channels, height, width
        
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size = patch_size, stride = patch_size, bias = bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                _assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
                _assert(W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
            elif not self.dynamic_img_pad:
                _assert(
                    H % self.patch_size[0] == 0,
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                )
                _assert(
                    W % self.patch_size[1] == 0,
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
                )
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h)) # pad left, right, top, bottom, in-place operation
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2) # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x
            
        
    
    