from functools import partial

import torch
import torch.nn as nn

from vision_transformer import Block, PatchEmbed
from util.pos_embed import get_2d_sincos_pos_embed
from timm.models.layers import trunc_normal_

class MAE(nn.Module):
    """
    Masked Autoencoder with ViT backbone
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=1,
                 embed_dim=768,
                 depth=24,
                 num_heads=16,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 norm_pix_loss=False,
                 mask_ratio=0.75):
        super().__init__()
        # ---------Encoder----------------
        self.mask_ratio = mask_ratio
        # print("at start, ", mask_ratio)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.blocks = nn.ModuleList([
            Block(int(embed_dim*(1-mask_ratio)), num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)
        ])
        
        self.norm = norm_layer(int(embed_dim*(1-mask_ratio)))
        # ---------Decoder----------------
        self.decoder_embed = nn.Linear(int(embed_dim*(1-mask_ratio)), decoder_embed_dim, bias=True)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)
        ])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
        
        # --------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()
        
    def initialize_weights(self):
        # initialize patch embedding like nn.Linear
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def patchify(self, img):
        """
        reshapes an input image into patches
        img: (B, C, H, W)
        x: (N, L, patch_size**2 * C)
        grayscale image: C=1
        """
        p = self.patch_embed.patch_size[0]
        assert img.shape[2] == img.shape[3] and img.shape[2] % p == 0
        
        h = w = img.shape[2] // p   # number of patches
        x = img.reshape(shape=(img.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(img.shape[0], h*w, p**2 * 1))
        return x
    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * C)
        img: (N, C, H, W)
        grayscale image: C=1
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        img = x.reshape(shape=(x.shape[0], 1, h*p, w*p))
        return img
    
    def random_masking(self, x, mask_ratio):
        """
        x: (N, L, D), sequence
        """
        N, L, D = x.shape   # batch, number of patches, embed dim
        len_keep = int(D * (1 - mask_ratio))
        
        noise = torch.rand(N, L, D, device=x.device)   # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=-1)   # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=-1) # shape: (32, 768), (N, D)
        
        # keep the first subset
        ids_keep = ids_shuffle[..., :len_keep]
        x_masked = torch.gather(x, dim=-1, index=ids_keep)
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L, D], device = x.device)
        mask[..., :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=-1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        # print(mask_ratio)
        # embed patches
        x = self.patch_embed(x) # output shape -> (N, L, D), batch size, sequence length == number of patches, embed dim
        x, mask, ids_restore = self.random_masking(x, mask_ratio)   # -> (N, L, D*(1-mask_ratio))
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # (N, L, D_remaining)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        """input x have shape (N, L, embed_dim)"""
        # embed tokens
        x = self.decoder_embed(x)       # -> (N, L, decoder_embed_dim)
        
        # # append mask tokens to sequence
        # mask_tokens = self.mask_token.repeat(x.shape[0], x.shape[1], ids_restore.shape[2] - x.shape[2])
        # x_ = torch.cat([x, mask_tokens], dim=2)  # no cls token, shape: (N, L, decoder_embed_dim)
        # x_ = torch.gather(x_, dim=2, index=ids_restore.repeat(1, 1, x.shape[2])) # unshuffle to original order based on ids_restore

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # predictor projection
        x = self.decoder_pred(x)
        
        return x
    
    def forward_loss(self, imgs, pred):
        """
        imgs: (N, C, H, W)
        pred: (N, L, p*p*C)
        mask: (N, L), 0 is keep, 1 is remove
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:      # optional: normalize pixel values
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)    # [N, L], mean loss per patch
        # mean loss for all patches
        loss = loss.sum(dim=-1) / loss.shape[1] # [N], mean loss per sample
        loss = loss.mean()  # scalar
        return loss
    
    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(*kwargs):
    model = MAE(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MAE(
        patch_size=16, embed_dim=768, depth=12, num_heads=1,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MAE(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks