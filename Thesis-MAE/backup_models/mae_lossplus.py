from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision_transformer import Block, PatchEmbed
from util.pos_embed import get_2d_sincos_pos_embed
from util.sobelfilter import SobelFilter

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
                 norm_pix_loss=False):
        super().__init__()
        
        # ---------Encoder----------------
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        
        # ---------Decoder----------------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)
        ])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
        self.sobel_filter = SobelFilter()
        # --------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()
        
    def initialize_weights(self):
        # initialize and freeze the positional embeddings (sin-cos)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0)) # copy to nn.Parameter
        
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        # initialize patch embedding like nn.Linear
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        
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
        N, L, D = x.shape   # batch, length (h*w), dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)   # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)   # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1) # shape: (32, 196), (N, L)
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device = x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x) # output shape -> (N, L, D), batch size, sequence length == number of patches, embed dim
        
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]    # -> (N, L, D)
        
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)       # -> (N, L+1, D)
        
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        """input x have shape (N, L, embed_dim)"""
        # embed tokens
        x = self.decoder_embed(x)       # -> (N, L, decoder_embed_dim)
        
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token, shape: (N, L, decoder_embed_dim)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])) # unshuffle to original order based on ids_restore
        x = torch.cat([x[:, :1, :], x_], dim=1) # add cls token
        
        # add pos embed
        x = x + self.decoder_pos_embed
        
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # predictor projection
        x = self.decoder_pred(x)
        
        # remove cls token
        x = x[:, 1:, :]
        
        return x
    
    def forward_loss(self, imgs, pred, mask, edge_loss_weight):
        """
        imgs: (N, C, H, W), C=1
        pred: (N, L, p*p*C)
        mask: (N, L), 0 is keep, 1 is remove
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:      # optional: normalize pixel values
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        reconstruction_loss = (pred - target) ** 2
        reconstruction_loss = reconstruction_loss.mean(dim=-1)    # [N, L], MSE
        
        reconstruction_loss = (reconstruction_loss * mask).sum() / mask.sum() # mean loss on removed patches
        # print("reconstruction loss is: ", reconstruction_loss.item())
        edge_loss = self.get_edge_loss(target, pred, edge_loss_weight=edge_loss_weight)
        # print("edge loss is: ", edge_loss.item())
        
        loss = reconstruction_loss + edge_loss
        
        return loss
    
    def get_edge_loss(self, imgs, pred, edge_loss_weight=0.):
        """
        imgs: (N, L, p*p*C)
        pred: (N, L, p*p*C)
        """
        ori, pred = self.unpatchify(imgs), self.unpatchify(pred)        # -> (N, C, H, W)
        edge_map_ori, edge_map_pred = self.sobel_filter(ori), self.sobel_filter(pred)
        edge_loss = F.mse_loss(edge_map_ori, edge_map_pred, reduction='mean')
        edge_loss = edge_loss_weight * edge_loss
        return edge_loss
        
    def forward(self, imgs, mask_ratio=0.75, edge_loss_weight=0.):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred,mask, edge_loss_weight=edge_loss_weight)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MAE(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MAE(
        patch_size=16, embed_dim=1024, depth=12, num_heads=16,
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