import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelFilter(nn.Module):
    def __init__(self,
                 in_channels: int=1,
                 out_channels: int=2,
                 kernel_size: int=3):
        super(SobelFilter, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sobel_filter = self.setup_filter()
    
    def setup_filter(self):
        sobel_filter = nn.Conv2d(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 kernel_size=self.kernel_size,
                                 stride = 1,
                                 padding= 1,
                                 bias=False)
        Gx = torch.FloatTensor([[1, 0, -1], 
                           [2, 0, -2],
                           [1, 0, -1]]) 
        Gy = torch.FloatTensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        sobel_filter.weight = nn.Parameter(G, requires_grad=False)
        return sobel_filter
    
    def forward(self, img):
        N, C, H, W = img.shape
        assert C == self.in_channels and H == W == 224
        x = self.sobel_filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x        