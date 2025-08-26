import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SKA_PyTorch(nn.Module):
    """
    Spatially adaptive (kernelized) convolution without using unfold.
    Iterates over each offset in the kernel, shifts the input tensor,
    multiplies by the dynamic weight, and accumulates the result.
    This reduces peak memory usage compared to F.unfold.
    """
    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # w: (B, C//G, K2, H, W)  from LKP
        B, C, H, W = x.shape
        G = self.groups
        Cg = C // G
        _, _, K2, _, _ = w.shape
        ks = int(math.sqrt(K2))
        pad = ks // 2

        # Expand dynamic weights across groups and flatten group dimension
        # -> (B, G, Cg, K2, H, W) -> (B, C, K2, H, W)
        w = w.view(B, 1, Cg, K2, H, W).expand(B, G, Cg, K2, H, W)
        w = w.reshape(B, C, K2, H, W)

        # Pad input for easy spatial shifting
        x_pad = F.pad(x, (pad, pad, pad, pad))

        # Prepare output tensor
        out = x.new_zeros(B, C, H, W)

        # Precompute kernel offsets
        # Offsets: [(0,0), (0,1), ..., (ks-1, ks-1)]
        offsets = [(i, j) for i in range(ks) for j in range(ks)]

        # Accumulate shifted inputs * dynamic weights
        for k, (i_off, j_off) in enumerate(offsets):
            x_shift = x_pad[:, :, i_off:i_off + H, j_off:j_off + W]
            # w_k: (B, C, H, W)
            w_k = w[:, :, k, :, :]
            out += x_shift * w_k

        return out

class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

class LKP(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)
        self.sks = sks
        self.groups = groups
        self.dim = dim

    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        B, _, H, W = w.size()
        w = w.view(B, self.dim // self.groups, self.sks ** 2, H, W)
        return w

class LSConv(nn.Module):
    def __init__(self, dim, groups=8):
        super().__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=groups)
        self.ska = SKA_PyTorch(groups=groups)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        # dynamic kernel generation
        w = self.lkp(x)
        # spatially adaptive conv without unfold
        out = self.ska(x, w)
        # batchnorm + residual
        return self.bn(out) + x