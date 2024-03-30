import math
import torch
import torch.nn as nn

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class ResBlock(nn.Module):
    def __init__(
        self, conv=default_conv, n_feats=50, kernel_size=1,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()

        ratio = float(0.5) # mhca channel reduction ratio

        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

        # -- MHCA
        kernel_size_sam = 3
        out_channels = int(n_feats // ratio)
        spatial_attention = [
            nn.Conv2d(in_channels=n_feats, out_channels=out_channels, kernel_size=kernel_size_sam, padding=0, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=n_feats, kernel_size=kernel_size_sam, padding=0, bias=True)
        ]
        channel_attention = [
            nn.Conv2d(in_channels=n_feats, out_channels=out_channels, kernel_size=1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=out_channels, out_channels=n_feats, kernel_size=1, padding=0, bias=True)
        ]

        self.spatial_attention = nn.Sequential(*spatial_attention)
        self.channel_attention = nn.Sequential(*channel_attention)
        self.sigmoid = nn.Sigmoid()

        kernel_size_sam_2 = 5
        spatial_attention_2 = [
            nn.Conv2d(in_channels=n_feats, out_channels=out_channels, kernel_size=kernel_size_sam_2, padding=0, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=n_feats, kernel_size=kernel_size_sam_2, padding=0, bias=True)
        ]
        self.spatial_attention_2 = nn.Sequential(*spatial_attention_2)
        # -- END MHCA

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x



        channel = self.channel_attention(res)
        spatial = self.spatial_attention(res)
        spatial_2 = self.spatial_attention_2(res)
        m_c = self.sigmoid(channel + spatial + spatial_2)
        res = res * m_c


        return res