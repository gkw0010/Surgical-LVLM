'''
Acknowledgement : Code adopted from the official implementation of MCAN model from OpenVQA (https://github.com/MILVLG/openvqa).
'''

import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x

class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, hidden_size):
        super(MHAtt, self).__init__()
        # self.__C = __C
        self.hidden_size = hidden_size
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_merge = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(0.1)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            8,
            int(self.hidden_size / 8)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            8,
            int(self.hidden_size / 8)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            8,
            int(self.hidden_size / 8)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidden_size
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, hidden_size, ff_size):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=ff_size,
            out_size=hidden_size,
            dropout_r=0.1,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, hidden_size, ff_size):
        super(SA, self).__init__()

        self.mhatt = MHAtt(hidden_size)
        self.ffn = FFN(hidden_size, ff_size)

        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(0.1)
        self.norm2 = LayerNorm(hidden_size)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, hidden_size, ff_size):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(hidden_size)
        self.mhatt2 = MHAtt(hidden_size)
        self.ffn = FFN(hidden_size, ff_size)

        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(0.1)
        self.norm2 = LayerNorm(hidden_size)

        self.dropout3 = nn.Dropout(0.1)
        self.norm3 = LayerNorm(hidden_size)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


class BISGA(nn.Module):
    def __init__(self, hidden_size, ff_size):
        super(BISGA, self).__init__()

        self.mhatt1 = MHAtt(hidden_size)
        self.mhatt2 = MHAtt(hidden_size)
        self.ffn = FFN(hidden_size, ff_size)

        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(0.1)
        self.norm2 = LayerNorm(hidden_size)

        self.dropout3 = nn.Dropout(0.1)
        self.norm3 = LayerNorm(hidden_size)
        ################################################

        self.mhatt1_0 = MHAtt(hidden_size)
        self.mhatt2_0 = MHAtt(hidden_size)
        self.ffn_0 = FFN(hidden_size, ff_size)

        self.dropout1_0 = nn.Dropout(0.1)
        self.norm1_0 = LayerNorm(hidden_size)

        self.dropout2_0 = nn.Dropout(0.1)
        self.norm2_0 = LayerNorm(hidden_size)

        self.dropout3_0 = nn.Dropout(0.1)
        self.norm3_0 = LayerNorm(hidden_size)
        
    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask)
        ))
        ################################
        y = self.norm1(y + self.dropout1(
            self.mhatt1(v=y, k=y, q=y, mask=y_mask)
        ))
        ##############################
        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask)
        ))
        ################################
        y = self.norm2(y + self.dropout2(
            self.mhatt2(v=x, k=x, q=y, mask=x_mask)
        ))
        ##############################
        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))
        y = self.norm3(y + self.dropout3(
            self.ffn(y)
        ))
        return x, y


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, mca_hidden_size, mca_ffn_size, layers):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(mca_hidden_size, mca_ffn_size) for _ in range(layers)])
        self.dec_list = nn.ModuleList([SGA(mca_hidden_size, mca_ffn_size) for _ in range(layers)])
        # self.dec_list = nn.ModuleList([SA(mca_hidden_size, mca_ffn_size) for _ in range(layers)])
    def forward(self, y, x, y_mask, x_mask):
        # Get encoder last hidden vector
        for enc in self.enc_list:
            y = enc(y, y_mask)

        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for dec in self.dec_list:
            x = dec(x, y, x_mask, y_mask)

        # for dec in self.dec_list:
        #     x = dec(x, x_mask)
        return y, x
    
class BIMCA_ED(nn.Module):
    def __init__(self, mca_hidden_size, mca_ffn_size, layers):
        super(BIMCA_ED, self).__init__()

        self.dec_list = nn.ModuleList([BISGA(mca_hidden_size, mca_ffn_size) for _ in range(layers)])
                
    def forward(self, y, x, y_mask, x_mask):
        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for dec in self.dec_list:
            x, y = dec(x, y, x_mask, y_mask)

        return y, x

# class MHCA_Fusion(nn.Module):
#     def __init__(self, n_feats, ratio):
#         """
#         MHCA spatial-channel attention module.
#         :param n_feats: The number of filter of the input.
#         :param ratio: Channel reduction ratio.
#         """
#         super(MHCA_Fusion, self).__init__()

#         out_channels = int(n_feats // ratio)

#         head_1 = [
#             nn.Conv2d(in_channels=n_feats, out_channels=out_channels, kernel_size=1, padding=0, bias=True),
#             nn.ReLU(True),
#             nn.Conv2d(in_channels=out_channels, out_channels=n_feats, kernel_size=1, padding=0, bias=True)
#         ]

#         kernel_size_sam = 3
#         head_2 = [
#             nn.Conv2d(in_channels=n_feats, out_channels=out_channels, kernel_size=kernel_size_sam, padding=0, bias=True),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(in_channels=out_channels, out_channels=n_feats, kernel_size=kernel_size_sam, padding=0, bias=True)
#         ]

#         kernel_size_sam_2 = 5
#         head_3 = [
#             nn.Conv2d(in_channels=n_feats, out_channels=out_channels, kernel_size=kernel_size_sam_2, padding=0, bias=True),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(in_channels=out_channels, out_channels=n_feats, kernel_size=kernel_size_sam_2, padding=0, bias=True)
#         ]

#         self.head_1 = nn.Sequential(*head_1)
#         self.head_2 = nn.Sequential(*head_2)
#         self.head_3 = nn.Sequential(*head_3)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         res_h1 = self.head_1(x)
#         res_h2 = self.head_2(x)
#         res_h3 = self.head_3(x)
#         m_c = self.sigmoid(res_h1 + res_h2 + res_h3)
#         res = x * m_c
#         return res
    
