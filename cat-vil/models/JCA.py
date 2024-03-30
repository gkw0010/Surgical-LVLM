from __future__ import absolute_import
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import functional as F
import sys

class JCA(nn.Module):
    def __init__(self):
        super(JCA, self).__init__()
        #self.corr_weights = torch.nn.Parameter(torch.empty(
        #        1024, 1024, requires_grad=True).type(torch.cuda.FloatTensor))

        self.encoder1 = nn.Linear(768, 384)
        self.encoder2 = nn.Linear(768, 384)

        self.affine_a = nn.Linear(25, 25, bias=False)
        self.affine_v = nn.Linear(25, 25, bias=False)

        self.W_a = nn.Linear(25, 100, bias=False)
        self.W_v = nn.Linear(25, 100, bias=False)
        self.W_ca = nn.Linear(768, 100, bias=False)
        self.W_cv = nn.Linear(768, 100, bias=False)

        self.W_ha = nn.Linear(100, 25, bias=False)
        self.W_hv = nn.Linear(100, 25, bias=False)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.regressor = nn.Sequential(nn.Linear(640, 128),
                                     nn.Dropout(0.6),
                                 nn.Linear(128, 1))

    #def first_init(self):
    #    nn.init.xavier_normal_(self.corr_weights)

    def forward(self, f1_norm, f2_norm):
        fin_audio_features = []
        fin_visual_features = []
        sequence_outs = []

        for i in range(f1_norm.shape[0]):
            audfts = f1_norm[i,:,:]#.transpose(0,1)
            visfts = f2_norm[i,:,:]#.transpose(0,1)
            # print(audfts.shape)
            # print(visfts.shape)
            aud_fts = self.encoder1(audfts)
            vis_fts = self.encoder2(visfts)
            # print(aud_fts.shape)
            # print(vis_fts.shape)

            aud_vis_fts = torch.cat((aud_fts, vis_fts), 1)
            a_t = self.affine_a(aud_vis_fts.transpose(0,1))
            att_aud = torch.mm(aud_fts.transpose(0,1), a_t.transpose(0,1))
            audio_att = self.tanh(torch.div(att_aud, math.sqrt(aud_vis_fts.shape[1])))

            aud_vis_fts = torch.cat((aud_fts, vis_fts), 1)
            v_t = self.affine_v(aud_vis_fts.transpose(0,1))
            att_vis = torch.mm(vis_fts.transpose(0,1), v_t.transpose(0,1))
            vis_att = self.tanh(torch.div(att_vis, math.sqrt(aud_vis_fts.shape[1])))

            H_a = self.relu(self.W_ca(audio_att) + self.W_a(aud_fts.transpose(0,1)))
            H_v = self.relu(self.W_cv(vis_att) + self.W_v(vis_fts.transpose(0,1)))

            att_audio_features = self.W_ha(H_a).transpose(0,1) + aud_fts
            att_visual_features = self.W_hv(H_v).transpose(0,1) + vis_fts

            audiovisualfeatures = torch.cat((att_audio_features, att_visual_features), 1)
            # print(audiovisualfeatures.shape)
            # outs = self.regressor(audiovisualfeatures) #.transpose(0,1))
            #seq_outs, _ = torch.max(outs,0)
            #print(seq_outs)
            sequence_outs.append(audiovisualfeatures)
        #     fin_audio_features.append(att_audio_features)
        #     fin_visual_features.append(att_visual_features)
        # final_aud_feat = torch.stack(fin_audio_features)
        # final_vis_feat = torch.stack(fin_visual_features)
        # print(final_aud_feat.shape, final_aud_feat.shape)
        final_outs = torch.stack(sequence_outs)

        return final_outs
        # return final_outs #final_aud_feat.transpose(1,2), final_vis_feat.transpose(1,2)