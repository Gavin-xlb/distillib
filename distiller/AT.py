from ._base import Distiller
import torch
from torch import nn
import torch.nn.functional as F
from utils.loss_functions import *

def single_stage_at_loss(f_s, f_t, p):
    def _at(feat, p):
        return F.normalize(feat.pow(p).mean(1).reshape(feat.size(0), -1))
    # print('t_shape', f_t.shape)
    # print('s_shape', f_s.shape)
    s_H, t_H = f_s.shape[2], f_t.shape[2]
    if s_H > t_H:
        f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
    elif s_H < t_H:
        f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
    return (_at(f_s, p) - _at(f_t, p)).pow(2).mean()


def at_loss(g_s, g_t, p):
    return sum([single_stage_at_loss(f_s, f_t, p) for f_s, f_t in zip(g_s, g_t)])

class AT(Distiller):
    def __init__(self, student, teacher):
        super(AT, self).__init__(student, teacher)
        # KD loss para
        self.p = 2
        self.ce_loss_weight = 1.0
        self.feat_loss_weight = 1000.0

    def forward(self, batch, flag, support_image, support_mask, **kwargs):
        ct, mask, name = batch
        self.t_net.eval()
        t_out, t_low, t_high = self.t_net.net(ct)
        output, low, high, = self.net(ct)

        loss_seg = calc_loss(output, mask)

        # losses
        loss_ce = self.ce_loss_weight * loss_seg
        loss_feat = self.feat_loss_weight * at_loss(
            [low, high], [t_low, t_high], self.p
        )
        loss = loss_ce + loss_feat
        return loss