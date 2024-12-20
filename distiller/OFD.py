from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from ._base import Distiller
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.loss_functions import *


'''
Modified from https://github.com/clovaai/overhaul-distillation/blob/master/CIFAR-100/distiller.py
'''
class Connector(nn.Module):
    '''
    A Comprehensive Overhaul of Feature Distillation
    http://openaccess.thecvf.com/content_ICCV_2019/papers/
    Heo_A_Comprehensive_Overhaul_of_Feature_Distillation_ICCV_2019_paper.pdf
    '''
    def __init__(self, in_channels, out_channels):
        super(Connector, self).__init__()
        self.connector = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, fm_s, fm_t):
        margin = self.get_margin(fm_t)
        fm_t = torch.max(fm_t, margin)
        fm_s = self.connector(fm_s)
        n1,_,h1,w1 = fm_t.shape
        n2,_,h2,w2 = fm_s.shape
        # 尺寸相同
        if h2 < h1:
            fm_s = F.interpolate(fm_s, (h1, w1), mode="bilinear")
        elif h2 > h1:
            fm_s = F.adaptive_avg_pool2d(fm_s, (h1, w1))

        mask = 1.0 - ((fm_s <= fm_t) & (fm_t <= 0.0)).float()
        loss = torch.mean((fm_s - fm_t)**2 * mask)

        return loss

    def get_margin(self, fm, eps=1e-6):
        mask = (fm < 0.0).float()
        masked_fm = fm * mask

        margin = masked_fm.sum(dim=(0,2,3), keepdim=True) / (mask.sum(dim=(0,2,3), keepdim=True)+eps)

        return margin

class OFD(Distiller):
    def __init__(self, student, teacher):
        super(OFD, self).__init__(student, teacher)
        self.alpha = 1.0

    def forward(self, batch, flag, support_image, support_mask, **kwargs):
        ct, mask, name = batch
        self.t_net.eval()
        t_out, t_low, t_high = self.t_net.net(ct)
        output, low, high = self.net(ct)
        in_channels = [low.shape[1], high.shape[1]]
        out_channels = [t_low.shape[1], t_high.shape[1]]
        loss_ofd = Connector(in_channels[0], out_channels[0]).cuda()(low, t_low) + Connector(in_channels[1], out_channels[1]).cuda()(high, t_high)
        loss_ofd /= 2
        loss_seg = calc_loss(output, mask)

        # losses
        loss = loss_seg + self.alpha * loss_ofd
        return loss
    
