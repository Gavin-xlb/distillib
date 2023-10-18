from ._base import Distiller
import torch
from torch import nn
import torch.nn.functional as F
from utils.loss_functions import *

class SP(Distiller):
    def __init__(self, student, teacher):
        super(SP, self).__init__(student, teacher)
        self.gamma = 3000

    def forward(self, batch, flag, support_image, support_mask, **kwargs):
        ct, mask, name = batch
        self.t_net.eval()
        t_out, t_low, t_high = self.t_net.net(ct)
        output, low, high, = self.net(ct)

        loss_seg = calc_loss(output, mask)
        loss_sp = self.sp_loss(low, t_low) + self.sp_loss(high, t_high)
        
        loss = loss_seg + loss_sp * self.gamma
        return loss

    def sp_loss(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), -1)
        G_s  = torch.mm(fm_s, fm_s.t())
        norm_G_s = F.normalize(G_s, p=2, dim=1)

        fm_t = fm_t.view(fm_t.size(0), -1)
        G_t  = torch.mm(fm_t, fm_t.t())
        norm_G_t = F.normalize(G_t, p=2, dim=1)

        loss = F.mse_loss(norm_G_s, norm_G_t)
        return loss