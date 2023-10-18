import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.loss_functions import *
from ._base import Distiller

class KD(Distiller):
    def __init__(self, student, teacher):
        super(KD, self).__init__(student,teacher)

    def forward(self, batch, flag, support_image, support_mask, **kwargs):
        ct, mask, name = batch
        self.t_net.eval()
        t_output, t_low, t_high = self.t_net.net(ct)
        output, low,high = self.net(ct)
        loss_hard = calc_loss(output, mask)
        loss_soft = self.mask_loss(output, t_output)
        loss = loss_hard * 0.1 + loss_soft * 0.9
        return loss
    
    def mask_loss(self, stu_mask, teacher_mask, T=4):
        p = F.log_softmax(stu_mask / T, dim=1)
        q = F.softmax(teacher_mask / T, dim=1)

        p = p.view(-1, 2)
        q = q.view(-1, 2)

        mask_loss = F.kl_div(p, q, reduction='batchmean') * (T ** 2)
        return mask_loss