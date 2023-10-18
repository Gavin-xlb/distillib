from ._base import Distiller
import torch
from torch import nn
import torch.nn.functional as F
from utils.loss_functions import *

def prediction_map_distillation(y, teacher_scores, T=4) :
    """
    basic KD loss function based on "Distilling the Knowledge in a Neural Network"
    https://arxiv.org/abs/1503.02531
    :param y: student score map
    :param teacher_scores: teacher score map
    :param T:  for softmax
    :return: loss value
    """
    p = F.log_softmax(y / T, dim=1)
    q = F.softmax(teacher_scores / T, dim=1)

    p = p.view(-1, 2)
    q = q.view(-1, 2)

    l_kl = F.kl_div(p, q, reduction='batchmean') * (T ** 2)
    return l_kl


def at(x, exp):
    """
    attention value of a feature map
    :param x: feature
    :return: attention value
    """
    return F.normalize(x.pow(exp).mean(1).view(x.size(0), -1))


def importance_maps_distillation(s, t, exp=4):
    """
    importance_maps_distillation KD loss, based on "Paying More Attention to Attention:
    Improving the Performance of Convolutional Neural Networks via Attention Transfer"
    https://arxiv.org/abs/1612.03928
    :param exp: exponent
    :param s: student feature maps
    :param t: teacher feature maps
    :return: imd loss value
    """
    if s.shape[2] != t.shape[2]:
        s = F.interpolate(s, t.size()[-2:], mode='bilinear')
    return torch.sum((at(s, exp) - at(t, exp)).pow(2), dim=1).mean()


def region_contrast(x, gt):
    """
    calculate region contrast value
    :param x: feature
    :param gt: mask
    :return: value
    """
    gt_shape_h = gt.shape[2]
    x_shape_h = x.shape[2]
    if x_shape_h > gt_shape_h:
        x = F.adaptive_avg_pool2d(x, (gt_shape_h, gt_shape_h))
    elif x_shape_h < gt_shape_h:
        gt = F.adaptive_avg_pool2d(gt, (x_shape_h, x_shape_h))
        # x = F.interpolate(x, (gt_shape_h, gt_shape_h), mode="nearest")
    smooth = 1.0
    mask0 = gt[:, 0].unsqueeze(1) # 背景
    mask1 = gt[:, 1].unsqueeze(1) # 目标
    
    region0 = torch.sum(x * mask0, dim=(2, 3)) / torch.sum(mask0, dim=(2, 3))
    region1 = torch.sum(x * mask1, dim=(2, 3)) / (torch.sum(mask1, dim=(2, 3)) + smooth)

    return F.cosine_similarity(region0, region1, dim=1)


def region_affinity_distillation(s, t, gt):
    """
    region affinity distillation KD loss
    :param s: student feature
    :param t: teacher feature
    :return: loss value
    """
    gt = F.interpolate(gt, s.size()[2:])
    return (region_contrast(s, gt) - region_contrast(t, gt)).pow(2).mean()


class EMKD(Distiller):
    def __init__(self, student, teacher):
        super(EMKD, self).__init__(student, teacher)
        # KD loss para
        self.alpha = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.9

    def forward(self, batch, flag, support_image, support_mask, **kwargs):
        ct, mask, name = batch
        self.t_net.eval()
        t_out, t_low, t_high = self.t_net.net(ct)
        output, low, high, = self.net(ct)

        loss_seg = calc_loss(output, mask)

        loss_pmd = prediction_map_distillation(output, t_out)
        loss_imd = importance_maps_distillation(low, t_low) + importance_maps_distillation(high, t_high)
        loss_rad = region_affinity_distillation(low, t_low, mask) + region_affinity_distillation(high, t_high, mask)

        loss = loss_seg + self.alpha * loss_pmd + self.beta1 * loss_imd + self.beta2 * loss_rad
        return loss
