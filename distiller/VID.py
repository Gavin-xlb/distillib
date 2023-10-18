import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ._base import Distiller


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, padding=0, bias=False, stride=stride
    )


def vid_loss(regressor, log_scale, f_s, f_t, eps=1e-5):
    # pool for dimentsion match
    s_H, t_H = f_s.shape[2], f_t.shape[2]
    if s_H > t_H:
        f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
    elif s_H < t_H:
        f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
    else:
        pass
    pred_mean = regressor(f_s)
    pred_var = torch.log(1.0 + torch.exp(log_scale)) + eps
    pred_var = pred_var.view(1, -1, 1, 1).to(pred_mean)
    neg_log_prob = 0.5 * ((pred_mean - f_t) ** 2 / pred_var + torch.log(pred_var))
    loss = torch.mean(neg_log_prob)
    return loss

def get_feat_shapes(student, teacher):
    data = torch.randn(2, 1, 384, 384).cuda()
    with torch.no_grad():
        _, low, high = student(data)
        _, t_low, t_high = teacher(data)
    feat_s_shapes = [f.shape for f in [low, high]]
    feat_t_shapes = [f.shape for f in [t_low, t_high]]
    return feat_s_shapes, feat_t_shapes

class VID(Distiller):
    """
    Variational Information Distillation for Knowledge Transfer (CVPR 2019),
    code from author: https://github.com/ssahn0215/variational-information-distillation
    """

    def __init__(self, student, teacher):
        super(VID, self).__init__(student, teacher)
        self.ce_loss_weight = 1.0
        self.feat_loss_weight = 1.0
        self.init_pred_var = 5.0
        self.eps = 1e-5
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.net, self.t_net
        )
        feat_s_channels = [s[1] for s in feat_s_shapes]
        feat_t_channels = [s[1] for s in feat_t_shapes]
        self.init_vid_modules(feat_s_channels, feat_t_channels)

    def init_vid_modules(self, feat_s_shapes, feat_t_shapes):
        self.regressors = nn.ModuleList()
        self.log_scales = []
        for s, t in zip(feat_s_shapes, feat_t_shapes):
            regressor = nn.Sequential(
                conv1x1(s, t), nn.ReLU(), conv1x1(t, t), nn.ReLU(), conv1x1(t, t)
            ).cuda()
            self.regressors.append(regressor)
            log_scale = torch.nn.Parameter(
                np.log(np.exp(self.init_pred_var - self.eps) - 1.0) * torch.ones(t)
            )
            self.log_scales.append(log_scale)

    def get_learnable_parameters(self):
        parameters = super().get_learnable_parameters()
        for regressor in self.regressors:
            parameters += list(regressor.parameters())
        return parameters

    def get_extra_parameters(self):
        num_p = 0
        for regressor in self.regressors:
            for p in regressor.parameters():
                num_p += p.numel()
        return num_p

    def forward(self, batch, flag, support_image, support_mask, **kwargs):
        ct, mask, name = batch
        self.t_net.eval()
        t_output, t_low, t_high = self.t_net.net(ct)
        output, low, high = self.net(ct)
        feature_student = [low, high]
        feature_teacher = [t_low, t_high]
        loss_ce = self.ce_loss_weight * F.cross_entropy(output, mask[:, 1:].contiguous().squeeze(1).long())
        loss_vid = 0
        for i in range(2):
            loss_vid += vid_loss(
                self.regressors[i],
                self.log_scales[i],
                feature_student[i],
                feature_teacher[i],
                self.eps,
            )
        loss_vid = self.feat_loss_weight * loss_vid
        loss = loss_ce * self.ce_loss_weight + loss_vid * self.feat_loss_weight
        return loss