'''
这是一个基于注意力增强的低级特征和高级特征之间的KD
'''
from ._base import Distiller
import torch
from torch import nn
import torch.nn.functional as F
from utils.loss_functions import *


def hcl(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        #  统一student和teacher的尺寸
        s_H, t_H = fs.shape[2], ft.shape[2]
        if s_H > t_H:
            fs = F.adaptive_avg_pool2d(fs, (t_H, t_H))
        elif s_H < t_H:
            # ft = F.adaptive_avg_pool2d(ft, (s_H, s_H))
            fs = F.interpolate(fs, (t_H, t_H), mode="bilinear")

        n,c,h,w = fs.shape
        # 对齐通道
        fs = F.normalize(fs.mean(1))
        ft = F.normalize(ft.mean(1))
        loss = F.mse_loss(fs, ft, reduction='mean')
        cnt = 1.0
        tot = 1.0
        for l in [4,2,1]:
            if l >=h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
            tmpft = F.adaptive_avg_pool2d(ft, (l,l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all

def mse(fstudent, fteacher):
    loss = 0.0
    for fs, ft in zip(fstudent, fteacher):
        #  统一student和teacher的尺寸
        s_H, t_H = fs.shape[2], ft.shape[2]
        if s_H > t_H:
            fs = F.adaptive_avg_pool2d(fs, (t_H, t_H))
        elif s_H < t_H:
            ft = F.adaptive_avg_pool2d(ft, (s_H, s_H))

        # 对齐通道
        fs = F.normalize(fs.mean(1))
        ft = F.normalize(ft.mean(1))

        loss += F.mse_loss(fs, ft, reduction='mean')
    return loss

# class StudentTrans(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(StudentTrans, self).__init__()
#         self.conv_low = nn.Sequential(
#             nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels[0]),
#         )
#         self.conv_high = nn.Sequential(
#             nn.Conv2d(in_channels[1], out_channels[1],kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels[1]),
#         )

#     def forward(self, student_features):
#         ## results中存放转换之后的student_feature,且通道数于teacher相对应的一致
#         low, high = student_features
#         low = self.conv_low(low)
#         high = self.conv_high(high)

#         return [low, high]

def cal_fea_attention(feat, att_enhanced_weight, p=2):
    # 按通道平方取均值，得到每个样本的注意力图n,h,w
    att = F.normalize(feat.pow(p).mean(1) * att_enhanced_weight)
    # 扩张c通道 n,c(于feat的通道数一致),h,w
    return att.unsqueeze(1).repeat(1,feat.shape[1],1,1)

class EhcdAttKD(Distiller):
    def __init__(self, student, teacher):
        super(EhcdAttKD, self).__init__(student, teacher)
        # KD loss para
        self.FEAT_WEIGHT = 1.0
        self.ATTENTION_ENHANCED_WEIGHT = 10.0

    def forward(self, batch, flag, support_image, support_mask, **kwargs):
        ct, mask, name = batch
        self.t_net.eval()
        t_out, t_low, t_high = self.t_net.net(ct)
        output, low, high = self.net(ct)
        # 注意力增强的低级和高级特征
        s_features_att = [low * cal_fea_attention(low, self.ATTENTION_ENHANCED_WEIGHT), high * cal_fea_attention(high, self.ATTENTION_ENHANCED_WEIGHT)]
        t_features_att = [t_low * cal_fea_attention(t_low, self.ATTENTION_ENHANCED_WEIGHT), t_high * cal_fea_attention(t_high, self.ATTENTION_ENHANCED_WEIGHT)]
        
        # # 如果利用卷积将学生的通道变换成教师的通道，在loss中就不需要再对齐通道了
        # in_channels = [low.shape[1], high.shape[1]]
        # out_channels = [t_low.shape[1], t_high.shape[1]]
        # # 通道对齐
        # s_features_trans = StudentTrans(in_channels, out_channels).cuda()(s_features_att)
        # loss_att = hcl(s_features_trans, t_features_att)

        loss_att = hcl(s_features_att, t_features_att)
        # loss_review = mse(s_features_trans, t_features_att)
        loss_seg = calc_loss(output, mask)

        # losses
        loss = loss_seg + self.FEAT_WEIGHT * loss_att
        return loss