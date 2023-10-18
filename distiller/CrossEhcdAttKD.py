from ._base import Distiller
import torch
from torch import nn
import torch.nn.functional as F
from utils.loss_functions import *

class CrossAttention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CrossAttention, self).__init__()
        if in_channel != out_channel:
            self.align = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channel),
                # nn.LeakyReLU() # 没有激活函数一般会更好
            )
            nn.init.kaiming_uniform_(self.align[0].weight, a=1)  # pyre-ignore
        else:
            self.align = None

    def forward(self, stu_feat, tea_feat):
        
        if self.align is not None:
            # transform student features
            stu_feat = self.align(stu_feat) # 通道数与teacher相同
        n1,_,h1,w1 = tea_feat.shape
        n2,_,h2,w2 = stu_feat.shape
        # 尺寸相同
        if h2 < h1:
            # stu_feat = F.interpolate(stu_feat, (h1, w1), mode="nearest")
            # stu_feat = F.interpolate(stu_feat, (h1, w1), mode="bilinear")
            tea_feat = F.adaptive_avg_pool2d(tea_feat, (h2, w2))
        elif h2 > h1:
            stu_feat = F.adaptive_avg_pool2d(stu_feat, (h1, w1))
        

        return stu_feat, tea_feat # shape完全相同
    
def cal_fea_attention(feat, att_enhanced_weight, p=2):
    # 按通道平方取均值，得到每个样本的注意力图n,h,w
    n, c, h, w = feat.shape
    att = (F.normalize(feat.pow(p).mean(1).view(n, -1), dim=1)).view(n, h, w) * att_enhanced_weight # 归一化很必要,对h*w归一化
    # 扩张c通道 n,c(与feat的通道数一致),h,w
    return att.unsqueeze(1).repeat(1,c,1,1) * feat

def msloss(fs, ft):
    n,c,h,w = fs.shape
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
    return loss

def mse(fs, ft):
    loss = F.mse_loss(fs, ft, reduction='mean')
    return loss

# student teacher的输出loss
def mask_loss(stu_mask, teacher_mask, T=4):
    p = F.log_softmax(stu_mask / T, dim=1)
    q = F.softmax(teacher_mask / T, dim=1)

    p = p.view(-1, 2)
    q = q.view(-1, 2)

    mask_loss = F.kl_div(p, q, reduction='batchmean') * (T ** 2)
    return mask_loss

class CrossEhcdAttKD(Distiller):
    def __init__(self, student, teacher):
        super(CrossEhcdAttKD, self).__init__(student, teacher)
        # KD loss para
        self.ATTENTION_ENHANCED_WEIGHT = 20.0
        self.REVIEW = 0.6
        self.FORWARD = 0.5
        self.CROSS_LOSS_WEIGHT = 0.08
        self.MASK_LOSS_WEIGHT = 0.1999
        # self.ATTENTION_ENHANCED_WEIGHT = 20.0
        # self.CROSS_LOSS_WEIGHT = 0.08
        # self.REVIEW = 0.3
        # self.FORWARD = 0.3
        # self.MASK_LOSS_WEIGHT = 0.1999

    def CED_module(self, low, high, t_low, t_high):
        # 注意力增强的低级和高级特征
        stu_featatt_low = cal_fea_attention(low, self.ATTENTION_ENHANCED_WEIGHT)
        stu_low_channel = stu_featatt_low.shape[1]
        stu_featatt_high = cal_fea_attention(high, self.ATTENTION_ENHANCED_WEIGHT)
        stu_high_channel = stu_featatt_high.shape[1]
        tea_featatt_low = cal_fea_attention(t_low, self.ATTENTION_ENHANCED_WEIGHT)
        tea_low_channel = tea_featatt_low.shape[1]
        tea_featatt_high = cal_fea_attention(t_high, self.ATTENTION_ENHANCED_WEIGHT)
        tea_high_channel = tea_featatt_high.shape[1]

        # 交叉学习特征
        stu_featatt_low1, tea_featatt_low1 = CrossAttention(stu_low_channel, tea_low_channel).cuda()(stu_featatt_low, tea_featatt_low)
        loss1 = msloss(stu_featatt_low1, tea_featatt_low1)
        stu_featatt_high1, tea_featatt_low2 = CrossAttention(stu_high_channel, tea_low_channel).cuda()(stu_featatt_high, tea_featatt_low)
        loss2 = msloss(stu_featatt_high1, tea_featatt_low2)
        stu_featatt_high2, tea_featatt_high1 = CrossAttention(stu_high_channel, tea_high_channel).cuda()(stu_featatt_high, tea_featatt_high)
        loss3 = msloss(stu_featatt_high2, tea_featatt_high1)
        stu_featatt_low2, tea_featatt_high2 = CrossAttention(stu_low_channel, tea_high_channel).cuda()(stu_featatt_low, tea_featatt_high)
        loss4 = msloss(stu_featatt_low2, tea_featatt_high2)
        
        loss_cross = loss1 + loss2 * self.REVIEW + loss3 + loss4 * self.FORWARD
        return loss_cross

    def forward(self, batch, flag, support_image, support_mask, **kwargs):
        ct, mask, name = batch
        if flag is not None: # universeg
            support_image = support_image.cuda()
            support_mask = support_mask.cuda()
        self.t_net.eval()
        t_out, t_low, t_high = self.t_net.net(ct)
        if flag is not None: # universeg
            output, low, high = self.net(ct, support_image, support_mask)
        else:
            output, low, high = self.net(ct)
        
        loss_cross = self.CED_module(low = low, high = high, t_low = t_low, t_high = t_high)
        loss_seg = calc_loss(output, mask)

        loss = loss_seg + self.CROSS_LOSS_WEIGHT * loss_cross + self.MASK_LOSS_WEIGHT * mask_loss(output, t_out)
        # Ablation
        # loss = loss_seg + self.CROSS_LOSS_WEIGHT * loss_cross #ced
        # loss = loss_seg + self.MASK_LOSS_WEIGHT * mask_loss(output, t_out) #mmd
        return loss
