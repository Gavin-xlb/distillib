import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.loss_functions import *
from ._base import Distiller


class ALIGN(nn.Module):
    def __init__(self, stu_channel, tea_channel):
        super(ALIGN, self).__init__()
        self.stu_channel = stu_channel
        self.tea_channel = tea_channel
        if stu_channel != tea_channel:
            self.align = nn.Conv2d(stu_channel, tea_channel, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None
        
    def forward(self, stu_fea):
        if self.align is not None:
            stu_fea = self.align(stu_fea)
        return stu_fea

# y模仿teacher_scores/teacher_scores指导y
def kl(y, teacher_scores, T=4) :
    p = F.log_softmax(y / T, dim=1)
    q = F.softmax(teacher_scores / T, dim=1)

    p = p.view(-1, 2)
    q = q.view(-1, 2)

    kl = F.kl_div(p, q, reduction='batchmean') * (T ** 2)
    return kl

class LSR(Distiller):
    def __init__(self, student, teacher):
        super(LSR, self).__init__(student,teacher)
        # # 从正确的分类中(像素值为1)拿出epsilon的概率给背景(像素值为0)
        # self.epsilon = 0.1
        # 学生的输出去模仿gt的系数
        self.alpha_lsr = 0.95
        # 软化标签时的温度
        self.temperature = 40

    def forward(self, batch, flag, support_image, support_mask, **kwargs):
        ct, mask, name = batch
        self.t_net.eval()
        t_output, t_low, t_high = self.t_net.net(ct)
        output, low,high = self.net(ct)

        loss = self.alpha_lsr * kl(output, mask, self.temperature) + (1 - self.alpha_lsr) * kl(output, t_output, self.temperature)
            
        return loss