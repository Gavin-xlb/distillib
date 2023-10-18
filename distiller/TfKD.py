import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.loss_functions import *
from ._base import Distiller



class TfKD(Distiller):
    '''
    Tf-KD有两种实现方式,一种是以训练好的学生模型作为教师,训练相同的学生模型,二是人为设计一个表现很好的教师输出的分布来作为教师模型
    我们选用第一种，因为第二种针对分类任务，如果要适用于分割，需要进行修改
    '''
    def __init__(self, student, teacher):
        super(TfKD, self).__init__(student,teacher)
        self.alpha = 0.95
        self.temperature = 40
        self.multiplier = 1.0

    def loss_kd_self(self, outputs, labels, teacher_outputs):
        """
        loss function for self training: Tf-KD_{self}
        the teacher is the same as student ,which is pre-trained
        """
        alpha = self.alpha
        T = self.temperature
        multiplier = self.multiplier

        # loss_CE = F.cross_entropy(outputs, labels.long())
        loss_seg = calc_loss(outputs, labels)
        D_KL = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (T * T) * multiplier  # multiple is 1.0 in most of cases, some cases are 10 or 50
        KD_loss =  (1. - alpha)*loss_seg + alpha*D_KL

        return KD_loss


    def loss_kd_regularization(self, outputs, labels):
        """
        loss function for mannually-designed regularization: Tf-KD_{reg}
        """

        alpha = self.alpha
        T = self.temperature
        multiplier = self.multiplier
        correct_prob = 0.99    # the probability for correct class in u(k)
        loss_CE = F.cross_entropy(outputs, labels)
        K = outputs.size(1)

        teacher_soft = torch.ones_like(outputs).cuda()
        teacher_soft = teacher_soft*(1-correct_prob)/(K-1)  # p^d(k)
        for i in range(outputs.shape[0]):
            teacher_soft[i ,labels[i]] = correct_prob
        loss_soft_regu = nn.KLDivLoss()(F.log_softmax(outputs, dim=1), F.softmax(teacher_soft/T, dim=1))*multiplier

        KD_loss = (1. - alpha)*loss_CE + alpha*loss_soft_regu

        return KD_loss


    def forward(self, batch, flag, support_image, support_mask, **kwargs):
        ct, mask, name = batch
        self.t_net.eval()
        t_out, t_low, t_high = self.t_net.net(ct)
        output, low, high = self.net(ct)
        # self
        KD_loss = self.loss_kd_self(output, mask, t_out)
        # reg
        # KD_loss = self.loss_kd_regularization(output, mask)

        return KD_loss