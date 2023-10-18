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
    
class MASK(nn.Module):
    def __init__(self, stu_channel, tea_channel):
        super(MASK, self).__init__()
        self.stu_channel = stu_channel
        self.tea_channel = tea_channel
        self.generation = nn.Sequential(
            nn.Conv2d(tea_channel, tea_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(tea_channel, tea_channel, kernel_size=3, padding=1))
        
    def forward(self, stu_fea):
        stu_fea = self.generation(stu_fea)
        return stu_fea

class MGD(Distiller):

    """PyTorch version of `Masked Generative Distillation`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        name (str): the loss name of the layer
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00002
        lambda_mgd (float, optional): masked ratio. Defaults to 0.75
    """
    def __init__(self,
                 student,
                 teacher):
        super(MGD, self).__init__(student,teacher)
        self.alpha_mgd = 0.00002
        self.lambda_mgd = 0.75

    def forward(self, batch, flag, support_image, support_mask, **kwargs):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        ct, mask, name = batch
        self.t_net.eval()
        preds_T, t_low, t_high = self.t_net.net(ct)
        preds_S, low,high = self.net(ct)

        loss_seg = calc_loss(preds_S, mask)
    
        loss = self.get_dis_loss(high, t_high) * self.alpha_mgd + loss_seg
            
        return loss

    def get_dis_loss(self, preds_S, preds_T):
        # print(preds_S.shape)
        # print(preds_T.shape)
        # assert 1==2
        # 保证H,W一样
        s_H, t_H = preds_S.shape[2], preds_T.shape[2]
        if s_H > t_H:
            preds_S = F.adaptive_avg_pool2d(preds_S, (t_H, t_H))
        elif s_H < t_H:
            preds_T = F.adaptive_avg_pool2d(preds_T, (s_H, s_H))
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        # 保证通道一样
        align_module = ALIGN(preds_S.shape[1], preds_T.shape[1]).cuda()
        if align_module.align is not None:
            preds_S = align_module.align(preds_S)

        # device = preds_S.device
        mat = torch.rand((N,1,H,W)).cuda()
        mat = torch.where(mat>1-self.lambda_mgd, 0, 1).cuda()

        masked_fea = torch.mul(preds_S, mat)
        new_fea = MASK(preds_S.shape[1], preds_T.shape[1]).cuda()(masked_fea)

        dis_loss = loss_mse(new_fea, preds_T)/N

        return dis_loss