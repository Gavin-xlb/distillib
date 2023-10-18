from ._base import Distiller
import torch
from torch import nn
import torch.nn.functional as F
from utils.loss_functions import *
from torch.nn.modules.loss import KLDivLoss, CrossEntropyLoss

class GID(Distiller):
    def __init__(self, student, teacher):
        super(GID, self).__init__(student, teacher)
        

    def forward(self, batch, flag, support_image, support_mask, **kwargs):
        ct, mask, name = batch
        self.t_net.eval()
        t_out, t_low, t_high = self.t_net.net(ct)
        output, low, high, = self.net(ct)
        ce_loss = CrossEntropyLoss().cuda()
        dice_loss = DiceLoss(n_classes=2).cuda()
        kl_loss = KLLoss().cuda()

        loss_ce = ce_loss(output, mask[:, 1:].contiguous().squeeze(1).long())
        loss_dice = dice_loss(output, mask, softmax=True)
        loss_l = 0.5 * loss_ce + 0.5 * loss_dice
        loss_avg = AvgpoolLoss()(output, t_out.detach())
        
        loss = loss_l + 0.01 * loss_avg
        return loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        # target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
        self.kl = KLDivLoss(size_average=False, reduce=True)

    def forward(self, inputs, target):
        sum = inputs.shape[-1]*inputs.shape[-2]
        log_input = F.log_softmax(inputs)
        target = F.softmax(target)
        return self.kl(log_input, target)/sum
    
class AvgpoolLoss(nn.Module):
    def __init__(self):
        super(AvgpoolLoss, self).__init__()
        self.avgpool = nn.AvgPool2d((2, 2), stride=(2, 2))
        self.kl = KLLoss()

    def forward(self, inputs, target):
        inputs = self.avgpool(inputs.float())
        inputs = self.avgpool(inputs)
        inputs = self.avgpool(inputs)
        target = self.avgpool(target.float())
        target = self.avgpool(target)
        target = self.avgpool(target)
        return self.kl(inputs, target)