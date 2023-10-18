import torch
from pytorch_lightning.core import LightningModule
import numpy as np
from hausdorff import hausdorff_distance

class BasePLModel(LightningModule):
    def __init__(self):
        super(BasePLModel, self).__init__()
        self.metric = {}
        self.num_class = 2

    def training_epoch_end(self, outputs):
        train_loss_mean = 0
        for output in outputs:
            train_loss_mean += output['loss']

        train_loss_mean /= len(outputs)

        # log training accuracy at the end of an epoch
        self.log('train_loss', train_loss_mean)

    def validation_epoch_end(self, outputs):
        return self.test_epoch_end(outputs)

    def measure(self, batch, output):
        ct, mask, name = batch
        # print(name) # '00000'/'00001'...
        # print('mask_shape', mask.shape)  # [16,2,384,384]
        # print('output_shape', output.shape)  # [16,2,384,384]
        # 只需要目标类的output和mask
        output = torch.softmax(output, dim=1)[:, 1:].contiguous()
        mask = mask[:, 1:].contiguous()
        # print('mask_shape_a', mask.shape)  # [16,1,384,384]
        # print('output_shape_a', output.shape)  # [16,1,384,384]
        # threshold value
        output = (output > 0.4).float()

        # record values concerned with dice score
        for ib in range(len(ct)):
            # cal dice
            pre = torch.sum(output[ib], dim=(1, 2))
            gt = torch.sum(mask[ib], dim=(1, 2)) # shape 1   
            inter = torch.sum(torch.mul(output[ib], mask[ib]), dim=(1, 2))
            # dice_temp = (2*inter+1.0)/(pre+gt+1.0)
            # if dice_temp < 0.5 and dice_temp > 0.4:
            #     print(name[ib])
            # cal hausdorff_distance
            hd = hausdorff_distance(output[ib].squeeze().detach().cpu().numpy(), mask[ib].squeeze().detach().cpu().numpy(), distance="euclidean")
            hd = torch.tensor(hd).unsqueeze(0).cuda() # shape 1
            init = torch.ones_like(pre) # 每个病例的切片数
            if name[ib] not in self.metric.keys():
                self.metric[name[ib]] = torch.stack((pre, gt, inter, hd, init), dim=0)
            else:
                self.metric[name[ib]] += torch.stack((pre, gt, inter, hd, init), dim=0)

    def test_epoch_end(self, outputs):
        # calculate dice score
        num_class = self.num_class - 1
        scores = torch.zeros((num_class, 4))
        nums = torch.zeros((num_class, 1))
        # print('nums_init', nums)
        # print('self.metric.items()', self.metric.items())
        for k, v in self.metric.items():
            # print('v', v)
            # print('v.shape', v.shape)  # [3, 1]
            dice = (2. * v[2] + 1.0) / (v[0] + v[1] + 1.0)
            voe = (2. * (v[0] - v[2])) / (v[0] + v[1] + 1e-7)
            rvd = v[0] / (v[1] + 1e-7) - 1.
            # print('K=%s:dice=%f' % (k, dice))
            hd = v[3] / v[4]
            for i in range(num_class):
                # the dice is nonsensical when gt is 0
                if v[1][i].item() != 0:
                    nums[i] += 1
                    scores[i][0] += dice[i].item()
                    scores[i][1] += voe[i].item()
                    scores[i][2] += rvd[i].item()
                    scores[i][3] += hd[i].item()

        scores = scores / nums

        for i in range(num_class):
            # the dice is nonsensical when gt is 0
            self.log('dice_class{}'.format(i), scores[i][0].item())
            self.log('voe_class{}'.format(i), scores[i][1].item())
            self.log('rvd_class{}'.format(i), scores[i][2].item())
            self.log('hd_class{}'.format(i), scores[i][3].item())

            print('dice_class{}: {}'.format(i, scores[i][0].item()))
            print('voe_class{}: {}'.format(i, scores[i][1].item()))
            print('rvd_class{}: {}'.format(i, scores[i][2].item()))
            print('hd_class{}: {}'.format(i, scores[i][2].item()))

        self.metric = {}
