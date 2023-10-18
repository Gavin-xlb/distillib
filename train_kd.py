import torch
import numpy as np, random
import os
import argparse
from train import SegPL
from networks import get_model
from utils.loss_functions import *
from torch.utils.data import DataLoader
from utils.base_pl_model import BasePLModel
from datasets.midataset import SliceDataset
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.io import TorchCheckpointIO as tcio
# from pytorch_lightning import seed
# from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from distiller import distiller_dict
import time

def init_seed(seed):
    # defining random initial seeds
    print('######defining random initial seeds######')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
# seed = seed_everything(123)
seed = 3407
init_seed(seed)

kits_ckpt_dict = {
    'organ': {
        'raunet': 'checkpoint_kits_organ_raunet_epoch=053-dice_class0=0.9689.ckpt',
        'pspnet': 'checkpoint_kits_organ_pspnet_epoch=049-dice_class0=0.9618.ckpt',
        'unet': 'checkpoint_kits_organ_unet_epoch=047-dice_class0=0.9665.ckpt',
        'unet++': 'checkpoint_kits_organ_unet++_epoch=046-dice_class0=0.9666.ckpt',
        'enet': 'checkpoint_kits_organ_enet_epoch=074-dice_class0=0.9654.ckpt',
        'mobilenetv2': 'checkpoint_kits_organ_mobilenetv2_epoch=065-dice_class0=0.9400.ckpt',
        'resnet18': 'checkpoint_kits_organ_resnet18_epoch=074-dice_class0=0.9272.ckpt'
    },
    'tumor': {
        'raunet': 'checkpoint_kits_tumor_raunet_epoch=064-dice_class0=0.7856.ckpt',
        'pspnet': 'checkpoint_kits_tumor_pspnet_epoch=056-dice_class0=0.7494.ckpt',
        'unet': 'checkpoint_kits_tumor_unet_epoch=082-dice_class0=0.6433.ckpt',
        'unet++': 'checkpoint_kits_tumor_unet++_epoch=078-dice_class0=0.6649.ckpt',
        'enet': 'checkpoint_kits_tumor_enet_epoch=046-dice_class0=0.5261.ckpt',
        'mobilenetv2': 'checkpoint_kits_tumor_mobilenetv2_epoch=068-dice_class0=0.6783.ckpt',
        'resnet18': 'checkpoint_kits_tumor_resnet18_epoch=059-dice_class0=0.5068.ckpt',
        'universeg': 'checkpoint_kits_tumor_universeg_epoch=000-dice_class0=0.1933.ckpt'
    }
}

lits_ckpt_dict = {
    'organ': {
        'raunet': 'checkpoint_lits_organ_raunet_epoch=032-dice_class0=0.9627.ckpt',
        'pspnet': 'checkpoint_lits_organ_pspnet_epoch=031-dice_class0=0.9611.ckpt',
        'unet': 'checkpoint_lits_organ_unet_epoch=047-dice_class0=0.9549.ckpt',
        'unet++': 'checkpoint_lits_organ_unet++_epoch=042-dice_class0=0.9546.ckpt',
        'enet': 'checkpoint_lits_organ_enet_epoch=022-dice_class0=0.9603.ckpt',
        'mobilenetv2': 'checkpoint_lits_organ_mobilenetv2_epoch=069-dice_class0=0.9473.ckpt',
        'resnet18': 'checkpoint_lits_organ_resnet18_epoch=044-dice_class0=0.9450.ckpt'
    },
    'tumor': {
        'raunet': 'checkpoint_lits_tumor_raunet_epoch=030-dice_class0=0.6136.ckpt',
        'pspnet': 'checkpoint_lits_tumor_pspnet_epoch=035-dice_class0=0.6441.ckpt',
        'unet': 'checkpoint_lits_tumor_unet_epoch=072-dice_class0=0.6081.ckpt',
        'unet++': 'checkpoint_lits_tumor_unet++_epoch=081-dice_class0=0.6207.ckpt',
        'enet': 'checkpoint_lits_tumor_enet_epoch=059-dice_class0=0.5769.ckpt',
        'mobilenetv2': 'checkpoint_lits_tumor_mobilenetv2_epoch=069-dice_class0=0.5601.ckpt',
        'resnet18': 'checkpoint_lits_tumor_resnet18_epoch=038-dice_class0=0.4659.ckpt'
    }
}

# kits_ckpt_dict = {
#     'organ': {
#         'raunet': 'checkpoint_kits_organ_raunet_epoch=053-dice_class0=0.9689.ckpt',
#         'pspnet': 'checkpoint_kits_organ_pspnet_epoch=039-dice_class0=0.9646.ckpt',
#         'unet': 'checkpoint_kits_organ_unet_epoch=047-dice_class0=0.9665.ckpt',
#         'unet++': 'checkpoint_kits_organ_unet++_epoch=046-dice_class0=0.9666.ckpt',
#         'enet': 'checkpoint_kits_organ_enet_epoch=074-dice_class0=0.9654.ckpt',
#         'mobilenetv2': 'checkpoint_kits_organ_mobilenetv2_epoch=069-dice_class0=0.9421.ckpt',
#         'resnet18': 'checkpoint_kits_organ_resnet18_epoch=049-dice_class0=0.9264.ckpt'
#     },
#     'tumor': {
#         'raunet': 'checkpoint_kits_tumor_raunet_epoch=064-dice_class0=0.7856.ckpt',
#         'pspnet': 'checkpoint_kits_tumor_pspnet_epoch=068-dice_class0=0.7171.ckpt',
#         'unet': 'checkpoint_kits_tumor_unet_epoch=082-dice_class0=0.6433.ckpt',
#         'unet++': 'checkpoint_kits_tumor_unet++_epoch=078-dice_class0=0.6649.ckpt',
#         'enet': 'checkpoint_kits_tumor_enet_epoch=046-dice_class0=0.5261.ckpt',
#         'mobilenetv2': 'checkpoint_kits_tumor_mobilenetv2_epoch=070-dice_class0=0.6891.ckpt',
#         'resnet18': 'checkpoint_kits_tumor_resnet18_epoch=057-dice_class0=0.5174.ckpt'
#     }
# }

# lits_ckpt_dict = {
#     'organ': {
#         'raunet': 'checkpoint_lits_organ_raunet_epoch=032-dice_class0=0.9627.ckpt',
#         'pspnet': 'checkpoint_lits_organ_pspnet_epoch=027-dice_class0=0.9603.ckpt',
#         'unet': 'checkpoint_lits_organ_unet_epoch=047-dice_class0=0.9549.ckpt',
#         'unet++': 'checkpoint_lits_organ_unet++_epoch=042-dice_class0=0.9546.ckpt',
#         'enet': 'checkpoint_lits_organ_enet_epoch=022-dice_class0=0.9603.ckpt',
#         'mobilenetv2': 'checkpoint_lits_organ_mobilenetv2_epoch=054-dice_class0=0.9496.ckpt',
#         'resnet18': 'checkpoint_lits_organ_resnet18_epoch=033-dice_class0=0.9451.ckpt'
#     },
#     'tumor': {
#         'raunet': 'checkpoint_lits_tumor_raunet_epoch=030-dice_class0=0.6136.ckpt',
#         'pspnet': 'checkpoint_lits_tumor_pspnet_epoch=034-dice_class0=0.6533.ckpt',
#         'unet': 'checkpoint_lits_tumor_unet_epoch=072-dice_class0=0.6081.ckpt',
#         'unet++': 'checkpoint_lits_tumor_unet++_epoch=081-dice_class0=0.6207.ckpt',
#         'enet': 'checkpoint_lits_tumor_enet_epoch=059-dice_class0=0.5769.ckpt',
#         'mobilenetv2': 'checkpoint_lits_tumor_mobilenetv2_epoch=051-dice_class0=0.5677.ckpt',
#         'resnet18': 'checkpoint_lits_tumor_resnet18_epoch=020-dice_class0=0.4767.ckpt'
#     }
# }

parser = argparse.ArgumentParser('train_kd')
parser.add_argument('--train_data_path', type=str, default='')
parser.add_argument('--test_data_path', type=str, default='')
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--tckpt', type=str, default='', help='teacher model checkpoint path')
parser.add_argument('--tmodel', type=str, default='')
parser.add_argument('--smodel', type=str, default='')
parser.add_argument('--dataset', type=str, default='kits', choices=['kits', 'lits'])
parser.add_argument('--task', type=str, default='', choices=['tumor', 'organ'])
parser.add_argument('--epochs', type=int, default=90)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gpu_id', default='6', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--kd_method', default='CrossEhcdAttKD', type=str, help='which KD method you want to use')
parser.add_argument('--resume', type=str, default='false', help='')

import pytorch_lightning as pl

class TimingCallback(pl.Callback):
    def __init__(self):
        super(TimingCallback, self).__init__()
        self.epoch_times = []  # 用于存储每轮训练的时间

    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()  # 记录每轮训练的开始时间

    def on_train_epoch_end(self, trainer, pl_module):
        end_time = time.time()  # 记录每轮训练的结束时间
        epoch_time = end_time - self.start_time
        self.epoch_times.append(epoch_time)
        print(f"Epoch {trainer.current_epoch + 1} took {epoch_time:.2f} seconds")

    def get_average_epoch_time(self):
        if len(self.epoch_times) > 0:
            return sum(self.epoch_times) / len(self.epoch_times)
        else:
            return 0.0



class KDPL(BasePLModel):
    def __init__(self, params):
        super(KDPL, self).__init__()
        self.save_hyperparameters(params)
        # load and freeze teacher net
        self.task = self.hparams.task
        tmodel = self.hparams.tmodel
        self.dataset = self.hparams.dataset
        self.hparams.train_data_path = 'data/' + self.dataset + '/' + self.task + '/slices'
        self.hparams.test_data_path = 'data/' + self.dataset + '/' + self.task + '/slices'
        self.hparams.checkpoint_path = 'data/' + self.dataset + '/' + self.task + '/checkpoints/'
        des_dict = None
        if self.dataset == 'kits':
            des_dict = kits_ckpt_dict
        elif self.dataset == 'lits':
            des_dict = lits_ckpt_dict
        self.hparams.tckpt = self.hparams.checkpoint_path + des_dict[self.task][tmodel]
        self.t_net = SegPL.load_from_checkpoint(checkpoint_path=self.hparams.tckpt)
        self.t_net.freeze()

        # student net
        # print(self.hparams.smodel)
        self.net = get_model(self.hparams.smodel, channels=2)

        # KD method
        self.method = self.hparams.kd_method

        self.support_image = None 
        self.support_mask = None

        def create_support_info():
            print('create support_info!')
            support_data_path = 'data/' + self.dataset + '/' + self.task + '/slices'
            data = SliceDataset(
                    data_path=support_data_path,
                    dataset=self.dataset,
                    task=self.task,
                    mode='support'
                )
            support_image, support_mask, case= next(iter(DataLoader(data, batch_size=64, num_workers=1, pin_memory=False)))
            support_image = support_image.unsqueeze(0)
            support_mask = support_mask.unsqueeze(0)
            # print(support_image.shape)
            # print(support_mask.shape)
            return support_image, support_mask
        if self.hparams.smodel.lower() == 'universeg':
            self.support_image, self.support_mask = create_support_info()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # 这里每个batch训练的输出都会被存储在outputs中，在training_epoch_end中可以被使用
        ct, mask, name = batch
        # self.t_net.eval()
        # t_out, t_low, t_high = self.t_net.net(ct)
        # output, low, high, = self.net(ct)
        #
        # loss_seg = calc_loss(output, mask)
        #
        # loss_pmd = prediction_map_distillation(output, t_out)
        # loss_imd = importance_maps_distillation(low, t_low) + importance_maps_distillation(high, t_high)
        # loss_rad = region_affinity_distillation(low, t_low, mask) + region_affinity_distillation(high, t_high, mask)
        #
        # loss = loss_seg + alpha * loss_pmd + beta1 * loss_imd + beta2 * loss_rad
        if self.method.lower() == 'crd':
            distiller = distiller_dict[self.method](self.net, self.t_net, self.train_set_num).cuda()
        else:
            distiller = distiller_dict[self.method](self.net, self.t_net)
        if self.hparams.smodel == 'universeg':
            flag = 'universeg'
            loss = distiller(batch, flag, self.support_image.cuda(), self.support_mask.cuda())
        else:
            flag = None
            loss = distiller(batch, flag, self.support_image, self.support_mask)
        # with open('test2.txt', 'a') as fo:
        #     fo.write('batch_idx:' + str(batch_idx))
        #     fo.write('\n')
        #     fo.write('batch:' + str(torch.sum(ct).detach().cpu().numpy())) # 相同,说明种子固定后数据读取顺序相同
        #     fo.write('\n')
        #     fo.write('loss :' + str(loss.detach().cpu().numpy()))
        #     fo.write('\n')
        
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        ct, mask, name = batch
        if self.hparams.smodel == 'universeg':
            output, low, high = self.net(ct, self.support_image.cuda(), self.support_mask.cuda())
        else:
            output, low, high = self.net(ct)
        # # 测试中间特征和对应的注意力图
        # att = F.normalize(low.pow(2).mean(1))
        # print('low top 3:')
        # print(low[:3])
        # print('low_att top 3:')
        # print(att[:3])
        # # 教师的中间特征
        # _, t_low, t_high = self.t_net(ct)
        # t_att = F.normalize(t_low.pow(2).mean(1))
        # print('t_low top 3:')
        # print(t_low[:3])
        # print('t_low_att top 3:')
        # print(t_att[:3])
        # assert 1==2

        self.measure(batch, output)

    def train_dataloader(self):
        # init_seed(3407)
        # g = torch.Generator()
        dataset = SliceDataset(
            data_path=self.hparams.train_data_path,
            dataset=self.hparams.dataset,
            task=self.hparams.task,
            mode='train'
        )
        self.train_set_num = len(dataset) # the num of train_set
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=32, pin_memory=True, shuffle=True)

    def test_dataloader(self):
        dataset = SliceDataset(
            data_path=self.hparams.test_data_path,
            dataset=self.hparams.dataset,
            task=self.hparams.task,
            mode='test'
        )
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=16, pin_memory=True)

    def val_dataloader(self):
        return self.test_dataloader()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999))
        scheduler = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs, eta_min=1e-6),
                     'interval': 'epoch',
                     'frequency': 1}
        return [opt], [scheduler]


def main():
    args = parser.parse_args()
    # 获取gpus
    gpu_list = [int(x) for x in args.gpu_id.split(',')]
    model = KDPL(args)
    save_path = 'data/' + args.dataset + '/' + args.task + '/checkpoints/'
    is_resume = False if args.resume == 'false' else True
    resume_ckpt_path = None
    if is_resume:
        resume_file_name = 'CrossEhcdAttKD_checkpoint_kits_organ_unet_kd_enet_epoch=072-dice_class0=0.9651.ckpt'
        resume_ckpt_path = os.path.join(save_path) + resume_file_name
        # tc = tcio()
        # ckpt_dict = tc.load_checkpoint(path=resume_ckpt_path)
        # print(ckpt_dict['epoch'])
        # print(ckpt_dict['hparams_name'])
        # print(ckpt_dict['hyper_parameters'])
        # print(ckpt_dict['state_dict'])
    
    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_path),
        filename='%s_checkpoint_%s_%s_%s_kd_%s_{epoch:03d}-{dice_class0:.4f}' % (args.kd_method, args.dataset, args.task, args.tmodel, args.smodel),
        save_top_k=1,
        mode='max',
        monitor='dice_class0',
        save_last=True
    )
    timing_callback = TimingCallback()

    logger = TensorBoardLogger('log', name='%s_%s_%s_%s_kd_%s' % (args.kd_method, args.dataset, args.task,args.tmodel, args.smodel))
    trainer = Trainer.from_argparse_args(args, max_epochs=args.epochs, gpus=gpu_list, callbacks=[checkpoint_callback, timing_callback], logger=logger)
    trainer.fit(model, ckpt_path = resume_ckpt_path)

    # 获取每轮训练的平均时间
    average_epoch_time = timing_callback.get_average_epoch_time()
    print(f"Average time per epoch: {average_epoch_time:.2f} seconds")
    with open('training_time.txt', 'a+') as fo:
        fo.write('teacher : ' + str(args.tmodel) + '\n')
        fo.write('student : ' + str(args.smodel) + '\n')
        fo.write('dataset : ' + str(args.dataset) + '\n')
        fo.write('task : ' + str(args.task) + '\n')
        fo.write('kd_method : ' + str(args.kd_method) + '\n')
        fo.write('Average time per epoch : ' + str(round(average_epoch_time, 2)) + ' seconds' + '\n')
        fo.write('\n')


'''
CrossEhcdAttKD_checkpoint_kits_tumor_unet_kd_enet_epoch=077-dice_class0=0.6981.ckpt
CrossEhcdAttKD_checkpoint_kits_organ_unet_kd_enet_epoch=071-dice_class0=0.9662.ckpt
CrossEhcdAttKD_checkpoint_lits_organ_unet_kd_enet_epoch=061-dice_class0=0.9633.ckpt
CrossEhcdAttKD_checkpoint_lits_tumor_unet_kd_enet_epoch=039-dice_class0=0.6031.ckpt
'''
def test():
    args = parser.parse_args()
    # 获取gpus
    gpu_list = [int(x) for x in args.gpu_id.split(',')]
    model = KDPL.load_from_checkpoint(checkpoint_path='/data/xulingbing/projects/EMKD/data/'+args.dataset+'/'+args.task+'/checkpoints/' + 
                                      'CrossEhcdAttKD_checkpoint_lits_tumor_unet_kd_enet_epoch=075-dice_class0=0.5944.ckpt')
    trainer = Trainer(gpus=gpu_list)
    trainer.test(model)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == 'train':
        main()
    if args.mode == 'test':
        test()