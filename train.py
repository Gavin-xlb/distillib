import os
import torch
import argparse
from networks import get_model
from utils.base_pl_model import BasePLModel
from datasets.midataset import SliceDataset
from utils.loss_functions import calc_loss
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
# from pytorch_lightning.utilities import seed
# from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np, random


def init_seed(seed):
    # defining random initial seeds
    print('######defining random initial seeds!######')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed = 3407
# init_seed(seed)

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
        'resnet18': 'checkpoint_kits_tumor_resnet18_epoch=059-dice_class0=0.5068.ckpt'
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

parser = argparse.ArgumentParser('train')
parser.add_argument('--train_data_path', type=str, default='')
parser.add_argument('--test_data_path', type=str, default='')
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--model', type=str, default='')
parser.add_argument('--dataset', type=str, default='kits', choices=['kits', 'lits'])
parser.add_argument('--task', type=str, default='', choices=['tumor', 'organ'])
parser.add_argument('--epochs', type=int, default=90)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gpu_id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')


# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=22, stdoutToServer=True, stderrToServer=True)
class SegPL(BasePLModel):
    def __init__(self, params):
        super(SegPL, self).__init__()
        self.save_hyperparameters(params)
        self.task = self.hparams.task
        self.model = self.hparams.model
        self.dataset = self.hparams.dataset
        self.hparams.train_data_path = 'data/' + self.dataset + '/' + self.task + '/slices'
        self.hparams.test_data_path = 'data/' + self.dataset + '/' + self.task + '/slices'
        self.hparams.checkpoint_path = 'data/' + self.dataset + '/' + self.task + '/checkpoints/'
        # self.hparams.tckpt = self.hparams.checkpoint_path + kits_ckpt_dict[task][model]
        self.net = get_model(self.hparams.model, channels=2)

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
            print(support_image.shape)
            print(support_mask.shape)
            return support_image, support_mask
        if self.model.lower() == 'universeg':
            self.support_image, self.support_mask = create_support_info()

    def forward_universeg(self, x):
        output, low, high = self.net(x, self.support_image.cuda(), self.support_mask.cuda())
        return output, low, high

    def forward(self, x):
        output, low, high = self.net(x)
        return output, low, high

    def training_step(self, batch, batch_idx):
        ct, mask, name = batch
        if self.model.lower() == 'universeg':
            output = self.forward_universeg(ct)[0]
        else:
            output = self.forward(ct)[0]  # 输出通道数为2，代表背景类和器官/肿瘤类
        loss = calc_loss(output, mask)  # Dice_loss Used

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        ct, mask, name = batch
        if self.model.lower() == 'universeg':
            output = self.forward_universeg(ct)[0]
        else:
            output = self.forward(ct)[0]

        self.measure(batch, output)

    def train_dataloader(self):
        dataset = SliceDataset(
            data_path=self.hparams.train_data_path,
            dataset=self.hparams.dataset,
            task=self.hparams.task,
            mode='train'
        )
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=32, pin_memory=False, shuffle=True)

    def test_dataloader(self):
        dataset = SliceDataset(
            data_path=self.hparams.test_data_path,
            dataset=self.hparams.dataset,
            task=self.hparams.task,
            mode='test'
        )
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=16, pin_memory=False)

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
    model = SegPL(args)
    # 获取gpus
    gpu_list = [int(x) for x in args.gpu_id.split(',')]
    # checkpoint
    save_path = 'data/' + args.dataset + '/' + args.task + '/checkpoints/'
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_path),
        filename='checkpoint_%s_%s_%s_{epoch:03d}-{dice_class0:.4f}' % (args.dataset, args.task, args.model),
        save_last=True,
        save_top_k=1,
        mode='max',
        monitor='dice_class0'
    )

    logger = TensorBoardLogger('log', name='%s_%s_%s' % (args.dataset, args.task, args.model))
    trainer = Trainer.from_argparse_args(args, max_epochs=args.epochs, gpus=gpu_list, callbacks=checkpoint_callback, logger=logger)
    trainer.fit(model)


def test():
    args = parser.parse_args()
    # 获取gpus
    gpu_list = [int(x) for x in args.gpu_id.split(',')]
    # model = SegPL.load_from_checkpoint(checkpoint_path=os.path.join(args.checkpoint_path, 'last.ckpt'))
    if args.dataset == 'kits':
        des_dict = kits_ckpt_dict
    elif args.dataset == 'lits':
        des_dict = lits_ckpt_dict
    model = SegPL.load_from_checkpoint('/data/xulingbing/projects/distillib/data/' + args.dataset + '/' + args.task +'/checkpoints/' + des_dict[args.task][args.model])
    trainer = Trainer(gpus=gpu_list)
    trainer.test(model)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == 'train':
        main()
    if args.mode == 'test':
        test()
    
    # # test for universeg
    # def create_support_info():
    #         print('create support_info!')
    #         support_data_path = 'data/' + args.dataset + '/' + args.task + '/slices'
    #         data = SliceDataset(
    #                 data_path=support_data_path,
    #                 dataset=args.dataset,
    #                 task=args.task,
    #                 mode='support'
    #             )
    #         support_image, support_mask, case= next(iter(DataLoader(data, batch_size=64, num_workers=1, pin_memory=False)))
    #         support_image = support_image.unsqueeze(0)
    #         support_mask = support_mask[:,1:,:,:].unsqueeze(0)
    #         print(support_image.shape)
    #         print(support_mask.shape)
    #         return support_image, support_mask
    # if args.model.lower() == 'universeg':
    #     support_image, support_mask = create_support_info()
    # model = get_model(args.model, channels=2)
    # input = torch.randn((1, 1, 384,384))
    # output = model(input, support_image, support_mask)
    # print(output)
