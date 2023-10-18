import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import torch.functional as F
import torch.nn as nn
import numpy as np
import requests
import torchvision
from PIL import Image
from gradcam.pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from gradcam.pytorch_grad_cam.grad_cam import GradCAM
import os
import copy
from networks.ENet import ENet
from train import SegPL
from train_kd import KDPL
import argparse
import utils.data_utils as du
import cv2

def main(kd, dataset, task, model_type, img_idx, npz_path_base):

    kd_pth_dict = {
        'kits': {
            'tumor': {
                '': 'checkpoint_kits_tumor_enet_epoch=046-dice_class0=0.5261.ckpt',
                'CrossEhcdAttKD': 'CrossEhcdAttKD_checkpoint_kits_tumor_unet_kd_enet_epoch=077-dice_class0=0.6981.ckpt',
                'AT': 'AT_checkpoint_kits_tumor_unet_kd_enet_epoch=045-dice_class0=0.6590.ckpt',
                'EMKD': 'EMKD_checkpoint_kits_tumor_unet_kd_enet_epoch=051-dice_class0=0.6628.ckpt',
                'RKD': 'RKD_checkpoint_kits_tumor_unet_kd_enet_epoch=067-dice_class0=0.6057.ckpt'
            },
            'organ': {
                '': 'checkpoint_kits_organ_enet_epoch=074-dice_class0=0.9654.ckpt',
                'CrossEhcdAttKD': 'CrossEhcdAttKD_checkpoint_kits_organ_unet_kd_enet_epoch=071-dice_class0=0.9662.ckpt',
                'AT': '',
                'EMKD': '',
                'RKD': ''
            }
            
        },
        'lits': {
            'tumor': {
                '': 'checkpoint_lits_tumor_enet_epoch=059-dice_class0=0.5769.ckpt',
                'CrossEhcdAttKD': 'CrossEhcdAttKD_checkpoint_lits_tumor_unet_kd_enet_epoch=039-dice_class0=0.6031.ckpt',
                'AT': 'AT_checkpoint_lits_tumor_unet_kd_enet_epoch=065-dice_class0=0.5994.ckpt',
                'EMKD': 'EMKD_checkpoint_lits_tumor_unet_kd_enet_epoch=017-dice_class0=0.5949.ckpt',
                'RKD': 'RKD_checkpoint_lits_tumor_unet_kd_enet_epoch=053-dice_class0=0.5909.ckpt'
            },
            'organ': {
                '': 'checkpoint_lits_organ_enet_epoch=022-dice_class0=0.9603.ckpt',
                'CrossEhcdAttKD': 'CrossEhcdAttKD_checkpoint_lits_organ_unet_kd_enet_epoch=061-dice_class0=0.9633.ckpt',
                'AT': '',
                'EMKD': '',
                'RKD': ''
            }
            
        }
        
    }

    # kd_type:如果是单一模型(无蒸馏)->'',蒸馏后的模型->对应蒸馏方法
    kd_type = kd
    dataset = dataset
    task = task
    # task_type = 'tumor'
    # img_path_base = img_path_base
    npz_path_base = npz_path_base
    img_index = img_idx
    # img_path = img_path_base + img_index
    npz_path = npz_path_base + img_index
    output_path = 'output_visual/heal/' + dataset + '/' + img_index
    model_type = model_type
    pth_base = 'data/' + dataset +'/' + task + '/checkpoints/'

    def read():
        npz = np.load(npz_path + '.npz', allow_pickle=True)
        ct = npz.get('ct') # 512,512
        mask = npz.get('mask') # 512,512

        # Preprocess
        if task == 'organ':
            mask[mask > 0] = 1
        elif task == 'tumor':
            mask = mask >> 1
            mask[mask > 0] = 1

        ct = du.window_standardize(ct, -200, 300)

        # one-hot img0背景类 mask目标类
        img0 = copy.deepcopy(mask)
        img0 += 1
        img0[img0 != 1] = 0
        mask = np.stack((img0, mask), axis=0)
        mask[mask > 0] = 1

        # To tensor & cut to 384
        ct = torch.from_numpy(du.cut_384(ct.copy())).unsqueeze(0).float() # 1,384,384
        mask = torch.from_numpy(du.cut_384(mask.copy())).float() # 2,384,384

        ct_img = np.uint8(ct.numpy()*255/ct.max()).transpose(1,2,0)
        mask_img = np.uint8(mask[1,:].unsqueeze(0).numpy()*255/mask[1,:].max()).transpose(1,2,0)
        cv2.imwrite(output_path+'_img.jpg', ct_img)
        cv2.imwrite(output_path+'_mask_'+task+'.png', mask_img)
        return ct, mask

    def cut_384(img):
        """
        cut a 512*512 ct img to 384*384
        :param img:
        :return:
        """
        if len(img.shape) > 2:
            ret = img[:, 50:434, 60:444]
        else:
            ret = img[50:434, 60:444]
        return ret

    # 读入自己的图像
    ct, mask = read() # 预处理之后的结果
    ct = ct.unsqueeze(0) # 1,1,384,384
    # 用于重叠的原图
    image = np.array(Image.open(output_path + '_img.jpg'))
    image = np.float32(image) / 255
    image = torch.unsqueeze(torch.from_numpy(image), 0)
    image = image.permute(1,2,0)
    image = image.numpy()
    
    # 读入自己的模型并且加载训练好的权重
    if kd_type != '':
        model = KDPL.load_from_checkpoint(pth_base + kd_pth_dict[dataset][task][kd_type])
    else:
        model = SegPL.load_from_checkpoint(checkpoint_path=pth_base + kd_pth_dict[dataset][task][kd_type])
    model.cuda()
    model = model.eval()

    if torch.cuda.is_available(): # true
        model = model.cuda()
        ct = ct.cuda()
    
    # 推理
    if kd_type != '':
        output, _, _ = model(ct)
    else:
        output, _, _ = model(ct)
    normalized_masks = torch.softmax(output, dim=1).cpu()
    
    # 自己的数据集的类别
    sem_classes = [
        '__background__', task
    ]
    
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
    round_category = sem_class_to_idx[task]
    round_mask = torch.argmax(normalized_masks[0], dim=0).detach().cpu().numpy()
    round_mask_uint8 = 255 * np.uint8(round_mask == round_category)
    round_mask_float = np.float32(round_mask == round_category)
    
    
    class SemanticSegmentationTarget:
        def __init__(self, category, mask):
            self.category = category
            self.mask = torch.from_numpy(mask)
            if torch.cuda.is_available():
                self.mask = self.mask.cuda()
    
        def __call__(self, model_output):
            return (model_output[self.category, :, :] * self.mask).sum()
    
    # # 自己要放CAM的位置
    target_layers = [model.net.bottleneck3_8.act]
    # target_layers = [model.net.bottleneck4_1]

    # target_layers = [model.net.initial.act]
    targets = [SemanticSegmentationTarget(round_category, round_mask_float)]
    
    with GradCAM(model=model, target_layers=target_layers,
                use_cuda=torch.cuda.is_available(), kd_type=kd_type) as cam:
        grayscale_cam = cam(input_tensor=ct,
                            targets=targets)[0, :]
        # print(grayscale_cam.shape)
        cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    
    # 保存CAM的结果
    img = Image.fromarray(cam_image)
    # img.show()
    # img.save(output_path+ '_' + task + '_' + model_type + '_heat_' + kd_type + '.png')
    img.save(output_path + '_heat_' + kd_type + '.png')

if __name__ == '__main__':
    kd_list = ['', 'CrossEhcdAttKD','EMKD','AT','RKD']
    dataset = 'lits'
    task = 'tumor'
    model_type = 'enet'
    img_idx = []
    img_path = 'data_example/temp'
    # img_path_base = 'data_example/temp/'
    npz_path_base = 'data/' + dataset + '/' + task + '/slices/'
    import os
    files = os.listdir(img_path)
    files = [f.split('_img')[0] for f in files if f.find('img') !=- 1]
    
    for f in files:
        for kd in kd_list:
            main(kd, dataset, task, model_type, f, npz_path_base)