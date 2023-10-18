import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import torch.functional as F
import numpy as np
from PIL import Image
import copy
from networks.ENet import ENet
from train import SegPL
from train_kd import KDPL
import utils.data_utils as du
import cv2
from data_example.generator_img import generate

def main(kd, dataset, task, model_type, img_idx, v, img_path_base, npz_path_base):
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

    '''

    '''
    # kd_type:如果是单一模型(无蒸馏)->'',蒸馏后的模型->对应蒸馏方法 CrossEhcdAttKD
    kd_type = kd
    dataset = dataset
    task = task
    '''
    # CrossEhcdAttKD
    CrossEhcdAttKD_checkpoint_kits_tumor_unet_kd_enet_epoch=077-dice_class0=0.6981.ckpt
    CrossEhcdAttKD_checkpoint_kits_organ_unet_kd_enet_epoch=071-dice_class0=0.9662.ckpt
    CrossEhcdAttKD_checkpoint_lits_organ_unet_kd_enet_epoch=061-dice_class0=0.9633.ckpt
    CrossEhcdAttKD_checkpoint_lits_tumor_unet_kd_enet_epoch=039-dice_class0=0.6031.ckpt
    '''
    kd_dict = {
        'kits': {
            '': '',
            'CrossEhcdAttKD': 'CrossEhcdAttKD_checkpoint_kits_tumor_unet_kd_enet_epoch=077-dice_class0=0.6981.ckpt',
            'AT': 'AT_checkpoint_kits_tumor_unet_kd_enet_epoch=045-dice_class0=0.6590.ckpt',
            'EMKD': 'EMKD_checkpoint_kits_tumor_unet_kd_enet_epoch=051-dice_class0=0.6628.ckpt',
            'RKD': 'RKD_checkpoint_kits_tumor_unet_kd_enet_epoch=067-dice_class0=0.6057.ckpt'
        },
        'lits': {
            '': '',
            'CrossEhcdAttKD': 'CrossEhcdAttKD_checkpoint_lits_tumor_unet_kd_enet_epoch=039-dice_class0=0.6031.ckpt',
            'AT': 'AT_checkpoint_lits_tumor_unet_kd_enet_epoch=065-dice_class0=0.5994.ckpt',
            'EMKD': 'EMKD_checkpoint_lits_tumor_unet_kd_enet_epoch=017-dice_class0=0.5949.ckpt',
            'RKD': 'RKD_checkpoint_lits_tumor_unet_kd_enet_epoch=053-dice_class0=0.5909.ckpt'
        }
    }
    save_pth_base = '/data/xulingbing/projects/EMKD/data/' + dataset + '/' + task + '/checkpoints/'
    # kd_pth_path = 'EMKD_checkpoint_lits_tumor_unet_kd_enet_epoch=017-dice_class0=0.5949.ckpt'
    kd_pth_path = kd_dict[dataset][kd_type]
    pth_choice = kits_ckpt_dict if dataset == 'kits' else lits_ckpt_dict
    model_type = model_type
    single_model_pth = pth_choice[task][model_type]
    img_path_base = img_path_base
    # 修改编号
    img_index = img_idx
    npz_path = npz_path_base + img_index
    img_path = img_path_base + img_idx
    # 单一模型
    # single_model = 'enet'
    output_path_base = 'output_visual/prediction/' + dataset + '/'
    visual = v # mask:1;output:0


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
        if dataset == 'lits':
            ct = du.window_standardize(ct, -60, 140)
        elif dataset == 'kits':
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
        return ct, mask

    # 读入自己的图像
    ct, mask = read() # 预处理之后的结果
    ct = ct.unsqueeze(0) # 1,1,384,384
    # 用于重叠的原图
    origin_image = Image.open(img_path + '_img.jpg')
    image = np.array(origin_image)
    
    # 读入自己的模型并且加载训练好的权重
    # 无蒸馏
    # save_pth = '/data/xulingbing/projects/EMKD/data/kits/tumor/checkpoints/checkpoint_kits_tumor_enet_epoch=043-dice_class0=0.5086.ckpt'
    # AT
    # save_pth = '/data/xulingbing/projects/EMKD/data/kits/tumor/checkpoints/AT_checkpoint_kits_tumor_raunet_kd_enet_epoch=053-dice_class0=0.7069.ckpt'
    # ReviewEhcdAttKD


    if kd_type != '': # 蒸馏
        model = KDPL.load_from_checkpoint(save_pth_base + kd_pth_path)
    else: # 单一模型
        model = SegPL.load_from_checkpoint(checkpoint_path='data/' + dataset + '/' + task + '/checkpoints/' + single_model_pth)
    model.cuda()
    model = model.eval()

    if torch.cuda.is_available(): # true
        model = model.cuda()
        ct = ct.cuda()
    
    # 推理
    if kd_type != '':
        output, _, _ = model(ct)
    else:
        output, _, _ = model(ct) # 1,2,384,384
    output = torch.softmax(output, dim=1)[:, 1:].contiguous()
    mask = mask[1:, :].contiguous() # 1,384,384
    output = (output > 0.4).float().squeeze().detach().cpu().numpy() # 384,384
    output = np.clip(output, 0, 1)
    mask = mask.float().squeeze().detach().cpu().numpy() # 384,384
    mask = np.clip(mask, 0, 1)
    seg_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    seg_image[..., :3] = 0
    # 将掩码叠加到原始图像上，映射为透明蓝色
    blue_color = [27, 119, 213, 150]
    yellow_color = [252, 230, 35, 255]
    purple_color = [67, 1, 83, 255]
    # 如果可视化gt——mask>0;可视化output——output>0
    if visual:
        vo = mask
    else:
        vo = output
    # 特征可视化时生成mask在原图上的位置图时使用的颜色
    seg_image[vo > 0] = blue_color
    # 批量生成mask时使用的颜色
    # seg_image[vo > 0] = yellow_color
    # seg_image[vo == 0] = purple_color

    seg_image = Image.fromarray(seg_image, 'RGBA')
    result_image = Image.alpha_composite(origin_image.convert('RGBA'), seg_image)
    
    if visual:
        path = output_path_base + img_index+ '_' + task + '_mask' + '.png'
    else:
        path = output_path_base + img_index+ '_' + task + '_' + model_type + '_' + kd_type + '.png'
    result_image.save(path)

if __name__ == '__main__':
    kd_list = ['', 'CrossEhcdAttKD','EMKD','AT','RKD']
    dataset = 'lits'
    task = 'tumor'
    model_type = 'enet'
    img_idx = []
    img_path = 'data_example/temp'
    visual = [0, 1] # mask:1;output:0
    img_path_base = 'data_example/temp/'
    npz_path_base = 'data/' + dataset + '/' + task + '/slices/'
    import os
    # 批量生成mask
    # files = os.listdir(img_path)
    # files = [f.split('_img')[0] for f in files if f.find('img') !=- 1]
    # for f in files:
    #     for v in visual:
    #         if v == 1:
    #             main(kd, dataset, task, model_type, f, v, img_path_base, npz_path_base)
    #         else:
    #             for kd in kd_list:
    #                 main(kd, dataset, task, model_type, f, v, img_path_base, npz_path_base)
    
    # 特征可视化时生成mask在原图上的位置图
    main('', dataset, task, model_type, '104_288', 1, img_path_base, npz_path_base)
    