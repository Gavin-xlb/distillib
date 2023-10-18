import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import copy
import torch

def generate(dataset, task, case):
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

    loadData1 = np.load('/data/xulingbing/projects/distillib/data/'+dataset+'/'+task+'/slices/'+dataset+'_'+task+'_slices.npy')
    # print(loadData1)
    l = []
    for i in loadData1:
        if str(i).startswith(str(case) + '_'):
            l.append(str(i))
    print(l)
    path = '/data/xulingbing/projects/distillib/data/'+dataset+'/'+task+'/slices/'
    for index in l:
        mask_type = 'organ'
        loadData = np.load(path + index)
        ct = loadData['ct']
        mask = loadData['mask']
        #  organ
        if mask_type == 'organ':
            mask[mask > 0] = 1
        #  tumor
        if mask_type == 'tumor':
            mask = mask >> 1
            mask[mask > 0] = 1
        ct = np.clip(ct, -200, 300)
        # x=x*2-1: map x to [-1,1]
        ct = 2 * (ct + 200) / 500 - 1

        # one-hot
        img0 = copy.deepcopy(mask)
        img0 += 1
        img0[img0 != 1] = 0
        mask = np.stack((img0, mask), axis=0)
        mask[mask > 0] = 1
        # To tensor & cut to 384
        ct = torch.from_numpy(cut_384(ct.copy())).unsqueeze(0).float()

        mask = torch.from_numpy(cut_384(mask.copy())).float()

        ct = np.uint8(ct.numpy()*255/ct.max()).transpose(1,2,0)
        mask = np.uint8(mask[1,:].unsqueeze(0).numpy()*255/mask[1,:].max()).transpose(1,2,0)
        cv2.imwrite('temp/'+index.split('.')[0]+'_img.jpg', ct)
        # cv2.imwrite('temp/'+index.split('.')[0]+'_mask_'+mask_type+'.jpg', mask) 
    # assert 1==2
    # path = '/data/xulingbing/projects/distillib/data/lits/tumor/slices/'
    # index = '1_68'
    # mask_type = 'tumor'
    # loadData = np.load(path + index+'.npz')

    # ct = loadData['ct']
    # mask = loadData['mask']

    # #  organ
    # if mask_type == 'organ':
    #     mask[mask > 0] = 1
    # #  tumor
    # if mask_type == 'tumor':
    #     mask = mask >> 1
    #     mask[mask > 0] = 1

    # ct = np.clip(ct, -200, 300)
    # # x=x*2-1: map x to [-1,1]
    # ct = 2 * (ct + 200) / 500 - 1

    # # one-hot
    # img0 = copy.deepcopy(mask)
    # img0 += 1
    # img0[img0 != 1] = 0
    # mask = np.stack((img0, mask), axis=0)
    # mask[mask > 0] = 1
    # # To tensor & cut to 384
    # ct = torch.from_numpy(cut_384(ct.copy())).unsqueeze(0).float()

    # mask = torch.from_numpy(cut_384(mask.copy())).float()

    # ct = np.uint8(ct.numpy()*255/ct.max()).transpose(1,2,0)
    # mask = np.uint8(mask[1,:].unsqueeze(0).numpy()*255/mask[1,:].max()).transpose(1,2,0)
    # cv2.imwrite(index+'_img.jpg', ct)
    # cv2.imwrite(index+'_mask_'+mask_type+'.jpg', mask)

if __name__ == '__main__':
    # kits
    # case = ['0'*(5-len(str(i)))+str(i) for i in range(80, 85)]
    # lits
    case = [i for i in range(104,105)]
    dataset = 'lits'
    task = 'tumor'
    for c in case:
        generate(dataset, task, c)