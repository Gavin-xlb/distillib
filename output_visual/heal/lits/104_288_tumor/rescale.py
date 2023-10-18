import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import torch.functional as F
import numpy as np
from PIL import Image
import cv2

width = 192
x = 100
y = 20
delta = 192
img_idx = '104_288'
task = 'tumor'
# img_path_list = [img_idx + '_'+ task + '_mask' + '.png', img_idx + '_'+ task + '_unet_.png', img_idx + '_'+ task + '_enet_.png', img_idx + '_'+ task + '_enet_CrossEhcdAttKD.png', img_idx + '_img.jpg']
# img_path_list = [img_idx + '_'+ task + '_mask' + '.png', img_idx + '_'+ task + '_enet_.png', img_idx + '_'+ task + '_enet_CrossEhcdAttKD.png', img_idx + '_img.jpg', img_idx + '_'+ task + '_enet_AT.png', img_idx + '_'+ task + '_enet_EMKD.png', img_idx + '_'+ task + '_enet_RKD.png']
# heal
img_path_list = [img_idx + '_mask_tumor' + '.png', img_idx + '_heat_.png', img_idx + '_heat_CrossEhcdAttKD.png', img_idx + '_img.jpg', img_idx + '_heat_AT.png', img_idx + '_heat_EMKD.png', img_idx + '_heat_RKD.png']
for img_path in img_path_list:
    origin_image = cv2.imread(img_path)
    image = np.array(origin_image)
    crop = image[y:min(y+delta,383), x:min(x+delta,383)]
    # crop = cv2.resize(crop, (width, width))
    cv2.imwrite(img_path.split('.')[0] + '_scale.png', crop)
    # crop.save(img_path + '_scale.png')