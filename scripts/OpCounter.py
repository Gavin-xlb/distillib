"""
Count the MACs/FLOPs/parameters of PyTorch models.
Installation: pip install flopco-pytorch
"""
import sys
sys.path.append('..')
from flopco import FlopCo
from torchvision.models import resnet50
from networks.ENet import ENet
from networks.ERFNet import ERFNet
from networks.ESPNet import ESPNet
from networks.UNet import U_Net
from networks.RAUNet import RAUNet
from networks.NestedUNet import NestedUNet
from networks.mobilenetv2 import MobileNetV2
from networks.resnet18 import Resnet18
from networks.resnet34 import Resnet34
from networks.deeplabv3_plus import DeepLabV3Plus
from networks.PspNet.pspnet import PSPNet
from networks.universeg.UniverSeg import universeg
from networks.nnunet.network_architecture.UNet2022 import unet2022
import torch

channels = 2
device = 'cuda'
# model = PSPNet(num_classes=2).to(device)
# model = ENet(nclass=2).to(device)
# model = Resnet34(num_classes=2).to(device)
# model = universeg(num_classes=2).to(device)
# model = ERFNet(num_classes=2).to(device)
model = unet2022(num_classes=2).to(device)
# Estimate model statistics by making one forward pass througth the model,
# for the input image of size 3 x 224 x 224

stats = FlopCo(model, img_size = (1, 1, 384, 384), device = device)

print(stats.total_flops/1e9, stats.total_params/1e6)