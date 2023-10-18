import sys
sys.path.append('..')
from .deeplabv3_plus import DeepLabV3Plus
from .ENet import ENet
from .ERFNet import ERFNet
from .ESPNet import ESPNet
from .mobilenetv2 import MobileNetV2
from .NestedUNet import NestedUNet
from .RAUNet import RAUNet
from .resnet18 import Resnet18
from .UNet import U_Net
from .PspNet.pspnet import PSPNet
from .MNet.MNet import MNet
from .resnet34 import Resnet34
from .resnet50 import Resnet50
from .universeg.UniverSeg import universeg
from .nnunet.network_architecture.UNet2022 import unet2022

def get_model(model_name: str, channels: int):
    assert model_name.lower() in ['deeplabv3+', 'enet', 'erfnet', 'espnet', 'mobilenetv2', 'unet++', 'raunet', 'resnet18', 'unet', 'pspnet', 'mnet', 'resnet34', 'resnet50', 'universeg', 'unet2022']
    if model_name.lower() == 'deeplabv3+':
        model = DeepLabV3Plus(num_class=channels)
    elif model_name.lower() == 'unet':
        model = U_Net(in_ch=1, out_ch=channels)
    elif model_name.lower() == 'resnet18':
        model = Resnet18(num_classes=channels)
    elif model_name.lower() == 'raunet':
        model = RAUNet(num_classes=channels)
    elif model_name.lower() == 'pspnet':
        model = PSPNet(num_classes=2)
    elif model_name.lower() == 'mobilenetv2':
        model = MobileNetV2(num_classes=channels)
    elif model_name.lower() == 'espnet':
        model = ESPNet(classes=channels)
    elif model_name.lower() == 'erfnet':
        model = ERFNet(num_classes=channels)
    elif model_name.lower() == 'enet':
        model = ENet(nclass=channels)
    elif model_name.lower() == 'unet++':
        model = NestedUNet(in_ch=1, out_ch=channels)
    elif model_name.lower() == 'mnet':
        model = MNet(in_channels=1, num_classes=channels)
    elif model_name.lower() == 'resnet34':
        model = Resnet34(num_classes=channels)
    elif model_name.lower() == 'resnet50':
        model = Resnet50(num_classes=channels)
    elif model_name.lower() == 'universeg':
        model = universeg(num_classes=channels, pretrained=False)
    elif model_name.lower() == 'unet2022':
        model = unet2022(num_classes=channels)
    return model