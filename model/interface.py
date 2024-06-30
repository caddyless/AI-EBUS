import torch

from operator import eq
from utils import convert_dict_yaml
from pathlib import Path

import model.network.backbone.unetwork as unet
from model.typical import TypicalNet
from model.seg import MixNet
from model.classify import MMVideoNet
from model.contrastive import ContrastMMVideo
from model.network.backbone.senet import se_resnet50, senet154
from model.network.backbone.pnasnet import pnasnet5large
from model.network.backbone.nasnet_mobile import nasnetamobile
from model.network.backbone.inceptionv4 import inceptionv4
from model.network.backbone.xception import xception
from model.network.backbone.attention import attention56
from model.network.backbone.densenet import densenet121
from model.network.backbone.mobilenetv2 import mobilenetv2
from model.network.backbone.resnet import resnet18, resnet34, resnet50
from model.network.backbone.shufflenetv2 import shufflenetv2
from model.network.backbone.squeezenet import squeezenet
from model.network.backbone.vgg import vgg11_bn, vgg13_bn, vgg16_bn
from model.network.backbone.seincept import seincept
from model.network.backbone.mtan import MultiAttention, CentralNet
from model.network.backbone.bcnn import bcnnet
from model.network.backbone.attentionnet import attnet
from model.network.backbone.autoencoder import AutoEncoder
from model.network.aggregators import NeXtVLAD, NetVLAD
from model.network.multimodal import WholeModel
from model.network.backbone.msnet import ms_u_net, ms_att_u_net


backbone_dict = dict(
                     # cosinenet=backbone.cosinenet,
                     # simplenet2=backbone.simplenet2,
                     # simplenet3=backbone.simplenet3,
                     # graynet=backbone.graynet,
                     # bloodnet=backbone.bloodnet,
                     # inception=backbone.inception,
                     # unionnet=backbone.unionnet,
                     # auxnet=backbone.auxiliarynet,
                     unet=unet.unet,
                     r2attunet=unet.r2attunet,
                     r2unet=unet.r2unet,
                     attunet=unet.attunet,
                     reconstruct=unet.reconstruct,
                     # cascadenet=unet.cascadenet,
                     seresnet=se_resnet50,
                     pnas=pnasnet5large,
                     nasnet=nasnetamobile,
                     inceptionv4=inceptionv4,
                     xception=xception,
                     senet=senet154,
                     attention=attention56,
                     mobilenet=mobilenetv2,
                     densenet=densenet121,
                     resnet18=resnet18,
                     resnet34=resnet34,
                     resnet50=resnet50,
                     shufflenet=shufflenetv2,
                     squeezenet=squeezenet,
                     vgg11=vgg11_bn,
                     vgg13=vgg13_bn,
                     vgg16=vgg16_bn,
                     seincept=seincept,
                     mtan=MultiAttention,
                     bcnn=bcnnet,
                     attnet=attnet,
                     cennet=CentralNet,
                     auto=AutoEncoder,
                     nxvlad=NeXtVLAD,
                     nvlad=NetVLAD,
                     wholemodel=WholeModel,
                     msunet=ms_u_net,
                     msattunet=ms_att_u_net)

net_dict = dict(typical=TypicalNet,
                mixnet=MixNet,
                union=MMVideoNet,
                unsup=ContrastMMVideo)


def obtain_net(net, model, model_folder=None, **kwargs):
    assert model in backbone_dict.keys(), print('unknow backbone class!')
    assert net in net_dict.keys(), print('unknow net class!')

    if model_folder:
        model_folder = Path(model_folder)
        param_path = str(model_folder / 'model' / 'best-epoch.pth')
        pretrained_param = torch.load(param_path)['param']

        param = {'model_func': backbone_dict[model], **kwargs}
        print(param)
        net = net_dict[net](**param)

        if hasattr(net.model, 'module'):
            module = net.model.module
        else:
            module = net.model
        # init_param = module.state_dict()

        # new_param = {}

        # for k in init_param.keys():
        #     new_k = k.replace('C', 'B')
        #     if new_k in param.keys():
        #         new_param[k] = param[new_k]
        #     else:
        #         print('key %s not found in saved params' % k)
        #         new_param[k] = init_param[k]
        # print(pretrained_param.keys())
        # print(module.state_dict().keys())
        msg = module.load_state_dict(eliminate_mismatch(module, pretrained_param), strict=False)
        print(msg)

        print('Load pre-trained model params')

    else:
        param = {'model_func': backbone_dict[model], **kwargs}
        print(param)
        net = net_dict[net](**param)
        print('Randomly initialized model')
    return net


def eliminate_mismatch(model, params):
    state_dict = model.state_dict()

    for k, v in state_dict.items():
        if k in params:
            if not eq(v.size(), params[k].size()):
                del params[k]
    
    return params


# def load_model(model_folder: str, load_params: bool = True, net_param=None):
#     model_folder = Path(model_folder)
#     param_path = str(model_folder / 'model' / 'best-epoch.pth')
#     param = torch.load(param_path)['param']

#     if net_param is None:
#         net_param_path = str(model_folder / 'model' / 'net_config.yaml')
#         net_param = convert_dict_yaml(yaml_file=net_param_path)
#     net = obtain_net(**net_param)
#     if load_params:
#         if hasattr(net.model, 'module'):
#             module = net.model.module
#         else:
#             module = net.model
#         init_param = module.state_dict()

#         new_param = {}

#         for k in init_param.keys():
#             new_k = k.replace('C', 'B')
#             if new_k in param.keys():
#                 new_param[k] = param[new_k]
#             else:
#                 print('key %s not found in saved params' % k)
#                 new_param[k] = init_param[k]

#         module.load_state_dict(new_param)

#     return net
