import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from graphs.models.deeplab_multi import DeeplabMulti
from modeling.deeplab import *
def get_model(args):
    if args.backbone == "deeplabv2_multi":
        model = DeeplabMulti(num_classes=args.num_classes,
                            pretrained=args.imagenet_pretrained)
        params = model.optim_parameters(args)
        args.numpy_transform = True
    elif args.backbone == "deeplabv3_plus":
        model = DeepLab(num_classes=args.num_classes,
        backbone='resnet',
        output_stride=16,
        sync_bn = False, 
        freeze_bn=False)
        params = model.optim_parameters(args)
        args.numpy_transform = True

    return model, params
