# import torch
# from torch import nn
# import torchvision
# from torch.nn import functional as F
# from efficientnet_pytorch import EfficientNet
#
#
#
#
# class Detector(nn.Module):
#
#     def __init__(self):
#         super(Detector, self).__init__()
#         self.net=EfficientNet.from_pretrained("efficientnet-b4",advprop=True,num_classes=2)
#
#
#     def forward(self,x):
#         x=self.net(x)
#         return x
#
#
import torch
from torch import nn
import torchvision
from torch.nn import functional as F


class Detector(nn.Module):

    def __init__(self):
        super(Detector, self).__init__()
        self.net = torchvision.models.resnet50(pretrained=True)

    def forward(self, x):
        x = self.net(x)
        return x
