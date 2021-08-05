## Code from https://github.com/Lornatang/AlexNet-PyTorch/blob/master/alexnet_pytorch/model.py
## ImageNet Classification with Deep Convolutional Neural Networks

import torch
import torch.nn as nn

class AlexNet3D1(nn.Module):

    def __init__(self, global_params=None):
        super(AlexNet3D1, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(68, 64, kernel_size=3, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool3d(3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def extract_features(self, inputs):
        x = self.features(inputs)
        return x

    def forward(self, inputs):
        # See note [TorchScript super()]
        x = self.features(inputs)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
