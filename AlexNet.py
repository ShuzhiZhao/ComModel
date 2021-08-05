## Code from https://github.com/Lornatang/AlexNet-PyTorch/blob/master/alexnet_pytorch/model.py
## ImageNet Classification with Deep Convolutional Neural Networks

import torch
import torch.nn as nn

class AlexNet(nn.Module):

    def __init__(self, global_params=None):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(228, 64, kernel_size=5, stride=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=4, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

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
