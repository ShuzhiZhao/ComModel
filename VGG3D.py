import torch.nn as nn
import torch


class VGG3D(nn.Module):

    def __init__(self, num_classes):
        super(VGG3D, self).__init__()
        self.num_classes = num_classes

        self.Conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=116,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            nn.MaxPool3d(
                kernel_size=3,
                stride=1),
        )
        self.Conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            nn.MaxPool3d(
                kernel_size=2,
                stride=1),
        )
        self.Conv3 = nn.Sequential(
            nn.Conv3d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=2),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=2),
            nn.ReLU(),
            nn.MaxPool3d(
                kernel_size=2,
                stride=2),
        )
        self.Conv4 = nn.Sequential(
            nn.Conv3d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            nn.MaxPool3d(
                kernel_size=2,
                stride=1),
        )

        self.Conv5 = nn.Sequential(
            nn.Conv3d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            nn.MaxPool3d(
                kernel_size=2,
                stride=1),
        )


    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
#         x = self.Conv4(x)
#         x = self.Conv5(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()

