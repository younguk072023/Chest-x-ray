import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = in_channels,
                kernel_size = 3,
                stride=stride,
                padding= 1,
                groups = in_channels,
                bias = False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        #채널 합치기
        self.pointwise = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels = out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x
    
class MobileNet(nn.Module):

    def __init__(self, in_channels=3, num_classes=2, dropout=0.25):
        super(MobileNet, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.features = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=1),       # 112 -> 112
            DepthwiseSeparableConv(64, 128, stride=2),      # 112 -> 56
            DepthwiseSeparableConv(128, 128, stride=1),     # 56 -> 56

            DepthwiseSeparableConv(128, 256, stride=2),     # 56 -> 28
            DepthwiseSeparableConv(256, 256, stride=1),     # 28 -> 28

            DepthwiseSeparableConv(256, 512, stride=2),     # 28 -> 14
            DepthwiseSeparableConv(512, 512, stride=1),     # 14 -> 14
            DepthwiseSeparableConv(512, 512, stride=1),     # 14 -> 14

            DepthwiseSeparableConv(512, 1024, stride=2),    # 14 -> 7
            DepthwiseSeparableConv(1024, 1024, stride=1)    # 7 -> 7
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)

        x = self.global_pool(x)
        x = self.classifier(x)

        return x