import torch
import torch.nn as nn

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv_block(x)

        out = out + identity
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):

    def __init__(self, in_channels=3, num_classes=2, dropout=0.25):
        super(ResNet, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride =2,
                padding=3,
                bias= False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage1 = nn.Sequential(
            ResidualBlock(64,64, stride=1),
            ResidualBlock(64,64,stride=1)
        )

        self.stage2 = nn.Sequential(
            ResidualBlock(64,128,stride=2),
            ResidualBlock(128,128,stride=1)
        )

        self.stage3 = nn.Sequential(
            ResidualBlock(128,256,stride=2),
            ResidualBlock(256,256,stride=1)
        )

        self.stage4 = nn.Sequential(
            ResidualBlock(256,512,stride=2),
            ResidualBlock(512,512,stride=1)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256,num_classes)
        )

    def forward(self, x):

        # 7x7 큰 필터로 이미지의 큼직한 특징을 한 번 훑고 시작    
        x=self.stem(x)

        x=self.stage1(x)
        x=self.stage2(x)
        x=self.stage3(x)
        x=self.stage4(x)

        x=self.global_pool(x)
        x=self.classifier(x)

        return x



    

