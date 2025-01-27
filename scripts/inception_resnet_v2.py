import torch
import torch.nn as nn
import torch.nn.functional as F

from inception_v4 import Stem,  ReductionA


class Conv_Bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(Conv_Bn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        return F.relu(x)

class BlockA(nn.Module):
    def __init__(self, in_channels):
        super(BlockA, self).__init__()
        self.branch1 = nn.Sequential(
            Conv_Bn(in_channels, 32, 1),
            Conv_Bn(32, 48, 3, padding='same'),
            Conv_Bn(48, 64, 3, padding='same'),
        )
        self.branch2 = nn.Sequential(
            Conv_Bn(in_channels, 32, 1),
            Conv_Bn(32, 32, 3, padding='same'),
        )
        self.branch3 = Conv_Bn(in_channels, 32, 1)
        self.conv = Conv_Bn(128, 384, 1)
    
    def forward(self, x):
        identity = x
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        x = torch.concat([branch1, branch2, branch3], 1)
        x = self.conv(x)
        x += identity
        return x

class BlockB(nn.Module):
    def __init__(self, in_channels):
        super(BlockB, self).__init__()
        self.branch1 = nn.Sequential(
            Conv_Bn(in_channels, 128, 1),
            Conv_Bn(128, 160, (1, 7), padding=(0, 3)),
            Conv_Bn(160, 192, (7, 1), padding=(3, 0)),
        )
        self.branch2 = Conv_Bn(in_channels, 192, 1)
        self.conv1 = Conv_Bn(384, in_channels, 1)

    def forward(self, x):
        identity = x
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        x = torch.concat([branch1, branch2], 1)
        x = self.conv1(x)
        x += identity
        return x

class ReductionB(nn.Module):
    def __init__(self, in_channels):
        super(ReductionB, self).__init__()
        self.branch1 = nn.Sequential(
            Conv_Bn(in_channels, 256, 1),
            Conv_Bn(256, 288, 3, padding='same'),
            Conv_Bn(288, 320, 3, stride=2),
        )
        self.branch2 = nn.Sequential(
            Conv_Bn(in_channels, 256, 1),
            Conv_Bn(256, 288, 3, stride=2),
        )
        self.branch3 = nn.Sequential(
            Conv_Bn(in_channels, 256, 1),
            Conv_Bn(256, 288, 3, stride=2),
        )
        self.branch4 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        x = torch.concat([branch1, branch2, branch3, branch4], 1)
        return x

class BlockC(nn.Module):
    def __init__(self, in_channels):
        super(BlockC, self).__init__()
        self.branch1 = nn.Sequential(
            Conv_Bn(in_channels, 192, 1),
            Conv_Bn(192, 224, (1, 3), padding=(0, 1)),
            Conv_Bn(224, 256, (3, 1), padding=(1, 0)),
        )
        self.branch2 = Conv_Bn(in_channels, 192, 1)
        self.conv1 = Conv_Bn(448, in_channels, 1)

    def forward(self, x):
        identity = x
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        x = self.conv1(torch.concat([branch1, branch2], 1))
        x += identity
        return x
    

class Inception_ResNet_V2(nn.Module):
    def __init__(self, num_classes):
        super(Inception_ResNet_V2, self).__init__()
        self.stem = Stem(3)
        self.inception_blocks_a = nn.Sequential(*[BlockA(384) for _ in range(5)])
        self.reduction_a = ReductionA(384, 256, 256, 384, 384)
        self.inception_blocks_b = nn.Sequential(*[BlockB(1152) for _ in range(10)])
        self.reduction_b = ReductionB(1152)
        self.inception_blocks_c = nn.Sequential(*[BlockC(2048) for _ in range(5)])
        self.avg_pool = nn.AvgPool2d((8,8))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.inception_blocks_a(x)
        x = self.reduction_a(x)
        x = self.inception_blocks_b(x)
        x = self.reduction_b(x)
        x = self.inception_blocks_c(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x