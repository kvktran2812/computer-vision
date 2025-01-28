import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, **kwargs):
        super(DepthwiseConv, self).__init__()
        self.relu = nn.ReLU6(inplace=True)

        # Depthwise
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, **kwargs)
        self.bn1 = nn.BatchNorm2d(in_channels)

        # Pointwise
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class Conv2dBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, **kwargs):
        super(Conv2dBn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x
    

class MobileNet(nn.Module):
    def __init__(self, num_classes):
        super(MobileNet, self).__init__()
        self.relu = nn.ReLU6(inplace=True)
        self.conv1 = Conv2dBn(3, 32, 3, stride=2, padding=1)
        self.conv2 = DepthwiseConv(32, 64, 3, stride=1, padding=1)
        self.conv3 = DepthwiseConv(64, 128, 3, stride=2, padding=1)
        self.conv4 = DepthwiseConv(128, 128, 3, stride=1, padding=1)
        self.conv5 = DepthwiseConv(128, 256, 3, stride=2, padding=1)
        self.conv6 = DepthwiseConv(256, 256, 3, stride=1, padding=1)
        self.conv7 = DepthwiseConv(256, 512, 3, stride=2, padding=1)
        self.conv8 = nn.Sequential(*[DepthwiseConv(512, 512, 3, stride=1, padding=1) for _ in range(5)])
        self.conv9 = DepthwiseConv(512, 1024, 3, stride=2, padding=1)
        self.conv10 = DepthwiseConv(1024, 1024, 3, stride=1, padding=1)
        self.avg_pool = nn.AvgPool2d((7,7))
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x