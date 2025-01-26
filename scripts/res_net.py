import torch
import torch.nn as nn


# ResNet implementation
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.identity_mapping = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.identity_mapping = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.identity_mapping(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + identity
        x = self.relu(x)
        return x
    

class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_channels=None, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        
        if bottleneck_channels is not None:
            mid_channels = bottleneck_channels
        else:
            mid_channels = out_channels // 4
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_mapping = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.identity_mapping = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.identity_mapping(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=200):
        super(ResNet34, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(in_channels=64, out_channels=64, num_blocks=2, stride=1)
        self.layer2 = self.make_layer(in_channels=64, out_channels=128, num_blocks=2, stride=2)
        self.layer3 = self.make_layer(in_channels=128, out_channels=256, num_blocks=2, stride=2)
        self.layer4 = self.make_layer(in_channels=256, out_channels=512, num_blocks=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))  # Downsampling block
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))  # Identity blocks
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    
class ResNet34(nn.Module):
    def __init__(self, num_classes=200):
        super(ResNet34, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(in_channels=64, out_channels=64, num_blocks=3, stride=1)
        self.layer2 = self.make_layer(in_channels=64, out_channels=128, num_blocks=4, stride=2)
        self.layer3 = self.make_layer(in_channels=128, out_channels=256, num_blocks=6, stride=2)
        self.layer4 = self.make_layer(in_channels=256, out_channels=512, num_blocks=3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))  # Downsampling block
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))  # Identity blocks
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    

class ResNet50(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(64, 256, 64, stride=1, num_blocks=3)
        self.layer2 = self.make_layer(256, 512, 128, stride=2, num_blocks=4)
        self.layer3 = self.make_layer(512, 1024, 256, stride=2, num_blocks=6)
        self.layer4 = self.make_layer(1024, 2048, 512, stride=2, num_blocks=3)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)

    def make_layer(self, in_channels, out_channels, bottleneck_channels, stride, num_blocks):
        layers = []
        layers.append(BottleNeck(in_channels, out_channels, stride=stride, bottleneck_channels=bottleneck_channels))  
        for _ in range(1, num_blocks):
            layers.append(BottleNeck(out_channels, out_channels, bottleneck_channels=bottleneck_channels)) 
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class ResNet101(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet101, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(64, 256, 64, stride=1, num_blocks=3)
        self.layer2 = self.make_layer(256, 512, 128, stride=2, num_blocks=4)
        self.layer3 = self.make_layer(512, 1024, 256, stride=2, num_blocks=23)
        self.layer4 = self.make_layer(1024, 2048, 512, stride=2, num_blocks=3)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)

    def make_layer(self, in_channels, out_channels, bottleneck_channels, stride, num_blocks):
        layers = []
        layers.append(BottleNeck(in_channels, out_channels, stride=stride, bottleneck_channels=bottleneck_channels))  
        for _ in range(1, num_blocks):
            layers.append(BottleNeck(out_channels, out_channels, bottleneck_channels=bottleneck_channels)) 
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out



class ResNet152(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet152, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(64, 256, 64, stride=1, num_blocks=3)
        self.layer2 = self.make_layer(256, 512, 128, stride=2, num_blocks=8)
        self.layer3 = self.make_layer(512, 1024, 256, stride=2, num_blocks=36)
        self.layer4 = self.make_layer(1024, 2048, 512, stride=2, num_blocks=3)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)

    def make_layer(self, in_channels, out_channels, bottleneck_channels, stride, num_blocks):
        layers = []
        layers.append(BottleNeck(in_channels, out_channels, stride=stride, bottleneck_channels=bottleneck_channels))  
        for _ in range(1, num_blocks):
            layers.append(BottleNeck(out_channels, out_channels, bottleneck_channels=bottleneck_channels)) 
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out