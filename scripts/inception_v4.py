import torch
import torch.nn as nn

class Stem(nn.Module):
    # For Inception-v4 and Inception-ResNet-v2 
    def __init__(self, in_channels):
        super(Stem, self).__init__()

        # first conv
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(32)

        # second conv
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(64, 96, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(96)

        self.branch1 = nn.Sequential(
            nn.Conv2d(160, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            self.relu,
            nn.Conv2d(64, 96, kernel_size=3),
            nn.BatchNorm2d(96),
            self.relu
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(160, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            self.relu,
            nn.Conv2d(64, 64, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(64),
            self.relu,
            nn.Conv2d(64, 64, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(64),
            self.relu,
            nn.Conv2d(64, 96, kernel_size=3),
            nn.BatchNorm2d(96),
            self.relu,
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=2),
            nn.BatchNorm2d(192),
            self.relu
        )
        
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        maxpool1 = self.maxpool1(x)
        conv4 = self.relu(self.bn4(self.conv4(x)))

        x = torch.concat([maxpool1, conv4], 1)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)

        x = torch.concat([branch1, branch2], 1)
        maxpool2 = self.maxpool2(x)
        conv5 = self.conv5(x)
        x = torch.concat([maxpool2, conv5], 1)
        return x
    
class BlockA(nn.Module):
    def __init__(self, in_channels):
        super(BlockA, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            self.relu,
            nn.Conv2d(64, 96, kernel_size=3, padding='same'), 
            nn.BatchNorm2d(96),
            self.relu,
            nn.Conv2d(96, 96, kernel_size=3, padding='same'), 
            nn.BatchNorm2d(96),
            self.relu,
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            self.relu,
            nn.Conv2d(64, 96, kernel_size=3, padding='same'), 
            nn.BatchNorm2d(96),
            self.relu,
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=1),
            nn.BatchNorm2d(96),
            self.relu,
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, padding=1, stride=1),
            nn.Conv2d(in_channels, 96, kernel_size=1, padding='same'),
            nn.BatchNorm2d(96),
            self.relu,
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        x = torch.concat([branch1, branch2, branch3, branch4], 1)
        return x

class ReductionA(nn.Module):
    def __init__(self, in_channels, k, l, m, n):
        super(ReductionA, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, k, kernel_size=1, padding='same'),
            nn.BatchNorm2d(k),
            self.relu,
            nn.Conv2d(k, l, kernel_size=3, padding='same'),
            nn.BatchNorm2d(l),
            self.relu,
            nn.Conv2d(l, m, kernel_size=3, stride=2),
            nn.BatchNorm2d(m),
            self.relu,
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, n, kernel_size=3, stride=2),
            nn.BatchNorm2d(n),
            self.relu,
        )
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        x = torch.concat([branch1, branch2, branch3], 1)
        return x
    
class BlockB(nn.Module):
    def __init__(self, in_channels):
        super(BlockB, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=1),
            nn.BatchNorm2d(192),
            self.relu,
            nn.Conv2d(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(192),
            self.relu,
            nn.Conv2d(192, 224, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(224),
            self.relu,
            nn.Conv2d(224, 224, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(224),
            self.relu,
            nn.Conv2d(224, 256, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(256),
            self.relu,
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=1),
            nn.BatchNorm2d(192),
            self.relu,
            nn.Conv2d(192, 224, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(224),
            self.relu,
            nn.Conv2d(224, 256, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(256),
            self.relu,
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 384, kernel_size=1, padding='same'),
            nn.BatchNorm2d(384),
            self.relu,
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 128, kernel_size=1, padding='same'),
            nn.BatchNorm2d(128),
            self.relu,
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        x = torch.concat([branch1, branch2, branch3, branch4], 1)
        return x

class ReductionB(nn.Module):
    def __init__(self, in_channels):
        super(ReductionB, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, padding='same'),
            nn.BatchNorm2d(256),
            self.relu,
            nn.Conv2d(256, 256, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(256),
            self.relu,
            nn.Conv2d(256, 320, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(320),
            self.relu,
            nn.Conv2d(320, 320, kernel_size=3, stride=2),
            nn.BatchNorm2d(320),
            self.relu,
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=1, padding='same'),
            nn.BatchNorm2d(192),
            self.relu,
            nn.Conv2d(192, 192, kernel_size=3, stride=2),
            nn.BatchNorm2d(192),
            self.relu,
        )
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        x = torch.concat([branch1, branch2, branch3], 1)
        return x
    
class BlockC(nn.Module):
    def __init__(self, in_channels):
        super(BlockC, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 384, kernel_size=1),
            nn.BatchNorm2d(384),
            self.relu,
            nn.Conv2d(384, 448, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(448),
            self.relu,
            nn.Conv2d(448, 512, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(512),
            self.relu,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, 384, kernel_size=1),
            nn.BatchNorm2d(384),
            self.relu,
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(256),
            self.relu,
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            self.relu,
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            self.relu,
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(256),
            self.relu,
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, padding='same'),
            nn.BatchNorm2d(256),
            self.relu,
        )
        self.branch6 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 256, kernel_size=1, padding='same'),
            nn.BatchNorm2d(256),
            self.relu,
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        branch1 = self.branch1(conv1)
        branch2 = self.branch2(conv1)
        branch3 = self.branch3(conv2)
        branch4 = self.branch4(conv2)
        branch5 = self.branch5(x)
        branch6 = self.branch6(x)
        
        x = torch.concat([branch1, branch2, branch3, branch4, branch5, branch6], 1)
        return x
    
class InceptionV4(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV4, self).__init__()
        self.stem = Stem(3)
        self.inception_a = nn.Sequential(*[BlockA(384) for _ in range(4)])
        self.reduction_a = ReductionA(384, 192, 224, 256, 384)
        self.inception_b = nn.Sequential(*[BlockB(1024) for _ in range(7)])
        self.reduction_b = ReductionB(1024)
        self.inception_c = nn.Sequential(*[BlockC(1536) for _ in range(3)])
        self.avg_pool = nn.AvgPool2d(kernel_size=(8,8))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1536, num_classes)
    
    def forward(self, x):
        x = self.stem(x)

        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x