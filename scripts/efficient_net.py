import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, squeezed_channels):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, squeezed_channels, 1)
        self.conv2 = nn.Conv2d(squeezed_channels, in_channels, 1)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        se = self.global_avg_pool(x)
        se = self.silu(self.conv1(se))
        se = torch.sigmoid(self.conv2(se))
        x = x * se
        return x

class ExpandBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(ExpandBlock, self).__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.expand(x)

class DepthWiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DepthWiseConv, self).__init__()
        self.depth_wise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x = self.depth_wise(x)
        return x

class ProjectConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProjectConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio):
        super(MBConv, self).__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        
        expand_channels = in_channels * expand_ratio
        self.expand = None
        if expand_ratio != 1:
            self.expand = ExpandBlock(in_channels, expand_channels)

        squeezed_channels = max(1, int(in_channels * se_ratio))
        self.depth_wise = DepthWiseConv(expand_channels, kernel_size, stride)
        self.se = SEBlock(expand_channels, squeezed_channels)
        self.projection = ProjectConv(expand_channels, out_channels)

    def forward(self, x):
        identity = x
        
        if self.expand is not None:
            x = self.expand(x)

        x = self.depth_wise(x)
        x = self.se(x)
        x = self.projection(x)
        
        if self.use_residual:
            x += identity

        return x

class Efficient_Net_B0(nn.Module):
    def __init__(self, num_classes):
        super(Efficient_Net_B0, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )

        # in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, n_repeats
        self.conv1 = self._make_layers(32, 16, 3, 1, 1, 0.25, 1)
        self.conv2 = self._make_layers(16, 24, 3, 2, 6, 0.25, 2)
        self.conv3 = self._make_layers(24, 40, 5, 2, 6, 0.25, 2)
        self.conv4 = self._make_layers(40, 80, 3, 2, 6, 0.25, 3)
        self.conv5 = self._make_layers(80, 112, 5, 1, 6, 0.25, 3)
        self.conv6 = self._make_layers(112, 192, 5, 2, 6, 0.25, 4)
        self.conv7 = self._make_layers(192, 320, 3, 1, 6, 0.25, 1)

        self.head_conv = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )

    def _make_layers(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, n_repeats):
        layers = [MBConv(in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio)]
        for _ in range(1, n_repeats):
            layers.append(MBConv(out_channels, out_channels, kernel_size, 1, expand_ratio, se_ratio))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.head_conv(x)
        x = self.classifier(x)
        return x