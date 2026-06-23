"""
VarGFaceNet: Variable Group Convolutional Neural Network for Face Recognition.

Adapted from: "VarGFaceNet: An Efficient Variable Group Convolutional Neural
Network for Lightweight Face Recognition" (ICCVW 2019, Yan et al.)

Key ideas:
- Variable group convolution: uses different group numbers for efficiency
- Squeeze-and-Excitation (SE) modules for channel attention
- GDC (Global Depthwise Conv) embedding head
"""

import torch
import torch.nn as nn


class SEModule(nn.Module):
    """Squeeze-and-Excitation module for channel attention."""

    def __init__(self, channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction,
                             kernel_size=1, bias=False)
        self.relu = nn.PReLU(channels // reduction)
        self.fc2 = nn.Conv2d(channels // reduction, channels,
                             kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return x * scale


class VarGBlock(nn.Module):
    """
    Variable Group Convolution Block.

    Structure: BN -> Conv1x1 -> BN -> PReLU -> DWConv3x3 -> BN -> PReLU
               -> Conv1x1 -> BN -> SE -> (+ residual)
    """

    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super(VarGBlock, self).__init__()
        self.stride = stride
        self.use_se = use_se
        self.use_residual = (stride == 1 and in_channels == out_channels)

        mid_channels = out_channels

        self.layers = nn.Sequential(
            # Pre-BN
            nn.BatchNorm2d(in_channels),
            # Pointwise expansion
            nn.Conv2d(in_channels, mid_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.PReLU(mid_channels),
            # Depthwise conv
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                      stride=stride, padding=1, groups=mid_channels,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.PReLU(mid_channels),
            # Pointwise linear projection
            nn.Conv2d(mid_channels, out_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if self.use_se:
            self.se = SEModule(out_channels)

        # Shortcut for non-residual cases with dimension change
        if not self.use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None

    def forward(self, x):
        out = self.layers(x)
        if self.use_se:
            out = self.se(out)
        if self.use_residual:
            out = out + x
        elif self.shortcut is not None:
            out = out + self.shortcut(x)
        return out


class VarGFaceNet(nn.Module):
    """
    VarGFaceNet for lightweight face recognition.

    Spatial flow (112x112 input):
        Head (stride=2) -> 56x56
        Stage1 (stride=2 first block) -> 28x28
        Stage2 (stride=2 first block) -> 14x14
        Stage3 (stride=2 first block) -> 7x7
        Embedding conv (1x1)
        GDC (7x7 depthwise) -> 1x1
        Linear -> embedding
        BatchNorm1d

    Approximate parameter count: ~5M
    """

    def __init__(self, fp16=False, num_features=512):
        super(VarGFaceNet, self).__init__()
        self.fp16 = fp16

        # Channel configuration
        channels = [40, 80, 160, 320]
        num_blocks = [3, 7, 4]

        # Head setting: conv3x3(stride=2) + dwconv3x3(stride=1)
        self.head = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.PReLU(channels[0]),
            # Depthwise for spatial processing
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1,
                      padding=1, groups=channels[0], bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.PReLU(channels[0]),
        )

        # Body: 3 stages with VarGBlocks
        self.stages = nn.ModuleList()
        in_c = channels[0]
        for i in range(3):
            out_c = channels[i + 1]
            blocks = []
            # First block with stride=2 for spatial downsampling
            blocks.append(VarGBlock(in_c, out_c, stride=2, use_se=True))
            # Remaining blocks with stride=1
            for _ in range(num_blocks[i] - 1):
                blocks.append(VarGBlock(out_c, out_c, stride=1, use_se=True))
            self.stages.append(nn.Sequential(*blocks))
            in_c = out_c

        # Embedding setting
        embed_conv_channels = 512
        self.embed_conv = nn.Sequential(
            nn.Conv2d(channels[-1], embed_conv_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(embed_conv_channels),
            nn.PReLU(embed_conv_channels),
        )

        # GDC (Global Depthwise Conv) — same style as MobileFaceNet
        self.gdc = nn.Sequential(
            nn.Conv2d(embed_conv_channels, embed_conv_channels, kernel_size=7,
                      stride=1, padding=0, groups=embed_conv_channels,
                      bias=False),
            nn.BatchNorm2d(embed_conv_channels),
        )

        self.linear = nn.Linear(embed_conv_channels, num_features, bias=False)
        self.bn = nn.BatchNorm1d(num_features)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        with torch.amp.autocast('cuda', enabled=self.fp16):
            x = self.head(x)
            for stage in self.stages:
                x = stage(x)
        x = self.embed_conv(x.float() if self.fp16 else x)
        return x

    def forward_from_features(self, x):
        x = self.gdc(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.bn(x)
        return x

    def forward_with_features(self, x):
        features = self.forward_features(x)
        embedding = self.forward_from_features(features)
        return embedding, features

    def forward(self, x):
        return self.forward_from_features(self.forward_features(x))


def get_vargfacenet(fp16=False, num_features=512):
    return VarGFaceNet(fp16=fp16, num_features=num_features)
