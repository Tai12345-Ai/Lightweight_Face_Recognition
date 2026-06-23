"""
ShuffleFaceNet: ShuffleNetV2-style backbone for lightweight face recognition.

Adapted from ShuffleNetV2 (Ma et al., ECCV 2018) for 112x112 aligned face input.
Uses GDC (Global Depthwise Conv) embedding head similar to MobileFaceNet.

Reference:
- ShuffleNetV2: Practical Guidelines for Efficient CNN Architecture Design (ECCV 2018)
"""

import torch
import torch.nn as nn


def channel_shuffle(x, groups):
    """Channel shuffle operation for information flow across channel groups."""
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x


class InvertedResidual(nn.Module):
    """ShuffleNetV2 building block with channel split and shuffle."""

    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        if stride not in [1, 2]:
            raise ValueError('Illegal stride value: {}'.format(stride))
        self.stride = stride

        branch_features = oup // 2

        if self.stride == 2:
            # Both branches active on full input
            self.branch1 = nn.Sequential(
                # depthwise
                nn.Conv2d(inp, inp, kernel_size=3, stride=self.stride,
                          padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pointwise
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1,
                          padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.PReLU(branch_features),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1,
                          padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.PReLU(branch_features),
                nn.Conv2d(branch_features, branch_features, kernel_size=3,
                          stride=self.stride, padding=1,
                          groups=branch_features, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.PReLU(branch_features),
            )
        else:
            # stride == 1: channel split, only one branch processes
            assert inp == branch_features * 2
            self.branch1 = None
            self.branch2 = nn.Sequential(
                nn.Conv2d(branch_features, branch_features, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.PReLU(branch_features),
                nn.Conv2d(branch_features, branch_features, kernel_size=3,
                          stride=1, padding=1, groups=branch_features,
                          bias=False),
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.PReLU(branch_features),
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out


class ShuffleFaceNet(nn.Module):
    """
    ShuffleNetV2 adapted for face recognition.

    Spatial flow (112x112 input):
        Conv1 (stride=2) -> 56x56
        Stage2 (stride=2 first block) -> 28x28
        Stage3 (stride=2 first block) -> 14x14
        Stage4 (stride=2 first block) -> 7x7
        Conv5 (1x1)
        GDC (7x7 depthwise) -> 1x1
        Linear -> embedding
        BatchNorm1d
    """

    def __init__(self, fp16=False, num_features=512, width_mult=1.0):
        super(ShuffleFaceNet, self).__init__()
        self.fp16 = fp16

        # Width configuration (standard ShuffleNetV2 channel counts)
        stage_repeats = [4, 8, 4]
        if width_mult == 0.5:
            stage_out_channels = [24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            stage_out_channels = [24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            stage_out_channels = [24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            stage_out_channels = [24, 244, 488, 976, 2048]
        else:
            raise ValueError('Unsupported width_mult: {}'.format(width_mult))

        # Stage 1: initial conv (no MaxPool — faces are already 112x112)
        input_channels = 3
        output_channels = stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.PReLU(output_channels),
        )

        # Stages 2-4: ShuffleNetV2 blocks
        input_channels = output_channels
        self.stages = nn.ModuleList()
        for i in range(len(stage_repeats)):
            output_channels = stage_out_channels[i + 1]
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for _ in range(stage_repeats[i] - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            self.stages.append(nn.Sequential(*seq))
            input_channels = output_channels

        # Conv5: 1x1 channel expansion
        output_channels = stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.PReLU(output_channels),
        )

        # GDC (Global Depthwise Conv) — same head style as MobileFaceNet
        self.gdc = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=7,
                      stride=1, padding=0, groups=output_channels, bias=False),
            nn.BatchNorm2d(output_channels),
        )

        # Embedding projection
        self.linear = nn.Linear(output_channels, num_features, bias=False)
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
            x = self.conv1(x)
            for stage in self.stages:
                x = stage(x)
            x = self.conv5(x)
        return x

    def forward_from_features(self, x):
        x = self.gdc(x.float() if self.fp16 else x)
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


def get_shufflefacenet(fp16=False, num_features=512, width_mult=1.0):
    return ShuffleFaceNet(fp16=fp16, num_features=num_features,
                          width_mult=width_mult)
