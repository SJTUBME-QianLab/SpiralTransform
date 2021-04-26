import torch.nn as nn
import torch
import torch.nn.functional as F
# from .utils import load_state_dict_from_url
from compact_bilinear_pooling_v2 import CompactBilinearPooling, BilinearPooling, TrilinearPooling
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channel=3, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, mcbp=False):
        super(ResNet, self).__init__()
        self.mcbp = mcbp
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        if self.mcbp is True:
            input_size1 = 512
            input_size2 = 512
            output_size = 512
            self.mcbpooling = CompactBilinearPooling(input_size1, input_size2, output_size)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        elif self.mcbp is False:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        elif self.mcbp == 'mbp':
            self.mbpooling = BilinearPooling()
            self.fc = nn.Linear(512 * block.expansion * 512, num_classes)   # 双线性维度要改

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):       # 3*100*100
        x = self.conv1(x)    # [32, 64, 50, 50]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [32, 64, 25, 25]

        x = self.layer1(x)  # [32, 64, 25, 25]
        x = self.layer2(x)  # [32, 128, 13, 13]
        x = self.layer3(x)  # [32, 256, 7, 7]
        x = self.layer4(x)  # [32, 512, 4, 4]
        if self.mcbp is True:
            x = x.permute(0, 2, 3, 1)  # [32, 4, 4, 512]
            x = self.mcbpooling(x)  # [32, 512]
        elif self.mcbp == 'mbp':
            x = self.mbpooling(x)  # [32*512*512]
        else:
            x = self.avgpool(x)  # [32, 512, 1, 1]
            x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


class Multimodal_ResNet(nn.Module):
    def __init__(self, num_class, mcbp=False, pretrained=False):
        super(Multimodal_ResNet, self).__init__()
        self.mcbp = mcbp
        # model = resnet18()
        if self.mcbp is True:
            input_size1 = 512
            input_size2 = 512
            output_size = 512
            self.model1 = torch.nn.Sequential(*list(resnet18(pretrained=pretrained).children())[:-2])  # Remove fc (avgpool).
            self.model2 = torch.nn.Sequential(*list(resnet18(pretrained=pretrained).children())[:-2])  # Remove fc (avgpool).
            self.model3 = torch.nn.Sequential(*list(resnet18(pretrained=pretrained).children())[:-2])  # Remove fc (avgpool).
            self.mcbpooling1 = CompactBilinearPooling(input_size1, input_size2, output_size)
            self.mcbpooling2 = CompactBilinearPooling(input_size1, input_size2, output_size)
            self.mcbpooling3 = CompactBilinearPooling(input_size1, input_size2, output_size)
            self.fc1 = torch.nn.Sequential(
                # nn.Dropout(0.5),
                nn.Linear(512, num_class))
            self.fc2 = torch.nn.Sequential(
                # nn.Dropout(0.5),
                nn.Linear(512, num_class))
            self.fc3 = torch.nn.Sequential(
                # nn.Dropout(0.5),
                nn.Linear(512, num_class))
        elif self.mcbp == 'mbp':
            self.model1 = torch.nn.Sequential(*list(resnet18(pretrained=pretrained).children())[:-2])  # Remove fc (avgpool).
            self.model2 = torch.nn.Sequential(*list(resnet18(pretrained=pretrained).children())[:-2])  # Remove fc (avgpool).
            self.model3 = torch.nn.Sequential(*list(resnet18(pretrained=pretrained).children())[:-2])  # Remove fc (avgpool).
            self.mbpooling = BilinearPooling()
            self.fc1 = torch.nn.Sequential(
                # nn.Dropout(0.5),
                nn.Linear(512 * 512, num_class))
            self.fc2 = torch.nn.Sequential(
                # nn.Dropout(0.5),
                nn.Linear(512 * 512, num_class))
            self.fc3 = torch.nn.Sequential(
                # nn.Dropout(0.5),
                nn.Linear(512 * 512, num_class))
        else:
            self.model1 = torch.nn.Sequential(*list(resnet18(pretrained=pretrained).children())[:-1])  # Remove fc
            self.model2 = torch.nn.Sequential(*list(resnet18(pretrained=pretrained).children())[:-1])  # Remove fc
            self.model3 = torch.nn.Sequential(*list(resnet18(pretrained=pretrained).children())[:-1])  # Remove fc
            self.fc1 = torch.nn.Sequential(
                # nn.Dropout(0.2),  # drop 50% neurons
                nn.Linear(512, num_class))
            self.fc2 = torch.nn.Sequential(
                # nn.Dropout(0.2),  # drop 50% neurons
                nn.Linear(512, num_class))
            self.fc3 = torch.nn.Sequential(
                # nn.Dropout(0.2),  # drop 50% neurons
                nn.Linear(512, num_class))
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2, x3):
        out1 = self.model1(x1)  # torch.Size([32, 512, 1, 1])
        out2 = self.model2(x2)
        out3 = self.model3(x3)
        if self.mcbp == 'mcbp':
            out1 = out1.permute(0, 2, 3, 1)  # [32, 4, 8, 512]
            out1 = self.mcbpooling1(out1, out1)
            out2 = out2.permute(0, 2, 3, 1)  # [32, 4, 8, 512]
            out2 = self.mcbpooling2(out2, out2)
            out3 = out3.permute(0, 2, 3, 1)  # [32, 4, 8, 512]
            out3 = self.mcbpooling3(out3, out3)
        elif self.mcbp == 'mbp':
            out1 = self.mbpooling(out1, out1)  # [32, 512*512]
            out2 = self.mbpooling(out2, out2)  # [32, 512*512]
            out3 = self.mbpooling(out3, out3)  # [32, 512*512]
        else:
            out1 = out1.squeeze(2).squeeze(2)
            out2 = out2.squeeze(2).squeeze(2)
            out3 = out3.squeeze(2).squeeze(2)

        out1 = self.fc1(out1)
        out2 = self.fc2(out2)
        out3 = self.fc3(out3)
        out = out1 + out2 + out3    # F.softmax(out1, dim=1)
        return out, F.softmax(out1, dim=1), F.softmax(out2, dim=1), F.softmax(out3, dim=1)


def _resnet(arch, block, layers, in_channel, num_classes, pretrained, progress, mcbp, **kwargs):
    model = ResNet(block, layers, in_channel=in_channel, num_classes=num_classes, mcbp=mcbp, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet10(in_channel=3, num_classes=1000, pretrained=False, progress=True, mcbp=False, **kwargs):
    """Constructs a ResNet-10 model.
    """
    return _resnet('resnet10', BasicBlock, [1, 1, 1, 1], in_channel, num_classes, pretrained, progress, mcbp,
                   **kwargs)

def resnet18(in_channel=3, num_classes=1000, pretrained=False, progress=True, mcbp=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], in_channel, num_classes, pretrained, progress, mcbp,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-50 32x4d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
