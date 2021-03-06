import math
import os
from os.path import join
import torch
from torch import nn

import numpy as np

BatchNorm = nn.BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, return_levels=False,
                 pool_size=7, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            BatchNorm(channels[0]),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        self.avgpool = nn.AvgPool2d(pool_size)
        self.fc = nn.Conv2d(channels[-1], num_classes, kernel_size=1,
                            stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(planes),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                BatchNorm(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        if self.return_levels:
            return y
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)

            return x

    def load_pretrained_model(self):
        model_path = join(os.path.dirname(__file__), "../models/dla34-ba72cf86.pth")
        model_weights = torch.load(model_path)
        self.load_state_dict(model_weights)


def dla34(pretrained, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model()
    return model


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class IDAUp(nn.Module):
    def __init__(self, node_kernel, out_dim, channels, up_factors):
        super(IDAUp, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = Identity()
            else:
                proj = nn.Sequential(
                    nn.Conv2d(c, out_dim,
                              kernel_size=1, stride=1, bias=False),
                    BatchNorm(out_dim),
                    nn.ReLU(inplace=True))
            f = int(up_factors[i])
            if f == 1:
                up = Identity()
            else:
                up = nn.ConvTranspose2d(
                    out_dim, out_dim, f * 2, stride=f, padding=f // 2,
                    output_padding=0, groups=out_dim, bias=False)
                fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)

        for i in range(1, len(channels)):
            node = nn.Sequential(
                nn.Conv2d(out_dim * 2, out_dim,
                          kernel_size=node_kernel, stride=1,
                          padding=node_kernel // 2, bias=False),
                BatchNorm(out_dim),
                nn.ReLU(inplace=True))
            setattr(self, 'node_' + str(i), node)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, layers):
        assert len(self.channels) == len(layers), \
            '{} vs {} layers'.format(len(self.channels), len(layers))
        layers = list(layers)
        for i, l in enumerate(layers):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            layers[i] = upsample(project(l))
        x = layers[0]
        y = []
        for i in range(1, len(layers)):
            node = getattr(self, 'node_' + str(i))
            x = node(torch.cat([x, layers[i]], 1))
            y.append(x)
        return x, y


class DLAUp(nn.Module):
    def __init__(self, channels, scales=(1, 2, 4, 8, 16), in_channels=None):
        super(DLAUp, self).__init__()
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(3, channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        layers = list(layers)
        assert len(layers) > 1
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            x, y = ida(layers[-i - 2:])
            layers[-i - 1:] = y
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class DLASeg(nn.Module):
    def __init__(self, base_name, heads,
                 pretrained=True, down_ratio=4, head_conv=256):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.heads = heads
        self.first_level = int(np.log2(down_ratio))
        self.base = globals()[base_name](
            pretrained=pretrained, return_levels=True)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(channels[self.first_level:], scales=scales)

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(channels[self.first_level], head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=1, stride=1,
                              padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(channels[self.first_level], classes,
                               kernel_size=1, stride=1,
                               padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x[self.first_level:])

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return ret


def centernet(heads, head_conv=256, down_ratio=4,
    pretrained=True):
    model = DLASeg('dla34', heads,
                   pretrained=pretrained,
                   down_ratio=down_ratio,
                   head_conv=head_conv)
    return model

if __name__ == '__main__':
    import sys
    sys.path.append('../master')
    from models.utils import (profile, profile_training, init_torch_seeds,
        model_info, profile_FP16)
    
    from models.dla import centernet
    
    heads = {"cpt_hm": 30, "cpt_off": 2, "wh": 2}
    model = centernet(heads)
    
    #"""
    init_torch_seeds(seed=1234)
    model_info(model)
    #profile_FP16(model)
    #profile(model, amp=True)
    profile_training(model, amp=True, bsz=32)
    #"""
    exit()
    """ investigate batch size / memory relation
    PyTorch version 1.7.1
    CUDA version 11.0
    cuDNN version 8005
    cuDNN deterministic False
    cuDNN benchmark True
    Model Summary: 310 layers, 17.9M parameters, 17.9M gradients, 62.0 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    Forward time: 161.364ms (cpu)
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 45.408ms (cuda)
    Backward time: 311.690ms (cuda)
    Maximum of managed memory: 8.428453888GB
    
    OOM with 32 bsz
    """
    
    # now try diff convs for centernet
    from models import convs
    
    # settings previously:
    #convs.BASE = nn.Conv2d
    #convs.GROUPS = 1  ...but now we will try other convs
    
    from models.convs import (DeformConv, SpatiallyConv, 
        DepthwiseConv, FlattenedConv, GroupedConv, ShuffledGroupedConv)
    
    init_torch_seeds(seed=1234)
    
    # currently OOM with deformable and autocast not supported,
    # (error states FP32 has to be used)
    
    """
    # releases all unoccupied cached memory held by the caching allocator 
    torch.cuda.empty_cache()
    
    should not be called, an indeed it does not free memory s.t. there is
    somehow more available for pytorch, it even slows down since memory that
    was allocate and reserved must be reallocated (costly)
    """
    
    for Base in [DeformConv, SpatiallyConv, DepthwiseConv, 
        FlattenedConv, GroupedConv, ShuffledGroupedConv]:
        
        # change 'BASE' class for 'Conv' wrapper class
        convs.BASE = Base
        if 'group' in Base.__name__.lower():
            convs.GROUPS = 8
        else:
            convs.GROUPS = 1
            
        print(f'BASE: {convs.BASE.__name__}, GROUPS: {convs.GROUPS}')
        # build new model with other conv block types
        model = centernet(heads)
        model_info(model)
        profile(model, amp=True)
        profile_training(model, amp=True)
    
    """
    ------- old version -------
    PyTorch version 1.7.1
    CUDA version 11.0
    cuDNN version 8005
    cuDNN deterministic False
    cuDNN benchmark True
    Model Summary: 220 layers, 18.5M parameters, 18.5M gradients, 62.1 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 10.269ms (cuda)
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 75.848ms (cuda)
    Backward time: 408.222ms (cuda)
    Maximum of managed memory: 12.696158208GB
    
    another day: => benchmark may select diff. algorithm
    PyTorch version 1.7.1
    CUDA version 11.0
    cuDNN version 8005
    cuDNN deterministic False
    cuDNN benchmark True
    Model Summary: 220 layers, 18.5M parameters, 18.5M gradients, 62.1 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 13.174ms (cuda)
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 75.138ms (cuda)
    Backward time: 408.696ms (cuda)
    Maximum of managed memory: 14.034141184GB
    
    # without benchmark: => significanlty slower
    Model Summary: 220 layers, 18.5M parameters, 18.5M gradients, 62.1 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    Forward time: 50.602ms (cuda)
    Input size: torch.Size([16, 3, 512, 512])
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 89.595ms (cuda)
    Backward time: 550.627ms (cuda)
    Maximum of managed memory: 13.490978816GB
    
    # with benchmark and automatic mixed precision (amp):
    PyTorch version 1.7.1
    CUDA version 11.0
    cuDNN version 8005
    cuDNN deterministic False
    cuDNN benchmark True
    Model Summary: 220 layers, 18.5M parameters, 18.5M gradients, 62.1 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 18.515ms (cuda)
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 49.097ms (cuda)
    Backward time: 292.130ms (cuda)
    Maximum of managed memory: 8.32569344GB
    
    ------ new version --------
    PyTorch version 1.7.1
    CUDA version 11.0
    cuDNN version 8005
    cuDNN deterministic False
    cuDNN benchmark True
    Model Summary: 310 layers, 17.9M parameters, 17.9M gradients, 62.0 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 12.555ms (cuda)
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 74.691ms (cuda)
    Backward time: 419.300ms (cuda)
    Maximum of managed memory: 15.216934912GB
    
    another day:
    PyTorch version 1.7.1
    CUDA version 11.0
    cuDNN version 8005
    cuDNN deterministic False
    cuDNN benchmark True
    Model Summary: 310 layers, 17.9M parameters, 17.9M gradients, 62.0 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 49.608ms (cuda)
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 89.089ms (cuda)
    Backward time: 415.187ms (cuda)
    Maximum of managed memory: 15.105785856GB
    
    # with benchmark and automatic mixed precision (amp):
    PyTorch version 1.7.1
    CUDA version 11.0
    cuDNN version 8005
    cuDNN deterministic False
    cuDNN benchmark True
    Model Summary: 310 layers, 17.9M parameters, 17.9M gradients, 62.0 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 20.943ms (cuda)
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 47.379ms (cuda)
    Backward time: 314.996ms (cuda)
    Maximum of managed memory: 8.64026624GB
    
    # tried profile_FP16: (slower as mixed precision??)
    # it might be the optimization done in amp mode!
    PyTorch version 1.7.1
    CUDA version 11.0
    cuDNN version 8005
    cuDNN deterministic False
    cuDNN benchmark True
    Model Summary: 310 layers, 17.9M parameters, 17.9M gradients, 62.0 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 30.367ms (cuda)
    
    ------ new version (other convs then the nn.Conv2d) --------
    and benchmark with automatic mixed precision...
    
    no amp with DeformConv possible...
    
    PyTorch version 1.7.1
    CUDA version 11.0
    cuDNN version 8005
    cuDNN deterministic False
    cuDNN benchmark True
    BASE: SpatiallyConv, GROUPS: 1
    Model Summary: 424 layers, 16.6M parameters, 16.6M gradients, 58.4 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 76.315ms (cuda)
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 66.443ms (cuda)
    Backward time: 466.973ms (cuda)
    Maximum of managed memory: 14.992539648GB
    BASE: DepthwiseConv, GROUPS: 1
    Model Summary: 424 layers, 3.98M parameters, 3.98M gradients, 24.1 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 42.954ms (cuda)
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 57.037ms (cuda)
    Backward time: 384.364ms (cuda)
    Maximum of managed memory: 10.624172032GB
    BASE: FlattenedConv, GROUPS: 1
    Model Summary: 481 layers, 3.97M parameters, 3.97M gradients, 25.1 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 34.495ms (cuda)
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    ...OOM error => even if network size is much smaller than the
    original implementation one should not forget that the memory
    expensive part of a network is caching the activation from the
    forward pass for the backward calculation, since the flattened 
    convolution has 3 times! nn.Conv2d module in it we ran OOM even
    if the original implemenation only used half of the available (16GB)
    memory (8.64026624GB), for convolutions feature maps of the size of
    output feature maps have to be stored
    
    BASE: GroupedConv, GROUPS: 8
    Model Summary: 367 layers, 17.9M parameters, 17.9M gradients, 62.0 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 48.222ms (cuda)
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 48.164ms (cuda)
    Backward time: 316.168ms (cuda)
    Maximum of managed memory: 8.64026624GB
    BASE: ShuffledGroupedConv, GROUPS: 8
    Model Summary: 367 layers, 17.9M parameters, 17.9M gradients, 62.0 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 13.272ms (cuda)
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 48.040ms (cuda)
    Backward time: 315.715ms (cuda)
    Maximum of managed memory: 11.106516992GB
    """
    