import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import torchvision
from torchvision import ops
from torch.cuda.amp import autocast
import sys
import numpy as np
import math

from .utils import (autopad, fuse_conv_and_bn, 
    time_synchronized, model_info)

BASE = nn.Conv2d
GROUPS = 1

class DeformConv(nn.Module):
    def __init__(self, chi, cho, k, s=1, p=None, dilation=1, groups=1, bias=True):
        super().__init__() 
        k = k if isinstance(k, tuple) else (k, k) 
        p = autopad(k, p)
        self.conv = ops.DeformConv2d(chi, cho, k, s, p, 
            dilation=dilation, groups=groups, bias=bias)
        # for each group we need output channels of 2 to get for each kernel weight
        # offset position in x, y (2 channels) and we have to know that for every 
        # pixel in the convolution output, thus we use same kernel size and padding!!
        self.offset = nn.Conv2d(chi, groups * 2 * k[0] * k[1], k, s, p, 
            dilation=dilation, groups=groups, bias=True)
        self._init_offset()

    def _init_offset(self):
        # as the original paper suggests initialize offsets with zeros
        # thus we start with a standard convolution
        nn.init.constant_(self.offset.weight, 0)
        nn.init.constant_(self.offset.bias, 0)

    @autocast(enabled=False)  # doesn't support amp
    def forward(self, x):
        return self.conv(x, self.offset(x))
    
    def fuse(self, bn):
        self.conv = fuse_conv_and_bn(self.conv, bn)
        return self

class SpatiallyConv(nn.Module):
    def __init__(self, chi, cho, k, s=1, p=None, dilation=1, groups=1, bias=True):
        super().__init__() 
        # decreases complexity, but kernel space limited
        k = k if isinstance(k, tuple) else (k, k)
        p = p if isinstance(p, tuple) else (p, p)
        s = s if isinstance(s, tuple) else (s, s)
        p = autopad(k, p)
        self.conv1 = nn.Conv2d(chi, chi, (k[0], 1), (s[0], 1), (p[0], 0), 
            dilation=dilation, groups=groups, bias=True)
        self.conv2 = nn.Conv2d(chi, cho, (1, k[1]), (1, s[1]), (0, p[1]), 
            dilation=dilation, groups=groups, bias=bias)  
        
    def forward(self, x):
        return self.conv2(self.conv1(x))
    
    def fuse(self, bn):
        self.conv2 = fuse_conv_and_bn(self.conv2, bn)
        return self

class DepthwiseConv(nn.Module):
    def __init__(self, chi, cho, k, s=1, p=None, dilation=1, groups=1, bias=True):
        super().__init__() 
        p = autopad(k, p)
        # decreases complexity, smaller networks can be wider,
        # each filter soley has access to a single input channel
        # and we keep the number of input channels at first
        self.conv1 = nn.Conv2d(chi, chi, k, s, p, 
            dilation=dilation, groups=chi, bias=True)
        # learn channelwise(inter group) correlation with 1x1 convolutions
        self.conv2 = nn.Conv2d(chi, cho, 1, 1, 0, 
            dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv2(self.conv1(x))
    
    def fuse(self, bn):
        self.conv2 = fuse_conv_and_bn(self.conv2, bn)
        return self

class FlattenedConv(nn.Module):
    def __init__(self, chi, cho, k, s=1, p=None, dilation=1, groups=1, bias=True):
        super().__init__() 
        # paper claims importance of bias term!
        k = k if isinstance(k, tuple) else (k, k)
        p = p if isinstance(p, tuple) else (p, p)
        s = s if isinstance(s, tuple) else (s, s)
        p = autopad(k, p)
        # lateral, kernel: C x 1 x 1
        self.conv1 = nn.Conv2d(chi, cho, 1, 1, 0, groups=groups, bias=True)
        # vertical, kernel: 1 x Y x 1
        self.conv2 = nn.Conv2d(cho, cho, (k[0], 1), (s[0], 1), (p[0], 0), 
            dilation=dilation, groups=cho, bias=True)
        # horizontal, kernel: 1 x 1 x X,
        # last term can omit bias e.g. if batchnorm is done anyway afterwards 
        self.conv3 = nn.Conv2d(cho, cho, (1, k[1]), (1, s[1]), (0, p[1]), 
            dilation=dilation, groups=cho, bias=bias)

    def forward(self, x):
        return self.conv3(self.conv2(self.conv1(x)))
    
    def fuse(self, bn):
        self.conv3 = fuse_conv_and_bn(self.conv3, bn)
        return self

class ShuffledGroupedConv(nn.Module):
    def __init__(self, chi, cho, k, s=1, p=None, dilation=1, groups=8, bias=True):
        super().__init__()  # typically groups are 2, 4, 8, 16
        self.cho = cho
        self.groups = groups
        p = autopad(k, p)
        # decreases complexity, the idea of grouped convolutions is that 
        # the correlation between feature channels is sparse anyway,
        # and here we will be even more sparse since we only allow 
        # intra channel group correlation, 
        # use grouped convolution also for 1x1 convolutions (see ShuffleNet)
        # which are then called pointwise grouped convolutions
        self.conv = nn.Conv2d(chi, cho, k, s, p, 
            dilation=dilation, groups=groups, bias=bias)
        # example forward pass: 
        # assume g=2 groups and 6 channels (=> n=3 group size) 
        # 111222 (entries of the two groups are 1 and 2 respectively) 
        # reshaping to (2, 6): 
        # 111
        # 222
        # transposing:
        # 12
        # 12
        # 12
        # flattening:
        # 121212 => shuffled!

    def forward(self, x):
        x = self.conv(x)
        if self.groups != 1:
            # x has g * n output channels with g beeing the number 
            # of groups; to shuffel reshape to (g, n)
            x = x.reshape(x.size(0), self.groups, 
                int(self.cho / self.groups), x.size(-2), x.size(-1))
            # then transpose in the (g, n) dimensions
            x = torch.transpose(x, 1, 2)
            # finally flatten dimension (n, g) => channels shuffled!
            x = x.reshape(x.size(0), -1, x.size(-2), x.size(-1))
        return x 
    
    def fuse(self, bn):
        if self.groups != 1:  # here conv weights and bn weights don't match 
            s = self.conv.weight.shape  # remember original shape: cho, chi, k, k
            # shuffle conv weight in 'cho' dimension to match 'bn'
            x = self.conv.weight.reshape(self.conv.out_channels, -1)  # cho, chi*k^2 
            x = x.reshape(self.groups, int(self.cho / self.groups), -1)  # g, n, chi*k^2 
            x = torch.transpose(x, 1, 0)  # n, g, chi*k^2 
            x = x.reshape(self.conv.out_channels, -1)  # cho, chi*k^2 but shuffled
            self.conv.weight = x.reshape(*s)  # reshape copies, re-assign
            
            self.conv = fuse_conv_and_bn(self.conv, bn)  # now weigths match
            
            # shuffle conv weight in 'cho' dimension back to initial order
            x = self.conv.weight.reshape(self.conv.out_channels, -1)  # cho, chi*k^2
            x = x.reshape(int(self.cho / self.groups), self.groups, -1)  # n, g, chi*k^2 
            x = torch.transpose(x, 1, 0)  # g, n, chi*k^2 
            x = x.reshape(self.conv.out_channels, -1)  # cho, chi*k^2
            self.conv.weight = x.reshape(*s)  # reshape copies, re-assign
        else:  # straight forward case
            self.conv = fuse_conv_and_bn(self.conv, bn)
        return self
    
__BaseClasses = (
    nn.Conv2d,
    FlattenedConv,
    DeformConv,
    SpatiallyConv,
    DepthwiseConv,
    ShuffledGroupedConv,
)

class Conv(nn.Module):
    def __init__(self, chi, cho, k=1, s=1, p=None, d=1, g=GROUPS, act=True, affine=True): 
        super().__init__()
        if chi % g != 0 or cho % g != 0:
            g = 1
            print(f'Channel {chi} or {cho} not divisible by groups: {g}; using groups=1')
        p = autopad(k, p)
        self.conv = BASE(chi, cho, k, s, p, dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(cho, affine=affine)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def fuseforward(self, x):
        return self.act(self.conv(x))
    
    def fuse(self):  # merge batchnorm and convolution for inference speed up
        if not isinstance(self.conv, nn.Conv2d):
            # each custom BaseClass has own fuse method
            self.conv = self.conv.fuse(self.bn)  
        else:
            self.conv = fuse_conv_and_bn(self.conv, self.bn)
        delattr(self, 'bn')  # remove batchnorm
        self.forward = self.fuseforward  # update forward

def test_base(BaseClass, nruns=100, device='cuda'):
    # will do basic sanity check, we want to get same spatial 
    # dimension with custom convolutions as in standard module
    # to swap convolution types conveniently
    chi, cho, k, s = 8, 32, 3, 1
    
    x = torch.randn(16, chi, 512, 512)
    conv = BaseClass(chi, cho, k, s, autopad(k))
    conv_ = nn.Conv2d(chi, cho, k, s, autopad(k))
    
    if 'cuda' in device: 
        assert torch.cuda.is_available()
        conv.cuda().train()
        conv_.cuda().train()
        x = x.cuda()
        
        if torch.backends.cudnn.benchmark:
            # have to do warm up iterations for fair comparison
            print('benchmark warm up...')
            with torch.no_grad():
                for _ in range(50):
                    _ = conv(x)
    else:
        conv.cpu().train()
        conv_.cpu().train()
        nruns=1
    
    p = sum(p.numel() for p in conv.parameters() if p.requires_grad)
    p_ = sum(p.numel() for p in conv_.parameters() if p.requires_grad)
    # relative number of parameter change in brackets w.r.t. nn.conv2d
    print(f'Number of parameters: {p} ({p / p_ * 100:.2f}%)')
    
    # ensure same behaviour as standard module
    out = conv(x)
    out_ = conv_(x)
    assert out.shape == out_.shape, f'Shape missmatch, should be {out_.shape} but is {out.shape}'
    
    # g0 = torch.randn_like(out)
    # performance test without feature/target loading
    # because that would require a significant amount of overhead
    start = time_synchronized()
    for _ in range(nruns):
        out = conv(x)
        for param in conv.parameters():
            param.grad = None
        out.mean().backward()  # out.backward(g0)
    end = time_synchronized()
    
    print(f'Forward + Backward time: {(end - start) * 1000 / nruns:.3f}ms')
        
def test_fuse(BaseClass, nruns=10, device='cuda'):
    global BASE
    BASE = BaseClass

    chi, cho, k, s = 8, 16, 3, 1
    
    module = Conv(chi, cho, k, s)
    assert hasattr(module, 'fuse')
    
    x = torch.randn(8, chi, 512, 512)
    
    if 'cuda' in device: 
        assert torch.cuda.is_available()
        module.cuda().train()
        x = x.cuda()
    else:
        module.cpu().train()
        nruns=1
    
    optimizer = optim.Adam(module.parameters())
    
    # traing batchnorm and convolution: 
    for _ in range(nruns):
        optimizer.zero_grad(set_to_none=True)  # version >=1.7
        out = module(x)
        loss = out.mean()
        loss.backward()
        optimizer.step()
    
    # test fuse function:
    x = torch.randn(1, chi, 512, 512, device=device)
    module.eval()  # MUST BE in eval mode to work
    
    with torch.no_grad():  # compare module outputs
        unfused = module(x)
        module.fuse()
        #import pdb; pdb.set_trace()
        fused = module(x)  # merged batchnorm
        d = torch.norm(unfused - fused).div(unfused.norm()).item()
        print('fuse relative error: %.8f' % d)
        
def test_fuse_conv_and_bn():
    x = torch.randn(16, 3, 256, 256)
    rn18 = torchvision.models.resnet18(pretrained=True)
    rn18.eval()
    net = torch.nn.Sequential(
        rn18.conv1,
        rn18.bn1
    )
    y1 = net.forward(x)
    fusedconv = fuse_conv_and_bn(net[0], net[1])
    y2 = fusedconv.forward(x)
    d = (y1 - y2).norm().div(y1.norm()).item()
    print('fuse relative error: %.8f' % d)
    
class UpSample(nn.Module):
    def __init__(self, chi, cho, k=1, s=3, p=None, d=1, g=GROUPS, sf=2):
        super().__init__()  # avoids checkerboard artifacts
        self.sf = sf
        self.conv = Conv(chi, cho, k, s, p, d=d, g=g)
    
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=self.sf, mode='nearest'))

class Bottleneck(nn.Module):
    def __init__(self, chi, cho, k=3, s=1, p=None, d=1, g=GROUPS, e=0.5):
        super().__init__()
        chh = int(cho * e)
        self.conv1 = Conv(chi, chh, 1, 1, 0, d=d, g=g)
        self.conv2 = Conv(chh, chh, k, s, p=d, d=d, g=g)
        self.conv3 = Conv(chh, cho, 1, 1, 0, d=d, g=g, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        return self.relu(residual + self.conv3(self.conv2(self.conv1(x))))
    
class BasicBlock(nn.Module):
    def __init__(self, chi, cho, k=3, s=1, p=None, d=1, g=GROUPS):
        super().__init__()
        self.conv1 = Conv(chi, cho, k, s, p=d, d=d, g=g)
        self.conv2 = Conv(cho, cho, k, 1, p=d, d=d, g=g, act=False)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        return self.relu(residual + self.conv2(self.conv1(x)))

class Root(nn.Module):
    def __init__(self, chi, cho, k, residual):
        super().__init__()
        self.conv = Conv(chi, cho, 1, 1, act=False)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        if self.residual:
            x += children[0]
        return self.relu(x)
    
class Tree(nn.Module):
    def __init__(self, levels, block, chi, cho, s=1,
                 level_root=False, root_dim=0, root_k=1,
                 dilation=1, root_residual=False):
        super().__init__()
        if root_dim == 0:
            root_dim = 2 * cho
        if level_root:
            root_dim += chi
        if levels == 1:
            self.tree1 = block(chi, cho, s=s, d=dilation)
            self.tree2 = block(cho, cho, s=1, d=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, chi, cho, s=s, root_dim=0,
                root_k=root_k, dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, cho, cho, root_dim=root_dim + cho,
                root_k=root_k, dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, cho, root_k, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if s > 1:
            self.downsample = nn.MaxPool2d(s, stride=s)
        if chi != cho:
            self.project = Conv(chi, cho, k=1, s=1, act=False)
                
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
        super().__init__()
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes
        self.base_layer = Conv(3, channels[0], k=7, s=1, p=3)
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], s=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
            level_root=False, root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
            level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
            level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
            level_root=True, root_residual=residual_root)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_conv_level(self, chi, cho, convs, s=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.append(Conv(chi, cho, k=3, s=s if i == 0 else 1,
                p=dilation, d=dilation))
            chi = cho
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        if self.return_levels:
            return y

def dla34(**kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    return model

def dla60(**kwargs):  # DLA-60
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, **kwargs)
    return model

class IDAUp(nn.Module):
    def __init__(self, node_k, out_dim, channels, up_factors):
        super().__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = nn.Identity()
            else:
                proj = Conv(c, out_dim, k=1, s=1)
                    
            f = int(up_factors[i])
            if f == 1:
                up = nn.Identity()
            else:
                up = UpSample(out_dim, out_dim, k=3, 
                    s=1, p=None, d=1, g=out_dim, sf=2)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            

        for i in range(1, len(channels)):
            node = Conv(out_dim * 2, out_dim, k=node_k, s=1)
            setattr(self, 'node_' + str(i), node)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
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
        super().__init__()
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

def fill_head_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class DLASeg(nn.Module):
    def __init__(self, base_name, heads, down_ratio=4, head_conv=256):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.heads = heads
        self.first_level = int(np.log2(down_ratio))
        self.base = globals()[base_name](return_levels=True)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(channels[self.first_level:], scales=scales)

        for head_key in self.heads:
            num_channels = self.heads[head_key]
            head = nn.Sequential(
              nn.Conv2d(channels[self.first_level], head_conv,
                kernel_size=3, padding=1, bias=True),
              nn.ReLU(inplace=True),
              nn.Conv2d(head_conv, num_channels, 
                kernel_size=1, stride=1, 
                padding=0, bias=True))
            if 'hm' in head_key:
                head[-1].bias.data.fill_(-2.19)
            else:
                fill_head_weights(head)
            self.__setattr__(head_key, head)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x[self.first_level:])
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return ret
    
    def fuse(self):  # fuse Conv2d() + BatchNorm2d() layers
        # used to speed up inference
        print('Fusing layers... ')
        self.eval()  # MUST BE in eval mode to work!
        for m in self.children():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.fuse()
        self.info()  # print model information
        return self
    
    def info(self, verbose=False):  
        # print model information
        model_info(self, verbose=verbose)

def get_model(heads, BaseClass=nn.Conv2d, groups=1,
    num_layers=34, head_conv=256, down_ratio=4, pretrained=False):
    global BASE, GROUPS
    BASE = BaseClass
    GROUPS = groups

    assert pretrained == False

    model = DLASeg('dla{}'.format(num_layers), heads,
        down_ratio=down_ratio, head_conv=head_conv)
    return model

@torch.no_grad()
def centernet_test(BaseClass=nn.Conv2d, groups=1, pretrained=False):
    global BASE, GROUPS
    BASE = BaseClass
    GROUPS = groups

    heads_dict = {'cpt_hm': 2, 'cpt_off': 2, 'wh': 2}
    model = get_model(heads_dict, pretrained=False,
        down_ratio=4)
    model.cuda()
    out = model(torch.randn(8, 3, 512, 512).cuda())
    for o in out.values():
        assert o.shape == (8, 2, 128, 128)

if __name__ == '__main__':
    from .utils import (profile, profile_training, 
        init_torch_seeds, model_info)

    def profile_model(heads, BaseClass):
        if 'group' in BaseClass.__name__.lower():
            groups = 8
        else:
            groups = 1
            
        print('profile with groups =', groups)
        model = get_model(heads, BaseClass, groups=groups)
        model_info(model)
        model.cuda()
        profile(model, amp=True)  # test inference speed on GPU
        profile_training(model, amp=True, batch_size=8)

        if not 'group' in BaseClass.__name__.lower():
            print('profile again but with groups =', 8)
            model = get_model(heads, BaseClass, groups=8)
            model_info(model)
            model.cuda()
            profile(model, amp=True)  # test inference speed on GPU
            profile_training(model, amp=True, batch_size=8)

    # to get a feeling of the realtive error when used 
    # on a pretrained ResNet model
    print('Fuse test on ResNet:')
    test_fuse_conv_and_bn()

    heads = {"cpt_hm": 30, "cpt_off": 2, "wh": 2}
    init_torch_seeds(seed=1234, verbose=True)

    # test all the implementations
    for BaseClass in __BaseClasses:
        print('==>', BaseClass.__name__)
        try:
            test_base(BaseClass); print('Passed: test_base')
            test_fuse(BaseClass); print('Passed: test_fuse')
            centernet_test(BaseClass); print('Passed: centernet_test')
            profile_model(heads, BaseClass); print('Passed: profile_model')
        except Exception as e:
            print(e)
            # reset globals
            BASE = nn.Conv2d
            GROUPS = 1
    
    # nn.Conv2d with batch size of 16, the memory 
    # increased (before 8.896118784GB) relative to original model
    """  
    PyTorch version 1.7.1
    CUDA version 11.0
    cuDNN version 8005
    cuDNN deterministic False
    cuDNN benchmark True
    DLASeg: 310 layers, 17.9M parameters, 17.9M gradients, 62.0GFLOPs
    benchmark warm up...
    Forward time: 11.016ms (cuda) @ input: torch.Size([1, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 45.518ms (cuda) @ input: torch.Size([16, 3, 512, 512])
    Backward time: 311.688ms (cuda) @ input: torch.Size([16, 3, 512, 512])
    Maximum of managed memory: 12.188647424GB
    """ 
    
    # run with different types of convolutions and compare 
    # with batch size of 8 
    """
    Fuse test on ResNet:
    fuse relative error: 0.00000030
    PyTorch version 1.7.1
    CUDA version 11.0
    cuDNN version 8005
    cuDNN deterministic False
    cuDNN benchmark True
    ==> Conv2d
    benchmark warm up...
    Number of parameters: 2336 (100.00%)
    Forward + Backward time: 9.156ms
    Passed: test_base
    fuse relative error: 0.00000019
    Passed: test_fuse
    Passed: centernet_test
    profile with groups = 1
    DLASeg: 310 layers, 17.9M parameters, 17.9M gradients, 62.0GFLOPs
    benchmark warm up...
    Forward time: 11.036ms (cuda) @ input: torch.Size([1, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 27.397ms (cuda) @ input: torch.Size([8, 3, 512, 512])
    Backward time: 176.181ms (cuda) @ input: torch.Size([8, 3, 512, 512])
    Maximum of managed memory: 8.873050112GB
    profile again but with groups = 8
    DLASeg: 310 layers, 17.9M parameters, 17.9M gradients, 62.0GFLOPs
    benchmark warm up...
    Forward time: 10.928ms (cuda) @ input: torch.Size([1, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 27.434ms (cuda) @ input: torch.Size([8, 3, 512, 512])
    Backward time: 176.293ms (cuda) @ input: torch.Size([8, 3, 512, 512])
    Maximum of managed memory: 8.873050112GB
    Passed: profile_model
    ==> FlattenedConv
    benchmark warm up...
    Number of parameters: 544 (23.29%)
    Forward + Backward time: 33.868ms
    Passed: test_base
    fuse relative error: 0.00000009
    Passed: test_fuse
    Passed: centernet_test
    profile with groups = 1
    DLASeg: 481 layers, 3.86M parameters, 3.86M gradients, 24.3GFLOPs
    benchmark warm up...
    Forward time: 15.215ms (cuda) @ input: torch.Size([1, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 36.173ms (cuda) @ input: torch.Size([8, 3, 512, 512])
    Backward time: 265.118ms (cuda) @ input: torch.Size([8, 3, 512, 512])
    Maximum of managed memory: 4.490002432GB
    profile again but with groups = 8
    DLASeg: 481 layers, 3.86M parameters, 3.86M gradients, 24.3GFLOPs
    benchmark warm up...
    Forward time: 15.251ms (cuda) @ input: torch.Size([1, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 36.172ms (cuda) @ input: torch.Size([8, 3, 512, 512])
    Backward time: 265.214ms (cuda) @ input: torch.Size([8, 3, 512, 512])
    Maximum of managed memory: 7.545552896GB
    Passed: profile_model
    ==> DeformConv
    benchmark warm up...
    Number of parameters: 3650 (156.25%)
    Forward + Backward time: 72.059ms
    Passed: test_base
    fuse relative error: 0.00000023
    Passed: test_fuse
    Passed: centernet_test
    profile with groups = 1
    DLASeg: 424 layers, 19.1M parameters, 19.1M gradients, 32.1GFLOPs
    benchmark warm up...
    Forward time: 47.221ms (cuda) @ input: torch.Size([1, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 154.439ms (cuda) @ input: torch.Size([8, 3, 512, 512])
    Backward time: 1850.625ms (cuda) @ input: torch.Size([8, 3, 512, 512])
    Maximum of managed memory: 11.771314176GB
    profile again but with groups = 8
    DLASeg: 424 layers, 19.1M parameters, 19.1M gradients, 32.1GFLOPs
    benchmark warm up...
    Forward time: 47.494ms (cuda) @ input: torch.Size([1, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 153.963ms (cuda) @ input: torch.Size([8, 3, 512, 512])
    Backward time: 1850.163ms (cuda) @ input: torch.Size([8, 3, 512, 512])
    Maximum of managed memory: 11.773411328GB
    Passed: profle_model
    ==> SpatiallyConv
    benchmark warm up...
    Number of parameters: 1000 (42.81%)
    Forward + Backward time: 14.073ms
    Passed: test_base
    fuse relative error: 0.00000015
    Passed: test_fuse
    Passed: centernet_test
    profile with groups = 1
    DLASeg: 424 layers, 16.6M parameters, 16.6M gradients, 58.4GFLOPs
    benchmark warm up...
    Forward time: 14.500ms (cuda) @ input: torch.Size([1, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 36.106ms (cuda) @ input: torch.Size([8, 3, 512, 512])
    Backward time: 324.715ms (cuda) @ input: torch.Size([8, 3, 512, 512])
    Maximum of managed memory: 13.918797824GB
    profile again but with groups = 8
    DLASeg: 424 layers, 16.6M parameters, 16.6M gradients, 58.4GFLOPs
    benchmark warm up...
    Forward time: 14.491ms (cuda) @ input: torch.Size([1, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 36.095ms (cuda) @ input: torch.Size([8, 3, 512, 512])
    Backward time: 324.762ms (cuda) @ input: torch.Size([8, 3, 512, 512])
    Maximum of managed memory: 13.918797824GB
    Passed: profle_model
    ==> DepthwiseConv
    benchmark warm up...
    Number of parameters: 368 (15.75%)
    Forward + Backward time: 16.629ms
    Passed: test_base
    fuse relative error: 0.00000010
    Passed: test_fuse
    Passed: centernet_test
    profile with groups = 1
    DLASeg: 424 layers, 3.87M parameters, 3.87M gradients, 23.3GFLOPs
    benchmark warm up...
    Forward time: 12.635ms (cuda) @ input: torch.Size([1, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 26.310ms (cuda) @ input: torch.Size([8, 3, 512, 512])
    Backward time: 217.626ms (cuda) @ input: torch.Size([8, 3, 512, 512])
    Maximum of managed memory: 5.150605312GB
    profile again but with groups = 8
    DLASeg: 424 layers, 3.87M parameters, 3.87M gradients, 23.3GFLOPs
    benchmark warm up...
    Forward time: 12.675ms (cuda) @ input: torch.Size([1, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 26.314ms (cuda) @ input: torch.Size([8, 3, 512, 512])
    Backward time: 217.354ms (cuda) @ input: torch.Size([8, 3, 512, 512])
    Maximum of managed memory: 6.06076928GB
    Passed: profle_model
    ==> FlattenedConv
    benchmark warm up...
    Number of parameters: 544 (23.29%)
    Forward + Backward time: 33.718ms
    Passed: test_base
    fuse relative error: 0.00000010
    Passed: test_fuse
    Passed: centernet_test
    profile with groups = 1
    Not supported!
    ==> ShuffledGroupedConv
    benchmark warm up...
    Number of parameters: 320 (13.70%)
    Forward + Backward time: 16.922ms
    Passed: test_base
    fuse relative error: 0.00000019
    Passed: test_fuse
    Passed: centernet_test
    profile with groups = 8
    DLASeg: 367 layers, 17.9M parameters, 17.9M gradients, 62.0GFLOPs
    benchmark warm up...
    Forward time: 11.942ms (cuda) @ input: torch.Size([1, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 27.296ms (cuda) @ input: torch.Size([8, 3, 512, 512])
    Backward time: 178.556ms (cuda) @ input: torch.Size([8, 3, 512, 512])
    Maximum of managed memory: 6.765412352GB
    Passed: profle_model
    """
    