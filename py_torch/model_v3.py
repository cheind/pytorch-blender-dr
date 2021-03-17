"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
"""
from py_torch.utils import autopad
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os, contextlib
import thop 
from copy import deepcopy
from typing import List, Dict
from torch.nn.functional import interpolate

from .utils import autopad, model_info

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x    
   
class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        
        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)
        
        x += self.shortcut(residual)
        return x

def load_model(model, model_path):
    # the model state dictionary we want to load 
    state_dict_ = torch.load(model_path, map_location='cpu')

    state_dict = {}  # build new model state dictionary

    # convert data parallel to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]

    # the model state dictionary we actually have 
    model_state_dict = model.state_dict()

    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '\
                    'loaded shape{}.'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape))
                # if same key but different shape use current model weight
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
            
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    return model

class GhostNet(nn.Module):
    '''
    When loading the pretrained weights trained on the image 
    classification task it is fine to drop following weights:
    Drop parameter conv_head.weight.
    Drop parameter conv_head.bias.
    Drop parameter classifier.weight.
    Drop parameter classifier.bias.
    When using the 'load_model' function!
    '''
    def __init__(self, cfgs, width=1.0, pretrained=True):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.pretrained = pretrained

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        blocks = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                              se_ratio=se_ratio))
                input_channel = output_channel
            blocks.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        blocks.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))     
        self.blocks = nn.ModuleList(blocks) 

        if pretrained:
            self = load_model(self, './models/state_dict_73.98.pth')

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        ret = []  # return low level feature maps too
        for i, stage in enumerate(self.blocks):
            x = stage(x)
            if i in (2, 4, 6, 9):
                ret.append(x)
        return ret 

class CenternetHeads(nn.Module):

    def __init__(self, heads: Dict[str, int], in_channels: int, 
        head_hidden_channels=256):
        super().__init__()
        
        self.in_channels = in_channels  # num. of input channels to heads 
        self.heads = heads
        self.head_hidden_channels = head_hidden_channels

        for head_name, head_out_channels in heads.items():
            self.create_head(head_name, head_out_channels)

    def forward(self, x):
        return {head: self.__getattr__(head)(x) for head in self.heads}
    
    @staticmethod
    def fill_head_weights(layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def create_head(self, name: str, out_channels: int):
        head = nn.Sequential(
            nn.Conv2d(self.in_channels, self.head_hidden_channels,
                kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_hidden_channels, out_channels,
                kernel_size=1, stride=1, padding=0, bias=True))
        if 'hm' in name:
            head[-1].bias.data.fill_(-2.19)
        else:
            self.fill_head_weights(head)
        self.__setattr__(name, head)

def rescale(x, scale):
    N, C, H, W = x.size()
    # N, W, H, C
    x = x.permute(0, 3, 2, 1)  
    # N, W, H*scale, C/scale
    x.contiguous().view((N, W, H * scale, int(C / scale)))
    # N, H*scale, W, C/scale
    x = x.permute(0, 2, 1, 3)
    # N, H*scale, W*scale, C/(scale**2)
    x = x.contiguous().view((N, W * scale, H * scale, int(C / scale**2)))
    # N, C/(scale**2), H*scale, W*scale
    x = x.permute(0, 3, 1, 2)
    return x

class MixDecoder(nn.Module):

    def __init__(self, scale, channels: List[int], down_ratio: int = 4, 
        hidden_channels: int = 256, out_channels=64):
        super().__init__()
        assert len(channels) > 1
        assert down_ratio in [2, 4, 8]

        # how much we have to upsample at the end to get
        # the desired output dimensions according to 'down_ratio'
        self.scale = scale 

        # the Encoder must expose a list of tensors that includes 
        # the last output and some intermediate low level feature maps
        self.channels = channels 

        self.down_ratio = down_ratio
        self.hidden_channels = hidden_channels

        self.low_level_layer = nn.Sequential(
            nn.Conv2d(sum(channels[:-1]), 
                hidden_channels, 
                kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        self.high_level_layer = nn.Sequential(
            nn.Conv2d(channels[-1] + hidden_channels, 
                hidden_channels, 
                kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        self.pre_rescale_layer = nn.Sequential(
            nn.Conv2d(hidden_channels, 
                out_channels * self.scale**2, 
                kernel_size=1, bias=True),
            nn.BatchNorm2d(out_channels * self.scale**2, affine=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: List[torch.Tensor]):
        high_level_features = x[-1]
        low_level_features = x[:-1]

        size = high_level_features.shape[-2:]  # h x w
        low_level_features = [interpolate(llf, size, mode='bilinear', 
            align_corners=True) for llf in low_level_features]

        low_level_features = torch.cat(low_level_features, dim=1)
        low_level_features = self.low_level_layer(low_level_features)

        x = torch.cat((high_level_features, low_level_features), dim=1)
        x = self.high_level_layer(x)

        x = self.pre_rescale_layer(x)
        x = rescale(x, self.scale)
        return x  # bsz x out_channels x h/down_ratio x w/down_ratio

def ghostnet(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [  # input e.g. 512 x 512
        # k, t, c, SE, s 
        # stage1
        [[3,  16,  16, 0, 1]],  # 256 x 256
        # stage2
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],  # 128 x 128
        # stage3
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],  # 64 x 64
        # stage4
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],  # 32 x 32
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]  # 16 x 16
    ]
    return GhostNet(cfgs, **kwargs)

class Conv(nn.Module):
    def __init__(self, chi, cho, k=1, s=1, p=None, g=1, act=True, affine=True):
        super().__init__()
        self.conv = nn.Conv2d(chi, cho, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(cho, affine=affine)

        if act is True:
            self.act = nn.ReLU(inplace=True)
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class FocusedConvLSTM(nn.Module):  
    """
    no forget gate, focus on either external x(t) or
    internal y(t-1) information (less parameter)
    """
    
    def __init__(self, chi, chh, cho):  
        super().__init__()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # we try to keep mean zero and unit variance by affine=False
        self.conv1 = Conv(chi, 16, k=3, affine=False)
        self.conv2 = Conv(16, 32, k=3, s=2, p=1, affine=False)

        # sigmoid as input activation when rare events are interesting
        # otherwise use tanh when fact are for or against something
        # for Z gate
        self.Z  = Conv(32, chh, k=3, act=self.sigmoid)
        self.I = Conv(chh, chh, k=3, act=self.sigmoid)
        self.O = Conv(chh, chh, k=3, act=self.sigmoid)

        self.conv3 = Conv(chh, cho, k=3, affine=False, act=False)

    def reset_states(self, shape):
        device = next(self.parameters()).device
        self.y = torch.zeros(shape, device=device)  # at t-1
        self.c = torch.zeros(shape, device=device)  # at t-1

    def init_weights(self):
        def init(m):
            if type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(init)  # recursively to every submodule

    def forward(self, x):  # bsz x sl x h x w
        x = x.unsqueeze(2)  # bsz x sl x 1 x h x w
        x = x.permute(1, 0, 2, 3, 4)  # sl x bsz x 1 x h x w
        
        for i, x_ in enumerate(x):  # input sequence
            x_ = self.conv1(x_)  # bsz x 1 x h x w
            x_ = self.conv2(x_)  # bsz x 1 x h/2 x w/2

            z = self.Z(x_)  # has hidden cell dimension
            if i == 0:  # initialize self.c, self.y
                self.reset_states(z.shape)  # bsz x chh x h x w

            i = self.I(self.y)  # input gate, feed y(t-1)
            o = self.O(self.y)  # output gate
            self.c += i * z  # memory cell state
            self.y = o * self.tanh(self.c)  # lstm output

        # upsample last lstm output 
        y = interpolate(self.y, scale_factor=2, mode='nearest')
        y = self.conv3(y)
        return y

class Model(nn.Module):

    def __init__(self, reducer, encoder, decoder, classifier):
        super().__init__()
        # may we have a sequence as input which we want to reduce to 
        # a single instance for the encoder
        self.reducer = reducer
        # the encoder maps a single input instance to high level features
        # plus low level feature maps for the decoder to merge and process
        self.encoder = encoder
        # the decoder expects a sequence of feature maps from the encoder
        self.decoder = decoder
        # the classifier expects a feature map to build the heads on top
        assert classifier is not None
        self.classifier = classifier  # returns dictionary

    def forward(self, x):
        x = self.reducer(x) if self.reducer is not None else x
        x = self.encoder(x) if self.encoder is not None else x
        x = self.decoder(x) if self.decoder is not None else x
        x = self.classifier(x)  # mandatory
        return x  # dictionary

    def info(self, in_channels, in_height, in_width, verbose=False):
        if verbose:  # show incremental complexity gain of overall model
            modules = []
            n_layers_prev, n_p_prev, n_g_prev, flops_prev = 0, 0, 0, 0
            for module in (self.reducer, self.encoder, self.decoder, self.classifier):
                if module is not None:
                    modules.append(module)
                    model = nn.Sequential(*modules)
                    
                    # surpress print statements
                    with open(os.devnull, 'w') as devnull:
                        with contextlib.redirect_stdout(devnull):
                            n_layers_cur, n_p_cur, n_g_cur, flops_cur = model_info(model, in_channels, in_height, in_width)
                            
                    n_layers = n_layers_cur - n_layers_prev
                    n_p = n_p_cur - n_p_prev
                    n_g = n_g_cur - n_g_prev
                    flops = flops_cur - flops_prev
                    n_layers_prev, n_p_prev, n_g_prev, flops_prev = n_layers_cur, n_p_cur, n_g_cur, flops_cur

                    print(f'+ {module.__class__.__name__}: {n_layers} layers, {n_p/10**6:0.3}M parameters, {n_g/10**6:0.3}M gradients, {flops:.1f}GFLOPs')
                
            print('= ', end='')
            model_info(self, in_channels, in_height, in_width)
        else:
            model_info(self, in_channels, in_height, in_width)
        
def get_scale(encoder, in_channels, in_height, in_width, down_ratio):
    # scale for upsampling lowest stage resolution
    # to obtain feature maps expected by 'down_ratio'
    with torch.no_grad():
        out = encoder(torch.randn(1, in_channels, in_height, in_width))
    scale = int(in_width / down_ratio / out[-1].size(-1))
    return scale

def encoder_info(encoder, in_channels, in_height, in_width, verbose=False):
    x = torch.randn(1, in_channels, in_height, in_width)
    out = encoder(x)
    assert isinstance(out, list) or isinstance(out, tuple), 'Encoder must provide multiple outputs!'

    if verbose:
        print(f'{encoder.__class__.__name__} intermediate feature maps:')
        print('\n'.join([str(o.shape) for o in out]), end='\n\n')
    
    channels = [o.size(1) for o in out]
    return channels

def get_model(heads, head_conv=256, down_ratio=4, 
    pretrained=True, input_size=(512, 512)):
    
    encoder = ghostnet()
    if pretrained:
        model_path = './models/state_dict_73.98.pth'
        assert os.path.exists(model_path)
        load_model(encoder, model_path)

    channels = encoder_info(encoder, 3, *input_size)
    scale = get_scale(encoder, 3, *input_size, down_ratio=down_ratio)
    decoder = MixDecoder(scale, channels, down_ratio=down_ratio, 
        hidden_channels=256, out_channels=64)

    classifier = CenternetHeads(heads, in_channels=64,
        head_hidden_channels=head_conv)

    model = Model(None, encoder, decoder, classifier)
    return model

@torch.no_grad()
def model_test(pretrained=False):
    heads_dict = {'cpt_hm': 2, 'cpt_off': 2, 'wh': 2}
    model = get_model(heads_dict, pretrained=pretrained, down_ratio=4)
    model.cuda()
    out = model(torch.randn(8, 3, 512, 512).cuda())
    for o in out.values():
        assert o.shape == (8, 2, 128, 128)

if __name__ == '__main__':
    from .utils import (profile, profile_training, 
        init_torch_seeds)
    
    init_torch_seeds(seed=1234, verbose=True)
    input_size=(512, 512)
    in_channels =3
    heads = {'cpt_hm': 30, 'cpt_off': 2, 'wh': 2}
    model = get_model(heads, input_size=input_size)
    model.info(in_channels, *input_size, verbose=True)

    model.cuda()
    profile(model, amp=True)  # test inference speed on GPU
    profile_training(model, amp=True, batch_size=8)
    profile_training(model, amp=True, batch_size=32)

    """
    PyTorch version 1.7.1
    CUDA version 11.0
    cuDNN version 8005
    cuDNN deterministic False
    cuDNN benchmark True
    Drop parameter conv_head.weight.
    Drop parameter conv_head.bias.
    Drop parameter classifier.weight.
    Drop parameter classifier.bias.
    Drop parameter conv_head.weight.
    Drop parameter conv_head.bias.
    Drop parameter classifier.weight.
    Drop parameter classifier.bias.
    + GhostNet: 403 layers, 2.67M parameters, 2.67M gradients, 1.5GFLOPs
    + MixDecoder: 13 layers, 1.41M parameters, 1.41M gradients, 0.7GFLOPs
    + CenternetHeads: 13 layers, 0.452M parameters, 0.452M gradients, 14.8GFLOPs
    = Model: 429 layers, 4.53M parameters, 4.53M gradients, 17.0GFLOPs
    benchmark warm up...
    Forward time: 16.976ms (cuda) @ input: torch.Size([1, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 18.398ms (cuda) @ input: torch.Size([8, 3, 512, 512])
    Backward time: 145.027ms (cuda) @ input: torch.Size([8, 3, 512, 512])
    Maximum of managed memory: 2.713714688GB
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 52.694ms (cuda) @ input: torch.Size([32, 3, 512, 512])
    Backward time: 425.192ms (cuda) @ input: torch.Size([32, 3, 512, 512])
    Maximum of managed memory: 9.261023232GB
    """