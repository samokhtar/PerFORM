import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import IterableDataset
from torch.func import functional_call
#from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import weight_norm
#from torch.nn.utils.stateless import functional_call 
from math import pi

from building_sdf.learning.network_loss import *
from building_sdf.learning.network_utils import *

from functools import partial
import functools
from collections import OrderedDict
from torch import Tensor
from typing import Any, Callable, List, Optional, Type, Union, Tuple
from torchvision.models.vision_transformer import ConvStemConfig, MLPBlock, EncoderBlock, Encoder
from torchvision.models.densenet import _DenseLayer, _DenseBlock, _Transition
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1, conv3x3



# From https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
# 'unet_128':  net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
# 'unet_256':  net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

        
class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, outermost_nonlinearity='tanh'):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        if(num_downs >= 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
            for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
                unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
            # gradually reduce the number of filters from ngf * 8 to ngf
            unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        elif(num_downs == 4):
            unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
            unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        elif(num_downs == 3):
            unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
            unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        else:
            print('num_downs is too small')
        
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, outermost_nonlinearity=outermost_nonlinearity)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, outermost_nonlinearity='tanh'):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        # Dictionary that maps nonlinearity name to the respective function
        nls = {'sine':Sine(),'relu':nn.ReLU(inplace=True),'leakyrelu':nn.LeakyReLU(inplace=True, negative_slope = 0.2),
               'sigmoid':nn.Sigmoid(), 'tanh':nn.Tanh(), 'selu':nn.SELU(inplace=True),'softplus':nn.Softplus(),
                         'elu':nn.ELU(inplace=True),'softmax':nn.Softmax(dim=-1)}

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nls[outermost_nonlinearity]]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)




# Based on pytorch classes
# https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
# https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html

class Transformer(nn.Module):
    def __init__(
        self,
        grid_size: int,
        mode: str = '2d',
        nhead: int = 16,
        dim_feedforward: int = 64,
        num_layers: int = 16,
        dropout: float = 0,
        norm_first: bool = False,
        outermost_linear: bool = True,
        outermost_nonlinearity: str = 'sigmoid',
        out_dim: int = 1,
        apply_fc = False,
    ) -> None:
        super().__init__()
        self.apply_fc = apply_fc
        
        # Dictionary that maps nonlinearity name to the respective function, initialization
        nls_and_inits = {'sine':(Sine(), sine_init), 
                         'relu':(nn.ReLU(inplace=True), init_weights_normal),
                         'leakyrelu':(nn.LeakyReLU(inplace=True, negative_slope = 0.2), init_weights_normal),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier),
                         'tanh':(nn.Tanh(), init_weights_xavier),
                         'selu':(nn.SELU(inplace=True), init_weights_selu),
                         'softplus':(nn.Softplus(), init_weights_normal),
                         'elu':(nn.ELU(inplace=True), init_weights_elu),
                         'softmax':(nn.Softmax(dim=-1), init_weights_xavier)}

        outermost_nl = nls_and_inits[outermost_nonlinearity][0]
        outermost_nl_weight_init = nls_and_inits[outermost_nonlinearity][1]
        
        self.pos_enc = CoordCat(mode)
        d_model = (grid_size*grid_size) if(mode=='2d') else (grid_size*grid_size*grid_size)
        layernorm = torch.nn.LayerNorm(d_model)       
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout,
                                                   norm_first=norm_first)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=layernorm)
        
        # Output layer
        if(self.apply_fc):
            output_layers = []
            output_layers.append(nn.Linear(d_model, out_dim))
            if(not outermost_linear):
                output_layers.append(outermost_nl)
            self.fc = nn.Sequential(*output_layers)
            if(not outermost_linear):
                self.fc.apply(outermost_nl_weight_init)       
        
    def forward(self, x: Tensor) -> Tensor:
        x_pos = self.pos_enc(x)
        x_pos = x_pos.reshape(x_pos.shape[0],x_pos.shape[1],-1)
        out = self.transformer_encoder(x_pos)
        out = out.mean(axis=-2)
        if(self.apply_fc):
            out = self.fc(out)
        return out


# Modified from https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
# DenseNet types
# densenet201: (32, (6, 12, 48, 32), 64)
# densenet169: (32, (6, 12, 32, 32), 64)
# densenet161: (48, (6, 12, 36, 24), 96)
# densenet121: ((32, (6, 12, 24, 16), 64)

class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        num_channels (int) - how many channels in 2D map
        output_dim (int) - size of the output
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
        outermost_linear (bool) - True if the last layer is linear
        outermost_nonlinearity (str) - nonlinearity type of the final layer
    """

    def __init__(
        self,
        n_channels: int,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        memory_efficient: bool = False,
        outermost_linear: bool = True,
        outermost_nonlinearity: str = 'sigmoid',
        out_dim: int = 1,
        apply_fc = False,
    ) -> None:

        super().__init__()
        self.apply_fc = apply_fc
        
        # Dictionary that maps nonlinearity name to the respective function, initialization
        nls_and_inits = {'sine':(Sine(), sine_init), 
                         'relu':(nn.ReLU(inplace=True), init_weights_normal),
                         'leakyrelu':(nn.LeakyReLU(inplace=True, negative_slope = 0.2), init_weights_normal),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier),
                         'tanh':(nn.Tanh(), init_weights_xavier),
                         'selu':(nn.SELU(inplace=True), init_weights_selu),
                         'softplus':(nn.Softplus(), init_weights_normal),
                         'elu':(nn.ELU(inplace=True), init_weights_elu),
                         'softmax':(nn.Softmax(dim=-1), init_weights_xavier)}

        outermost_nl = nls_and_inits[outermost_nonlinearity][0]
        outermost_nl_weight_init = nls_and_inits[outermost_nonlinearity][1]

        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(n_channels, num_init_features, kernel_size=5, stride=1, padding=3, bias=False)),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )
        

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # Output layer
        if(self.apply_fc):
            output_layers = []
            output_layers.append(nn.Linear(num_features, out_dim))
            if(not outermost_linear):
                output_layers.append(outermost_nl)
            self.fc = nn.Sequential(*output_layers)
            if(not outermost_linear):
                self.fc.apply(outermost_nl_weight_init)       

            # Official init from torch repo.
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        if(self.apply_fc):
            out = self.fc(out)
        return out


# Modified from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
# ResNet types
# wide_resnet101_2: (Bottleneck, [3, 4, 23, 3])
# wide_resnet50_2: (Bottleneck, [3, 4, 6, 3])
# resnext101_64x4d: (Bottleneck, [3, 4, 23, 3])
# resnext101_32x8d: (Bottleneck, [3, 4, 23, 3])
# resnext50_32x4d: (Bottleneck, [3, 4, 6, 3])
# resnet152: (Bottleneck, [3, 8, 36, 3])
# resnet101: (Bottleneck, [3, 4, 23, 3])
# resnet50: (Bottleneck, [3, 4, 6, 3])
# resnet34: (BasicBlock, [3, 4, 6, 3])
# resnet18: (BasicBlock, [2, 2, 2, 2])

class ResNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        block_type: str,
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        outermost_linear: bool = True,
        outermost_nonlinearity: str = 'sigmoid',
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        out_dim: int = 1,
        apply_fc: bool = False,

    ) -> None:
        super().__init__()
        block = BasicBlock if(block_type == 'basicblock') else Bottleneck
        self.apply_fc = apply_fc
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
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
            
        # Dictionary that maps nonlinearity name to the respective function, initialization
        nls_and_inits = {'sine':(Sine(), sine_init), 
                         'relu':(nn.ReLU(inplace=True), init_weights_normal),
                         'leakyrelu':(nn.LeakyReLU(inplace=True, negative_slope = 0.2), init_weights_normal),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier),
                         'tanh':(nn.Tanh(), init_weights_xavier),
                         'selu':(nn.SELU(inplace=True), init_weights_selu),
                         'softplus':(nn.Softplus(), init_weights_normal),
                         'elu':(nn.ELU(inplace=True), init_weights_elu),
                         'softmax':(nn.Softmax(dim=-1), init_weights_xavier)}

        outermost_nl = nls_and_inits[outermost_nonlinearity][0]
        outermost_nl_weight_init = nls_and_inits[outermost_nonlinearity][1]
            
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(n_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if(self.apply_fc):
            output_layers = []
            output_layers.append(nn.Linear(512 * block.expansion, out_dim))
            if(not outermost_linear):
                output_layers.append(outermost_nl)
            self.fc = nn.Sequential(*output_layers)
            if(not outermost_linear):
                self.fc.apply(outermost_nl_weight_init)        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
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
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if(self.apply_fc):
            x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


# Modified from https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
# ViT types
#(patch_size,num_layers,num_heads,hidden_dim,mlp_dim)
# vit_h_14: (14,32,16,1280,5120)
# vit_l_32: (32,24,16,1024,4096)
# vit_l_16: (16,24,16,1024,4096)
# vit_b_32: (32,12,12,768,3072)
# vit_b_16: (16,12,12,768,3072)

class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        n_channels: int,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        outermost_linear: bool = True,
        outermost_nonlinearity: str = 'sigmoid',
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
        out_dim: int = 1,
        apply_fc = False,
    ):
        super().__init__()
        self.apply_fc = apply_fc
        # Dictionary that maps nonlinearity name to the respective function, initialization
        nls_and_inits = {'sine':(Sine(), sine_init), 
                         'relu':(nn.ReLU(inplace=True), init_weights_normal),
                         'leakyrelu':(nn.LeakyReLU(inplace=True, negative_slope = 0.2), init_weights_normal),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier),
                         'tanh':(nn.Tanh(), init_weights_xavier),
                         'selu':(nn.SELU(inplace=True), init_weights_selu),
                         'softplus':(nn.Softplus(), init_weights_normal),
                         'elu':(nn.ELU(inplace=True), init_weights_elu),
                         'softmax':(nn.Softmax(dim=-1), init_weights_xavier)}

        outermost_nl = nls_and_inits[outermost_nonlinearity][0]
        outermost_nl_weight_init = nls_and_inits[outermost_nonlinearity][1]

        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = nn.Conv2d(
                in_channels=n_channels, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        seq_length = (image_size // patch_size) ** 2

        # Add an output token
        self.output_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        if(self.apply_fc):
            heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
            if representation_size is None:
                output_layers = []
                output_layers.append(nn.Linear(hidden_dim, out_dim))
                if(not outermost_linear):
                    output_layers.append(outermost_nl)
                out_layer = nn.Sequential(*output_layers)
                if(not outermost_linear):
                    out_layer.apply(outermost_nl_weight_init)     
                heads_layers["head"] = out_layer
            else:
                heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
                heads_layers["act"] = nn.Tanh()
                output_layers = []
                output_layers.append(nn.Linear(representation_size, out_dim))
                if(not outermost_linear):
                    output_layers.append(outermost_nl)
                out_layer = nn.Sequential(*output_layers)
                if(not outermost_linear):
                    out_layer.apply(outermost_nl_weight_init)     
                heads_layers["head"] = output_layers

            self.heads = nn.Sequential(heads_layers)

            if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
                fan_in = self.heads.pre_logits.in_features
                nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
                nn.init.zeros_(self.heads.pre_logits.bias)

            if isinstance(self.heads.head, nn.Linear):
                nn.init.zeros_(self.heads.head.weight)
                nn.init.zeros_(self.heads.head.bias)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        output_class_token = self.output_token.expand(n, -1, -1)
        x = torch.cat([output_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        if(self.apply_fc):
            x = self.heads(x)

        return x



# MLP adapted from https://github.com/vsitzmann/siren/blob/master/modules.py
class MLP(nn.Module):
    def __init__(
        self, 
        mlp_param, 
        input_dim=1, 
        output_dim=1, 
    ):
        super(MLP, self).__init__()

        self.first_layer_init = None
        self.skip_conn = mlp_param['skip_conn']

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init), #'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'leakyrelu':(nn.LeakyReLU(inplace=True, negative_slope = 0.2), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None),
                         'softmax':(nn.Softmax(dim=-1), init_weights_xavier, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[mlp_param['nonlinearity']]
        outermost_nl = nls_and_inits[mlp_param['outermost_nonlinearity']][0]
        outermost_nl_weight_init = nls_and_inits[mlp_param['outermost_nonlinearity']][1]

        if mlp_param['weight_init'] is not None:  # Overwrite weight init if passed
            self.weight_init = mlp_param['weight_init']
            self.weight_init_outmost = mlp_param['weight_init']
        else:
            self.weight_init = nl_weight_init
            self.weight_init_outmost = outermost_nl_weight_init

        self.first_layer_init = first_layer_init
        
        # Add missing parameters
        mlp_param['weight_norm'] = mlp_param['weight_norm'] if('weight_norm' in mlp_param.keys()) else False
        mlp_param['lipschitz_norm'] = mlp_param['lipschitz_norm'] if('lipschitz_norm' in mlp_param.keys()) else False

        # Define input layer
        input_layers = []
        input_layer_outputdim = mlp_param['n_hidden_neurons'][0] if(isinstance(mlp_param['n_hidden_neurons'], list)) else mlp_param['n_hidden_neurons']
        if(mlp_param['weight_norm'] and (not mlp_param['lipschitz_norm'])):
            #input_layers.append(WeightNorm(nn.Linear(in_features=input_dim, out_features=mlp_param['n_hidden_neurons']),['weight']))
            input_layers.append(weight_norm(nn.Linear(in_features=input_dim, out_features=input_layer_outputdim)))
        elif(mlp_param['lipschitz_norm']):
            #input_layers.append(LipWeightNorm(nn.Linear(in_features=input_dim, out_features=mlp_param['n_hidden_neurons'])))
            input_layers.append(LipschitzLinear(in_features=input_dim, out_features=input_layer_outputdim))
        else:
            input_layers.append(nn.Linear(in_features=input_dim, out_features=input_layer_outputdim))
        input_layers.append(nl)

        self.layer_in = nn.Sequential(*input_layers)
        if(self.first_layer_init is not None):
            self.layer_in.apply(self.first_layer_init)
        else:
            if(not mlp_param['lipschitz_norm']):
                self.layer_in.apply(self.weight_init)

        if(mlp_param['skip_conn'] == False):
            # Define intermediate layers without skip connections
            hidden_layers = []
            for i in range(mlp_param['n_hidden_layers']):
                h_layer_inputdim = mlp_param['n_hidden_neurons'][i] if(isinstance(mlp_param['n_hidden_neurons'], list)) else mlp_param['n_hidden_neurons'] 
                h_layer_outputdim = mlp_param['n_hidden_neurons'][i+1] if(isinstance(mlp_param['n_hidden_neurons'], list)) else mlp_param['n_hidden_neurons']                                            
                if(mlp_param['weight_norm'] and (not mlp_param['lipschitz_norm'])):
                    #hidden_layers.append(WeightNorm(nn.Linear(in_features=mlp_param['n_hidden_neurons'], out_features=mlp_param['n_hidden_neurons']),['weight']))
                    hidden_layers.append(weight_norm(nn.Linear(in_features=h_layer_inputdim, out_features=h_layer_outputdim)))
                elif(mlp_param['lipschitz_norm']):
                    #hidden_layers.append(LipWeightNorm(nn.Linear(in_features=mlp_param['n_hidden_neurons'], out_features=mlp_param['n_hidden_neurons'])))
                    hidden_layers.append(LipschitzLinear(in_features=h_layer_inputdim, out_features=h_layer_outputdim))
                else:
                    hidden_layers.append(nn.Linear(in_features=h_layer_inputdim, out_features=h_layer_outputdim))
                hidden_layers.append(nl)
                if(mlp_param['dropout']):
                    hidden_layers.append(nn.Dropout(p=float(mlp_param['dropout_prob'])))
            self.layers_mid = nn.Sequential(*hidden_layers)
            if(not mlp_param['lipschitz_norm']):
                self.layers_mid.apply(self.weight_init)

        else:
            # Define intermediate layers with skip connections
            hidden_layers_skip = []
            for i in range(mlp_param['n_hidden_layers']):
                h_layer_inputdim = mlp_param['n_hidden_neurons'][i] if(isinstance(mlp_param['n_hidden_neurons'], list)) else mlp_param['n_hidden_neurons'] 
                h_layer_outputdim = mlp_param['n_hidden_neurons'][i+1] if(isinstance(mlp_param['n_hidden_neurons'], list)) else mlp_param['n_hidden_neurons']                                              
                if(i%mlp_param['skip_conn_int']==0):
                    h_lay_set = []
                out_lay_d = h_layer_outputdim
                if(((i%mlp_param['skip_conn_int']) == (mlp_param['skip_conn_int']-1)) and (i != (mlp_param['n_hidden_layers']-1))):
                    out_lay_d = h_layer_outputdim-input_dim #if((h_layer_outputdim-input_dim)>0) else (h_layer_outputdim+input_dim)
                if(mlp_param['weight_norm'] and (not mlp_param['lipschitz_norm'])):
                    #h_lay_set.append(WeightNorm(nn.Linear(in_features=mlp_param['n_hidden_neurons'], out_features=out_lay_d),['weight']))
                    h_lay_set.append(weight_norm(nn.Linear(in_features=h_layer_inputdim, out_features=out_lay_d)))
                elif(mlp_param['lipschitz_norm']):
                    #h_lay_set.append(LipWeightNorm(nn.Linear(in_features=mlp_param['n_hidden_neurons'], out_features=out_lay_d)))
                    h_lay_set.append(LipschitzLinear(in_features=h_layer_inputdim, out_features=out_lay_d))
                else:
                    h_lay_set.append(nn.Linear(in_features=h_layer_inputdim, out_features=out_lay_d))
                h_lay_set.append(nl)
                if(mlp_param['dropout']):
                    h_lay_set.append(nn.Dropout(p=float(mlp_param['dropout_prob'])))
                if((i%mlp_param['skip_conn_int']) == (mlp_param['skip_conn_int']-1)):
                    layer_set = nn.Sequential(*h_lay_set)
                    if(not mlp_param['lipschitz_norm']):
                        layer_set.apply(self.weight_init)
                    
                    hidden_layers_skip.append(layer_set)
            self.hidden_layers_skip = nn.ModuleList(hidden_layers_skip)
            
        # Define output layer
        output_layers = []
        output_layer_inputdim = mlp_param['n_hidden_neurons'][-1] if(isinstance(mlp_param['n_hidden_neurons'], list)) else mlp_param['n_hidden_neurons']
        output_layers.append(nn.Linear(in_features=output_layer_inputdim, out_features=output_dim))
        if(not mlp_param['outermost_linear']):
            output_layers.append(outermost_nl)
        self.layer_out = nn.Sequential(*output_layers)
        if(not mlp_param['lipschitz_norm']):
            self.layer_out.apply(self.weight_init_outmost)

    def forward(self, x):
        
        y = self.layer_in(x)
      
        if(self.skip_conn):
            for i, l in enumerate(self.hidden_layers_skip):
                if(i == 0):
                    y = self.hidden_layers_skip[i](y)
                else:
                    #print(torch.cat([x,y], dim=-1).shape)
                    y = self.hidden_layers_skip[i](torch.cat([x,y], dim=-1))
        else:
            y = self.layers_mid(y)

        y = self.layer_out(y)

        return y

class MLP_Simple(nn.Module):
    def __init__(
        self, 
        mlp_param, 
        input_dim=1, 
        output_dim=1, 
    ):
        super(MLP_Simple, self).__init__()

        self.first_layer_init = None
        self.mlp_param = mlp_param

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'leakyrelu':(nn.LeakyReLU(inplace=True, negative_slope = 0.2), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None),
                         'softmax':(nn.Softmax(dim=-1), init_weights_xavier, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[mlp_param['nonlinearity']]
        outermost_nl = nls_and_inits[mlp_param['outermost_nonlinearity']][0]
        outermost_nl_weight_init = nls_and_inits[mlp_param['outermost_nonlinearity']][1]

        self.weight_init = nl_weight_init
        self.weight_init_outmost = outermost_nl_weight_init
        self.first_layer_init = first_layer_init

        # Define input layer
        input_layers = []
        if(isinstance(mlp_param['n_hidden_neurons'], list)):
            input_layers.append(nn.Linear(in_features=input_dim, out_features=mlp_param['n_hidden_neurons'][0]))
        else:
            input_layers.append(nn.Linear(in_features=input_dim, out_features=mlp_param['n_hidden_neurons']))
        input_layers.append(nl)
        self.layer_in = nn.Sequential(*input_layers)
        if(self.first_layer_init is not None):
            self.layer_in.apply(self.first_layer_init)
        else:
            self.layer_in.apply(self.weight_init)

        # Define intermediate layers 
        if(mlp_param['n_hidden_layers'] != 0):
            hidden_layers = []
            for i in range(mlp_param['n_hidden_layers']):
                if(isinstance(mlp_param['n_hidden_neurons'], list)):
                    hidden_layers.append(nn.Linear(in_features=mlp_param['n_hidden_neurons'][i], out_features=mlp_param['n_hidden_neurons'][i+1]))
                else:
                    hidden_layers.append(nn.Linear(in_features=mlp_param['n_hidden_neurons'], out_features=mlp_param['n_hidden_neurons']))
                hidden_layers.append(nl)
            self.layers_mid = nn.Sequential(*hidden_layers)
            self.layers_mid.apply(self.weight_init)
            
        # Define output layer
        output_layers = []
        if(isinstance(mlp_param['n_hidden_neurons'], list)):
            output_layers.append(nn.Linear(in_features=mlp_param['n_hidden_neurons'][-1], out_features=output_dim))
        else:
            output_layers.append(nn.Linear(in_features=mlp_param['n_hidden_neurons'], out_features=output_dim))
        if(not mlp_param['outermost_linear']):
            output_layers.append(outermost_nl)
        self.layer_out = nn.Sequential(*output_layers)
        self.layer_out.apply(self.weight_init_outmost)

    def forward(self, x):

        y = self.layer_in(x)
        if(self.mlp_param['n_hidden_layers'] != 0):
            y = self.layers_mid(y)
        y = self.layer_out(y)

        return y
    

class Autoencoder_MLP(nn.Module):
    def __init__(
        self, 
        autoenc_param, 
    ):
        super(Autoencoder_MLP, self).__init__()
        
        self.autoenc_param = autoenc_param

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'leakyrelu':(nn.LeakyReLU(inplace=True, negative_slope = 0.2), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None),
                         'softmax':(nn.Softmax(dim=-1), init_weights_xavier, None)}

        self.autoenc_mod = ['enc','dec']
        input_dims = {'enc': autoenc_param['enc_feature_dim'] , 'dec': autoenc_param['bottleneck_dim']}
        output_dims = {'enc': autoenc_param['bottleneck_dim'] , 'dec': autoenc_param['dec_feature_dim']}
        
        for e in self.autoenc_mod:
            
            self.first_layer_init = None
            nl, nl_weight_init, first_layer_init = nls_and_inits[autoenc_param[e + '_nonlinearity']]
            outermost_nl = nls_and_inits[autoenc_param[e + '_outermost_nonlinearity']][0]
            outermost_nl_weight_init = nls_and_inits[autoenc_param[e + '_outermost_nonlinearity']][1]

            self.weight_init = nl_weight_init
            self.weight_init_outmost = outermost_nl_weight_init
            self.first_layer_init = first_layer_init
            
            # Define input layer
            input_layers = []
            if(isinstance(autoenc_param[e + '_hidden_neurons'], list)):
                input_layers.append(nn.Linear(in_features=input_dims[e], out_features=autoenc_param[e + '_hidden_neurons'][0]))
            else:
                input_layers.append(nn.Linear(in_features=input_dims[e], out_features=autoenc_param[e + '_hidden_neurons']))
            input_layers.append(nl)
            layer_in = nn.Sequential(*input_layers)
            if(self.first_layer_init is not None):
                layer_in.apply(self.first_layer_init)
            else:
                layer_in.apply(self.weight_init)

            # Define intermediate layers 
            if(autoenc_param[e + '_hidden_layers'] != 0):
                hidden_layers = []
                for i in range(autoenc_param[e + '_hidden_layers']):
                    if(isinstance(autoenc_param[e + '_hidden_neurons'], list)):
                        hidden_layers.append(nn.Linear(in_features=autoenc_param[e + '_hidden_neurons'][i], out_features=autoenc_param[e + '_hidden_neurons'][i+1]))
                    else:
                        hidden_layers.append(nn.Linear(in_features=autoenc_param[e + '_hidden_neurons'], out_features=autoenc_param[e + '_hidden_neurons']))
                    hidden_layers.append(nl)
                layers_mid = nn.Sequential(*hidden_layers)
                layers_mid.apply(self.weight_init)

            # Define output layer
            output_layers = []
            if(isinstance(autoenc_param[e + '_hidden_neurons'], list)):
                output_layers.append(nn.Linear(in_features=autoenc_param[e + '_hidden_neurons'][-1], out_features=output_dims[e]))
            else:
                output_layers.append(nn.Linear(in_features=autoenc_param[e + '_hidden_neurons'], out_features=output_dims[e]))
            if(not autoenc_param[e + '_outermost_linear']):
                output_layers.append(outermost_nl)
            layer_out = nn.Sequential(*output_layers)
            layer_out.apply(self.weight_init_outmost)
            
            if(e == 'enc'):
                self.layer_in_enc = layer_in
                self.layers_mid_enc = layers_mid
                self.layer_out_enc = layer_out
            else:
                self.layer_in_dec = layer_in
                self.layers_mid_dec = layers_mid
                self.layer_out_dec = layer_out

    def forward(self, x):

        y = self.layer_in_enc(x)
        if(self.autoenc_param['enc' + '_hidden_layers'] != 0):
            y = self.layers_mid_enc(y)
        y = self.layer_out_enc(y)
        
        y = self.layer_in_dec(y)
        if(self.autoenc_param['dec' + '_hidden_layers'] != 0):
            y = self.layers_mid_dec(y)
        y = self.layer_out_dec(y)

        return y
        
# Adapted from https://github.com/ndahlquist/pytorch-fourier-feature-networks/blob/master/fourier_feature_transform.py
# for points in the shape of batch,num_points,num_dim

class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Given an input of size [batches, num_points, num_input_channels],
     returns a tensor of size [batches, num_points, mapping_size*2].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10, seed=0):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self._B = torch.randn((num_input_channels, mapping_size)) * scale
        #print(self._B)

    def forward(self, x):
        assert x.dim() == 3, 'Expected 3D input (got {}D input)'.format(x.dim())

        batches, num_points, channels = x.shape

        assert channels == self._num_input_channels,\
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, N, M] to [(B*N), M].
        x = x.reshape(batches * num_points, channels)
        #print('x.shape', x.shape)
        #print('x', x)

        x = x @ self._B.to(x.device)
        #print('x.shape', x.shape)
        #print('x', x)

        # From [(B*N), M] to [B, N, M]
        x = x.view(batches, num_points, self._mapping_size)
        #print('x.shape', x.shape)
        #print('x', x)

        x = 2 * pi * x
        #print('x.shape', x.shape)
        #print('x', x)
        #print('x.shape', torch.cat([torch.sin(x), torch.cos(x)], dim=-1).shape)
        # print('total',torch.sum(torch.cat([torch.sin(x), torch.cos(x)], dim=-1)))
        # print('total_B',torch.sum(self._B))
        
        #print('torch.cat([torch.sin(x), torch.cos(x)], dim=-1)',torch.cat([torch.sin(x), torch.cos(x)], dim=-1).shape)
        
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
    

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        torch.nn.init.kaiming_normal_(m.weight)

class CoordCat(nn.Module):
    """
    This class takes an input of shape (B, L, ...) and concatenates normalized 
    coordinates along the latent dimension. This is followed when we need 
    positional information, such as in the case of CoordConv. We will use this 
    when we write the decoder in ConvDecoder. 
    """
    def __init__(self, mode="2d"):
        super().__init__()
        self.mode = mode

    def forward(self, input):
        if self.mode == "2d":
            x = torch.linspace(-1, 1, input.shape[2])
            y = torch.linspace(-1, 1, input.shape[3])
            xy = torch.meshgrid(x, y, indexing='ij')
            xy = torch.stack(xy, dim=0)
            out = xy.repeat(input.shape[0], 1, 1, 1)
        else:
            x = torch.linspace(-1, 1, input.shape[2])
            y = torch.linspace(-1, 1, input.shape[3])
            z = torch.linspace(-1, 1, input.shape[4])
            xyz = torch.meshgrid(x, y, z, indexing='ij')
            xyz = torch.stack(xyz, dim=0)
            out = xyz.repeat(input.shape[0], 1, 1, 1, 1)
        return torch.cat([input, out.to(input.device)], dim=1)

class ConvDecoder(nn.Module):
    def __init__(self, in_ch, hidden_ch, num_up, out_ch, mode='2d', neg_slope=0.2):
        super().__init__()
        
        """
        In this function, we will construct a convolutional decoder based on the 
        input mode and other input parameters specifying the input, hidden, and
        output channels. num_up specifies the number of layers in the decoder.
        
        For mode is '2d', we use nn.Conv2d to perform the initial and final 
        convolutions, and nn.Conv3d is used when mode is '3d'.

        We use transposed convolutions to upscale the input. We will use 
        nn.ConvTranspose2d when mode is '2d' and nn.ConvTranspose3d when mode is 
        '3d'. This is used to decode the input into a 2D or 3D feature tensor.

        Note: Do not call '.to(device)' on any modules defined here. 
              All modules are automatically mounted to device when initiated 
              and mounted on device. 
        """

        if mode=='2d':
            conv = nn.Conv2d
            transpose_conv = nn.ConvTranspose2d
            input_hidden_channels = hidden_ch + 2
        elif mode == '3d':
            conv = nn.Conv3d
            transpose_conv = nn.ConvTranspose3d
            input_hidden_channels = hidden_ch + 3

        ###
        # Specify in_conv as a nn.Sequential with (conv, LeakyReLU) where conv 
        # is the convolution operation defined above. This uses the input 
        # channels as in_ch and outputs hidden_ch channels with stride of 1 and 
        # padding specified as 'same' with 'zeros'. Make sure to perform 
        # LeakyReLU inplace and use negative_slope of 0.2.
        self.in_conv = nn.Sequential(conv(in_channels = in_ch, out_channels = hidden_ch, kernel_size = 3, stride = 1, padding = 'same', padding_mode = 'zeros'), 
                                      nn.LeakyReLU(inplace=True, negative_slope = neg_slope))

        # Hidden transposed convolution layers
        # One way to define nn.Sequential module is nn.Sequential(*arr) where
        # arr is a list containing nn.Module units. Here, we will append 
        # (CoordCat, transpose_conv, LeakyReLU), 'num_up' times. CoordCat needs
        # to mode as the argument. Each of the transpose_conv use 
        # input_hidden_channels as the number of input channels and input and 
        # output hidden_ch channels using a kernel size of 3, stride
        # of 2, padding as 1, and output_padding as 1.
        hidden_conv = []
        for i in range(num_up):
            hidden_conv.append(CoordCat(mode))
            hidden_conv.append(transpose_conv(in_channels = input_hidden_channels, out_channels = hidden_ch, kernel_size = 3, stride = 2, padding = 1, output_padding = 1))
            hidden_conv.append(nn.LeakyReLU(inplace=True, negative_slope = neg_slope))
    
        self.hidden_conv = nn.Sequential(*hidden_conv)

        # Specify out_conv using conv with hidden_ch input channels and outputting
        # out_ch with kernel size as 3, padding as 'same', and padding_mode as 
        # 'zeros'.

        self.out_conv = conv(in_channels = hidden_ch, out_channels = out_ch, kernel_size = 3, padding = 'same', padding_mode = 'zeros')
        
        self.apply(init_weights)
        
    def forward(self, input):

        # Call in_conv, hidden_conv, and out_conv modules on the input.
        input = self.in_conv(input)
        input = self.hidden_conv(input)
        input = self.out_conv(input)
        return(input)
    
class AutoDecoderWrapper(nn.Module):
    def __init__(
        self, 
        num_latents:int, # Number of latent codes for the auto-decoder.
        submodule:nn.Module, # nn.Module instance we want to turn into an auto-decoder.
        param_name:str, # Name of the parameter of the nn.Module that we want to replace with the auto-decoded latents.
        in_wgt = 1e-4,
    ):
        super().__init__()

        trgt_param = dict(submodule.named_parameters())[param_name]
        self.trgt_param_shape = trgt_param.shape[1:]
        #print(self.trgt_param_shape)

        # nn.Embedding is like a "list" of parameters: In the forward pass,
        # you can pass a tensor of indices, and this will return the corresponding
        # latent.
        self.latents = nn.Embedding(num_embeddings=num_latents,
                                    embedding_dim=np.prod(self.trgt_param_shape))
        self.latents.weight.data.normal_(0, in_wgt)
           
        self.param_name = param_name
        self.submodule = submodule


    def forward(self, 
                inputs):
        '''
        inputs: dictionary. Expected to have key "idx" for the latent idcs, as well as other inputs for the sub-module.  
        '''

        #######

        # Retrieve the dictionary entry "idx" from the input dictionary
        latent_idcs =  inputs['idx']
        #print(latent_idcs.shape)
        batch_size = latent_idcs.shape[0]
        #print(batch_size)
        
        # Use the "idx" variable to index into the self.latents nn.Embedding module
        # and reshape it as (batch_size, target_param_shape[0], ..., target_param_shape[-1])
        param_latent = self.latents(latent_idcs)
        #print(torch.mean(params))
        p_shape = [batch_size]
        for t in self.trgt_param_shape:
            p_shape.append(t)
        param_latent = torch.reshape(param_latent, tuple(p_shape))
        #print(torch.mean(params))

        # Build a parameter dictionary where self.param_name key holds param as value.
        param_dict = {}
        param_dict[self.param_name] = param_latent

        # call functional_call using the submodule which is the network, the 
        # param_dict, and the inputs as arguments,
        output = functional_call(self.submodule, param_dict, inputs)
        
        #print(torch.mean(param_latent))
        #print(torch.mean(param_dict[self.param_name]))

        # return the output of the functional call and params
        return output, param_latent
    
