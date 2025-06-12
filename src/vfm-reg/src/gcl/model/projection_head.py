import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
import torch
from model.common import get_norm
from model.residual_block import get_block


class ProjectionHeadConv(ME.MinkowskiNetwork):
    #   CHANNELS = [None, 32, 64, 128, 256]
    #   TR_CHANNELS = [None, 32, 64, 64, 128]

    def __init__(self,
                 in_channels=128,
                 out_channels=16,
                 bn_momentum=None,
                 conv1_kernel_size=None,
                 D=3):
        ME.MinkowskiNetwork.__init__(self, D)

        self.conv1 = ME.MinkowskiConvolution(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=conv1_kernel_size,
                                             stride=1,
                                             dilation=1,
                                             bias=False,
                                             dimension=D)

    def forward(self, x):
        return self.conv1(x)


class ProjectionHeadMLP(ME.MinkowskiNetwork):
    CHANNEL = 128
    NORM_TYPE = None
    BLOCK_NORM_TYPE = 'BN'

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling initialize_coords
    def __init__(self,
                 in_channels=128,
                 out_channels=16,
                 bn_momentum=0.1,
                 conv1_kernel_size=None,
                 D=3):
        ME.MinkowskiNetwork.__init__(self, D)
        CHANNEL = self.CHANNEL
        NORM_TYPE = self.NORM_TYPE
        BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE

        self.conv1 = ME.MinkowskiConvolution(in_channels=in_channels,
                                             out_channels=CHANNEL,
                                             kernel_size=conv1_kernel_size,
                                             stride=1,
                                             dilation=1,
                                             bias=False,
                                             dimension=D)

        self.norm1 = get_block(BLOCK_NORM_TYPE, CHANNEL, CHANNEL, bn_momentum=bn_momentum, D=D)

        self.conv2 = ME.MinkowskiConvolution(in_channels=self.CHANNEL,
                                             out_channels=out_channels,
                                             kernel_size=conv1_kernel_size,
                                             stride=1,
                                             dilation=1,
                                             bias=False,
                                             dimension=D)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = MEF.relu(x)
        x = self.conv2(x)
        return x
