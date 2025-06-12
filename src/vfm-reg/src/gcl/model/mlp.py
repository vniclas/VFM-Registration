import torch.nn as nn

# import MinkowskiEngine as ME
# import MinkowskiEngine.MinkowskiFunctional as MEF
# from model.common import get_norm


class GenerativeMLP(nn.Module):
    CHANNELS = [None, 512, 128, None]

    def __init__(self, in_channel=125, out_points=6, bn_momentum=0.1):
        super().__init__()
        CHANNELS = self.CHANNELS
        self.mlp = nn.Sequential(nn.Linear(in_channel, CHANNELS[1]), nn.ReLU(),
                                 nn.BatchNorm1d(CHANNELS[1], momentum=bn_momentum),
                                 nn.Linear(CHANNELS[1], CHANNELS[2]), nn.ReLU(),
                                 nn.BatchNorm1d(CHANNELS[2], momentum=bn_momentum),
                                 nn.Linear(CHANNELS[2], out_points * 3), nn.ReLU())

    def forward(self, x):
        y = self.mlp(x)
        # print(y)
        return y


# class GenerativeMLP_99(GenerativeMLP):
#     CHANNELS = [None, 512, 512, None]


class GenerativeMLP_98(GenerativeMLP):
    CHANNELS = [None, 512, 256, None]


class GenerativeMLP_54(GenerativeMLP):
    CHANNELS = [None, 32, 16, None]


class GenerativeMLP_4(nn.Module):
    CHANNELS = [None, 16, None]

    def __init__(self, in_channel=125, out_points=6, bn_momentum=0.1):
        super().__init__()
        CHANNELS = self.CHANNELS
        self.mlp = nn.Sequential(nn.Linear(in_channel, CHANNELS[1]), nn.ReLU(),
                                 nn.BatchNorm1d(CHANNELS[1], momentum=bn_momentum),
                                 nn.Linear(CHANNELS[1], out_points * 3), nn.ReLU())

    def forward(self, x):
        y = self.mlp(x)
        # print(y)
        return y


class GenerativeMLP_11_10_9(nn.Module):
    CHANNELS = [None, 2048, 1024, 512, None]

    def __init__(self, in_channel=125, out_points=6, bn_momentum=0.1):
        super().__init__()
        CHANNELS = self.CHANNELS
        self.mlp = nn.Sequential(nn.Linear(in_channel, CHANNELS[1]), nn.ReLU(),
                                 nn.BatchNorm1d(CHANNELS[1], momentum=bn_momentum),
                                 nn.Linear(CHANNELS[1], CHANNELS[2]), nn.ReLU(),
                                 nn.BatchNorm1d(CHANNELS[2], momentum=bn_momentum),
                                 nn.Linear(CHANNELS[2], CHANNELS[3]), nn.ReLU(),
                                 nn.BatchNorm1d(CHANNELS[3], momentum=bn_momentum),
                                 nn.Linear(CHANNELS[3], out_points * 3), nn.ReLU())

    def forward(self, x):
        y = self.mlp(x)
        # print(y)
        return y


# import torch.nn as nn

# class GenerativeMLP(nn.Module):
#     CHANNELS = [None, 512, 128, None]

#     def __init__(self,
#                  in_channel=125,
#                  out_points=6,
#                  radius = 1,
#                  bn_momentum=0.1):
#         super().__init__()
#         # print(in_channel)
#         self.CHANNELS[0] = in_channel
#         self.CHANNELS[-1] = out_points*3
#         self.list_modules = []
#         self.radius = radius
#         for layer_idx in range(len(self.CHANNELS)-1):
#             # print(layer_idx)
#             if layer_idx < len(self.CHANNELS)-1:    # middle layer
#                 self.list_modules.append(
#                     nn.Sequential(
#                         nn.Linear(self.CHANNELS[layer_idx], self.CHANNELS[layer_idx+1]),
#                         nn.ReLU(),
#                         nn.BatchNorm1d(self.CHANNELS[layer_idx+1], momentum=bn_momentum),
#                     )
#                 )
#             else:   # last layer
#                 self.list_modules.append(
#                     nn.Sequential(
#                         nn.Linear(self.CHANNELS[layer_idx], self.CHANNELS[layer_idx+1]),
#                         nn.ReLU(),
#                     )
#                 )
#         self.list_modules = nn.ModuleList(self.list_modules)

#     def forward(self, x):
#         # print(x.size(), self.CHANNELS)
#         for module in self.list_modules:
#             x = module(x)
#         if self.radius is None:
#             return x
#         else:
#             return x, self.radius

# class GenerativeMLP_99(GenerativeMLP):
#     CHANNELS = [None, 512, 512, None]

# class GenerativeMLP_98(GenerativeMLP):
#     CHANNELS = [None, 512, 256, None]

# class GenerativeMLP_54(GenerativeMLP):
#     CHANNELS = [None, 32, 16, None]

# class GenerativeMLP_4(GenerativeMLP):
#     CHANNELS = [None, 16, None]

# class GenerativeMLP_11_10_9(GenerativeMLP):
#     CHANNELS = [None, 2048, 1024, 512, None]

# def get_GenerativeMLP(config, radius=None, in_channels=None):
#     models = [GenerativeMLP_4, GenerativeMLP_98, GenerativeMLP_99, GenerativeMLP_54, GenerativeMLP_11_10_9]
#     mdict = {model.__name__: model for model in models}
#     if in_channels is None:
#         in_channels = config.final_feats_dim
#     return mdict[config.generative_model](in_channel=in_channels,
#                                           out_points=config.point_generation_ratio,
#                                           radius=radius,
#                                           bn_momentum=config.batch_norm_momentum)
