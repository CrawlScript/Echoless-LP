# coding=utf-8

import torch
import torch.nn as nn
from itertools import chain
import torch.nn.functional as F

from echoless_lp.layers.torch_train_model import CommonTorchTrainModel
from echoless_lp.layers.transformer import MyMLP, Transformer
import numpy as np
# import F
import torch.nn.functional as F



class PReLU(nn.Module):

    __constants__ = ['num_parameters']
    num_parameters: int

    def __init__(self, num_parameters: int = 1, init: float = 0.25,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        super().__init__()

        # use alpha instead of weight
        self.alpha = nn.parameter.Parameter(torch.empty(num_parameters, **factory_kwargs).fill_(init))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.prelu(input, self.alpha)

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)


class Lambda(nn.Module):
    def __init__(self, func) -> None:
        super().__init__()

        self.func = func

    def forward(self, x):
        return self.func(x)
        
def create_act(name=None):

    if name is None:
        return nn.Identity()
    elif name == "relu":
        return nn.ReLU()
    elif name == "prelu":
        return PReLU()
    elif name == "softmax":
        return nn.Softmax(dim=-1)
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "identity":
        return Lambda(lambda x: x)
    else:
        raise Exception()


class Linear(nn.Linear):
    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)


class Conv1d(nn.Conv1d):
    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)


class MLPConv1d(nn.Module):
    """
    another implementation of Conv1d
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = Linear(in_channels, out_channels)

    def forward(self, x):
        h = torch.permute(x, (0, 2, 1))
        h = self.linear(h)
        h = torch.permute(h, (0, 2, 1))
        h = h.contiguous()
        return h


class MLP(nn.Module):

    def __init__(self,
                 channels_list,
                 input_shape,
                 drop_rate=0.0,
                 activation=None,
                 bn=False,
                 ln=False,
                 output_drop_rate=0.0,
                 output_activation=None,
                 output_bn=False,
                 output_ln=False,
                 kernel_regularizer=None,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.kernel_regularizer = kernel_regularizer

        in_channels = input_shape[-1]
        channels_list = [in_channels] + channels_list

        layers = [] 
        for i in range(len(channels_list) - 1):
            layers.append(Linear(channels_list[i], channels_list[i + 1]))
            if i < len(channels_list) - 2:

                if bn:
                    layers.append(nn.BatchNorm1d(channels_list[i + 1]))
                if ln:
                    layers.append(nn.LayerNorm(channels_list[i + 1]))

                layers.append(create_act(activation))
                layers.append(nn.Dropout(drop_rate))
            else:

                if output_bn:
                    layers.append(nn.BatchNorm1d(channels_list[i + 1]))
                if output_ln:
                    layers.append(nn.LayerNorm(channels_list[i + 1]))

                layers.append(create_act(output_activation))
                layers.append(nn.Dropout(output_drop_rate))




        self.layers = nn.Sequential(*layers)



    def forward(self, x):
        return self.layers(x)



















class GroupEncoders(nn.Module):

    def __init__(self,
                 filters_list,
                 drop_rate,
                 input_shape,
                 num_label_groups=0,
                 kernel_regularizer=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hop_encoders = None
        self.filters_list = filters_list
        self.drop_rate = drop_rate
        self.kernel_regularizer = kernel_regularizer
        self.real_filters_list = None

        self.num_label_groups = num_label_groups

        num_groups = len(input_shape)

        self.num_groups = num_groups

        self.group_sizes = [group_shape[1] for group_shape in input_shape]
        self.real_filters_list = [self._get_real_filters(i) for i in range(num_groups)]

        self.group_encoders = nn.ModuleList([
            nn.Sequential(
                # Conv1d(group_size, real_filters, 1, stride=1),
                # # if too slow, comment MyConv1d (above) and uncomment MyMLPConv1d (below)
                MLPConv1d(group_size, real_filters),
                Lambda(lambda x: x.view(x.size(0), -1))
            )
            for _, (group_size, real_filters) in enumerate(zip(self.group_sizes, self.real_filters_list))
        ])


    def _get_real_filters(self, i):

        # if i >= self.num_groups - self.num_label_groups:
        #     return 1
        # elif self.group_sizes[i] == 1:
        #     return 1
        # elif isinstance(self.filters_list, list):
        #     return self.filters_list[i]
        # else:
        #     return self.filters_list

        # if i >= self.num_groups - self.num_label_groups:
        #     return 1
        if self.group_sizes[i] == 1:
            return 1
        elif isinstance(self.filters_list, list):
            return self.filters_list[i]
        else:
            return self.filters_list
 

    def forward(self, x_group_list):
        group_h_list = []

        for i, (x_group, group_encoder) in enumerate(zip(x_group_list, self.group_encoders)):

            h = x_group
            group_h = group_encoder(h)
            group_h_list.append(group_h)

        return group_h_list



class MultiGroupFusionGT(nn.Module):

    def __init__(self,
                 group_channels_list,
                 global_channels_list,
                 merge_mode,
                 input_shape,
                 num_label_groups=0,
                 drop_rate=0.0,
                 activation="prelu",
                 output_activation=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_label_groups = num_label_groups

        self.group_fc_list = None
        self.global_fc = None

        self.group_channels_list = group_channels_list
        self.global_channels_list = global_channels_list
        self.merge_mode = merge_mode
        self.drop_rate = drop_rate

        self.use_shared_group_fc = False
        self.group_encoder_mode = "common" 

        num_groups = len(input_shape)
        self.num_groups = num_groups

        num_feat_groups = num_groups - num_label_groups

        self.feat_group_fc_list = nn.ModuleList([
            MLP(
                group_channels_list, 
                input_shape=group_input_shape,
                drop_rate=drop_rate,
                activation=activation,
                output_drop_rate=drop_rate,
                output_activation=activation,
                # bn=i >= num_groups - num_label_groups,
                # output_bn=i >= num_groups - num_label_groups,
            )
            for i, group_input_shape in enumerate(input_shape[:num_feat_groups])
        ])


        # transformer_channels = 128
        # transformer_channels = 256
        # transformer_channels = 512

        # # imdb best
        # transformer_channels = 64


        transformer_channels = 32

        self.label_group_fc_list = nn.ModuleList([
            nn.Sequential(
                # nn.Dropout(get_input_drop_rate(i)),
                MyMLP(
                    group_input_shape[-1],
                    [transformer_channels],
                    activation=None,
                    drop_rate=0.0,
                    bn=False,
                    output_activation="prelu",
                    # output_drop_rate=0.1,
                    # output_drop_rate=label_input_drop_rate,
                    output_drop_rate=0.0,
                    # output_drop_rate=get_input_drop_rate(i),
                    output_bn=False,
                    ln=True,
                    output_ln=True
                )
            )
            for i, group_input_shape in enumerate(input_shape[num_feat_groups:])
        ])




        num_heads = 2
        att_units = 2
        

        
        tf_drop_rate = 0.1

        
        att_drop_rate = 0.1
        # att_drop_rate = 0.5
        

        self.label_encoder_transformer = Transformer(
            transformer_channels,
            # att_units,
            att_units, # * 32,
            transformer_channels,
            ff_units_list=[],
            # ff_units_list=[],
            num_heads=num_heads,
            drop_rate=tf_drop_rate,
            att_drop_rate=att_drop_rate,

            att_ln=True,
            att_residual=True
        )


        label_hop_encodings = torch.randn(1, self.num_label_groups + 1, transformer_channels) / np.sqrt(transformer_channels)
        self.label_hop_encodings = nn.Parameter(label_hop_encodings)

        # q = torch.randn(1, 1, transformer_channels) / np.sqrt(transformer_channels)
        q = torch.zeros(1, 1, transformer_channels)
        self.q = nn.Parameter(q, requires_grad=False)









        if merge_mode in ["mean", "free"]:
            global_input_shape = [-1, group_channels_list[-1]]
        elif merge_mode == "concat":
            # global_input_shape = [-1, group_channels_list[-1] * num_groups]
            global_input_shape = [-1, group_channels_list[-1] * num_feat_groups]
        else:
            raise Exception("wrong merge mode: ", merge_mode)
        
        # if self.num_label_groups > 0:
        #     global_input_shape = [-1, global_input_shape[-1] + transformer_channels]
                                  
        # self.global_fc = MLP(self.global_channels_list, 
        #                      input_shape=global_input_shape,
        #                      drop_rate=self.drop_rate, 
        #                      activation=activation,
        #                      output_drop_rate=0.0,
        #                      output_activation=output_activation)

        
        self.global_feat_fc = MLP(self.global_channels_list[:-1], 
                                  input_shape=global_input_shape,
                                  drop_rate=self.drop_rate, 
                                  activation=activation,
                                  output_drop_rate=0.0,
                                  output_activation=activation)
        

        self.cls_fc = nn.Sequential(
            nn.Dropout(self.drop_rate),
            Linear(self.global_channels_list[-2] + transformer_channels, 
                   self.global_channels_list[-1])
        )

        self.group_dropout = nn.Dropout(0.1)


    def forward(self, inputs):

        x_list = inputs
        # group_h_list = [group_fc(x) for x, group_fc in zip(x_list, self.group_fc_list)]

        # if self.num_label_groups == 0:
        #     feat_group_h_list = group_h_list
        #     label_group_h_list = None
        # else:
        #     feat_group_h_list = group_h_list[:-self.num_label_groups]
        #     label_group_h_list = group_h_list[-self.num_label_groups:]

        if self.num_label_groups == 0:
            feat_x_list = x_list
            label_x_list = None
        else:
            feat_x_list = x_list[:-self.num_label_groups]
            label_x_list = x_list[-self.num_label_groups:]

        feat_group_h_list = [feat_group_fc(feat_x) for feat_x, feat_group_fc in zip(feat_x_list, self.feat_group_fc_list)]

        if self.merge_mode == "mean":
            feat_global_h = torch.stack(feat_group_h_list, dim=0).mean(dim=0)
        elif self.merge_mode == "concat":
            feat_global_h = torch.concat(feat_group_h_list, dim=-1)
        else:
            raise Exception("wrong merge mode: ", self.merge_mode)
        



    
        label_h_list = [label_group_fc(label_x) for label_x, label_group_fc in zip(label_x_list, self.label_group_fc_list)]

        label_h_list = torch.stack(label_h_list, dim=1)
        label_h_list = torch.concat([
            # 
            label_h_list,
            # self.label_q_fc(feat_h).unsqueeze(1),
            self.q.expand(label_h_list.size(0), -1, -1)
        ], dim=1)
        label_h_list = label_h_list + self.label_hop_encodings

        label_h_list = self.group_dropout(label_h_list)

        label_h_list = self.label_encoder_transformer(
            # label_h_list,
            label_h_list[:, :-1],
            label_h_list,
            # use_causal_mask=True
        )


        label_h = label_h_list[:, -1]



        feat_h = self.global_feat_fc(feat_global_h)

        global_h = torch.concat([feat_h, label_h], dim=-1)

        h = self.cls_fc(global_h)

        return h



class RpHGNNGTEncoder(CommonTorchTrainModel):

    def __init__(self,
                 filters_list,
                 group_channels_list,
                 global_channels_list,
                 merge_mode,
                 input_shape,
                 *args,

                 num_label_groups=0,
                 input_drop_rate=0.0,
                 label_input_drop_rate=0.0,

                 drop_rate=0.0,
                 activation="prelu",
                 output_activation=None,
                 **kwargs):
        
        super().__init__(*args, **kwargs)

        self.input_dropout = nn.Dropout(input_drop_rate)
        self.input_drop_rate = input_drop_rate

        group_encoders_input_shape = input_shape
        self.group_encoders = GroupEncoders(filters_list, 
                                            drop_rate, 
                                            group_encoders_input_shape,
                                            num_label_groups=num_label_groups)

        multi_group_fusion_input_shape = [[-1, group_input_shape[-1] * filters] 
                                          for group_input_shape, filters in zip(group_encoders_input_shape, self.group_encoders.real_filters_list)]
        self.multi_group_fusion = MultiGroupFusionGT(
            group_channels_list, global_channels_list, 
            merge_mode, 
            input_shape=multi_group_fusion_input_shape,
            num_label_groups=num_label_groups,
            drop_rate=drop_rate,
            activation=activation, 
            output_activation=output_activation)
        
        num_groups = len(input_shape)
        def get_input_drop_rate(i):
            if i < num_groups - num_label_groups:
                return input_drop_rate
            else:
                return label_input_drop_rate
            
        self.input_dropouts = nn.ModuleList([
            nn.Dropout(get_input_drop_rate(i)) for i in range(num_groups)
        ])


    def forward(self, inputs):

        x_group_list = inputs
        # dropped_x_group_list = [F.dropout(x_group, self.input_drop_rate, training=self.training, inplace=False) for x_group in x_group_list]

        assert len(x_group_list) == len(self.input_dropouts)
        dropped_x_group_list = [dropout(x_group) for x_group, dropout in zip(x_group_list, self.input_dropouts)]

        h_list = self.group_encoders(dropped_x_group_list)
        h = self.multi_group_fusion(h_list)

        return h
