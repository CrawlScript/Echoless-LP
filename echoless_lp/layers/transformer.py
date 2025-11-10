
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyLinear(nn.Linear):
    # pass
    def reset_parameters(self) -> None:
        # nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_normal_(self.weight)

        # print("init weight:")
        # print(self.weight)
        # asdfasdfadf

        # nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)




class MyNormalLinear(nn.Linear):
    # pass
    def reset_parameters(self) -> None:
        # nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MyPReLU(nn.Module):

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
    if name == "softmax":
        return nn.Softmax(dim=-1)
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "identity":
        return Lambda(lambda x: x)
    else:
        raise Exception()


# class MyLinear(nn.Linear):
#     def reset_parameters(self) -> None:
#         nn.init.xavier_normal_(self.weight)
#         nn.init.zeros_(self.bias)


def get_activation(activation):
    if activation == "prelu":
        return MyPReLU()
    elif activation == "relu":
        return nn.ReLU()
    
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    
    elif activation == "gelu":
        return nn.GELU()

    elif activation is None:
        return torch.nn.Identity()
    else:
        raise NotImplementedError(f"Activation {activation} not implemented")
    



class MyMLP(nn.Module):
    def __init__(self, in_channels, units_list, activation, drop_rate, bn, 
                 output_activation, output_drop_rate, output_bn,
                 ln=False,
                 output_ln=False,
                 ):
        super().__init__()

        layers = []
        units_list = [in_channels] + units_list  # Add in_channels to the list of units

        for i in range(len(units_list) - 1):
            layers.append(MyLinear(units_list[i], units_list[i+1]))  # Create a linear layer
            # layers.append(MyNormalLinear(units_list[i], units_list[i+1]))  # Create a linear layer


            if i < len(units_list) - 2:
                if bn:
                    layers.append(nn.BatchNorm1d(units_list[i+1]))  # Add a batch normalization layer

                if ln:
                    layers.append(nn.LayerNorm(units_list[i+1]))

                layers.append(get_activation(activation))  # Add the PReLU activation function
                layers.append(nn.Dropout(drop_rate))
            else:
                if output_bn:
                    layers.append(nn.BatchNorm1d(units_list[i+1]))
                
                if output_ln:
                    layers.append(nn.LayerNorm(units_list[i+1]))

                layers.append(get_activation(output_activation))  # Add the PReLU activation function
                layers.append(nn.Dropout(output_drop_rate))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)




class Transformer(nn.Module):
    def __init__(self, in_units, att_units, out_units, 
                 ff_units_list,
                 num_heads=1,
                 drop_rate=0.0,
                 att_drop_rate=0.0,
                 att_ln=True,
                 att_activation=None,
                 
                 use_self_output=False,

                 ff_drop_rate=0.0,
                 ff_activation="gelu",
                 ff_output_activation=None,
                 ff_output_drop_rate=0.0,
                 ff_output_ln=False,

                 att_residual=True,
                 ff_residual=True,
                 use_v=True
                 ) -> None:
        super().__init__()


        self.q_linear = MyLinear(in_units, att_units)
        self.k_linear = MyLinear(in_units, att_units)

        self.use_v = use_v
        if self.use_v:
            self.v_linear = MyLinear(in_units, out_units)


        self.use_self_output = use_self_output
        if use_self_output:
            self.self_output_linear = MyLinear(in_units, in_units)


        self.ff_units_list = ff_units_list

        # ff_units_list = [out_units] + ff_units_list

        self.att_dropout = nn.Dropout(att_drop_rate)

        if att_ln:
            self.ln = nn.LayerNorm(out_units)
        else:
            self.ln = None

        self.att_activation = get_activation(att_activation)
        self.dropout = nn.Dropout(drop_rate)
        

        if len(ff_units_list) == 0:
            self.ff = None
        else:
            self.ff = MyMLP(out_units, ff_units_list, 
                            activation=ff_activation,
                            drop_rate=ff_drop_rate,
                            bn=False,
                            output_activation=None,
                            output_drop_rate=0.0,
                            output_bn=False,
                            
                            # ln=True,
                            ln=False,
                            output_ln=False
                            )

        
        self.ff_output_activation = get_activation(ff_output_activation)
        self.ff_output_dropout = nn.Dropout(ff_output_drop_rate)



        # self.output_bn = nn.BatchNorm1d(ff_units_list[-1]) if output_bn else None
        if len(ff_units_list) > 0:
            self.ff_output_ln = nn.LayerNorm(ff_units_list[-1]) if ff_output_ln else None

        self.att_residual = att_residual
        self.ff_residual = ff_residual

        self.num_heads = num_heads
        

    def forward(self, q, k, q_pe=None, k_pe=None, 
                use_causal_mask=False,
                use_reversed_mask=False,
                return_all=False):

        if q_pe is not None:          
            Q = self.q_linear(q + q_pe)
        else:
            Q = self.q_linear(q)
        
        if k_pe is not None:
            K = self.k_linear(k + k_pe)
        else:
            K = self.k_linear(k)


        # if q_pe is not None:
        #     print("q", q.shape)
        #     print("k", k.shape)
        #     print("q_pe", q_pe.shape)
        #     print("k_pe", k_pe.shape)

        #     print("Q", Q.shape)
        #     print("K", K.shape)
        #     adsfadsf


        if self.use_v:
            V = self.v_linear(k)
        else:
            V = k

        # Q_ = torch.concat(Q.split(Q.size(-1) // self.num_heads, dim=-1), dim=0)
        # K_ = torch.concat(K.split(K.size(-1) // self.num_heads, dim=-1), dim=0)
        # V_ = torch.concat(V.split(V.size(-1) // self.num_heads, dim=-1), dim=0)

        Q_ = Q.view(Q.size(0), Q.size(1), self.num_heads, -1).permute(0, 2, 1, 3)
        K_ = K.view(K.size(0), K.size(1), self.num_heads, -1).permute(0, 2, 1, 3)
        V_ = V.view(V.size(0), V.size(1), self.num_heads, -1).permute(0, 2, 1, 3)

        sim_logits = Q_ @ K_.transpose(-2, -1)
        sim_logits = sim_logits / (Q_.size(-1) ** 0.5)

        if use_reversed_mask:
            asdfasdf
            # mask = torch.triu(torch.ones(sim_logits.size(-2), sim_logits.size(-1)), diagonal=0).to(sim_logits.device).bool()
            # mask = torch.unsqueeze(mask, dim=0).repeat(sim_logits.size(0), 1, 1)
            # sim_logits = torch.where(mask, sim_logits, -1e9)


        if use_causal_mask:
            mask = torch.tril(torch.ones(sim_logits.size(-2), sim_logits.size(-1)), diagonal=0).to(sim_logits.device).bool()
            mask = mask.unsqueeze(0).unsqueeze(0).repeat(sim_logits.size(0), self.num_heads, 1, 1)

            sim_logits = torch.where(mask, sim_logits, -1e9)

        sim = F.softmax(sim_logits, dim=-1)



        dropped_sim = self.att_dropout(sim)

        att_h = dropped_sim @ V_

        # att_h = torch.concat(att_h.split(q.size(0), dim=0), dim=-1)

        att_h = att_h.permute(0, 2, 1, 3).contiguous().view(q.size(0), q.size(1), -1)


        if self.use_self_output:
            att_h = self.self_output_linear(att_h)

        att_h = self.dropout(att_h)

        if self.att_residual:
            att_h = att_h + q
        
        if self.ln is not None:
            att_h = self.ln(att_h)
    
        att_h = self.att_activation(att_h)
        

        
        if self.ff is None:
            ff_h = att_h
        else:
            ff_h = self.ff(att_h)

            ff_h = self.ff_output_dropout(ff_h)

            if self.ff_residual:
                ff_h = ff_h + att_h

            if self.ff_output_ln is not None:
                ff_h = self.ff_output_ln(ff_h)

            ff_h = self.ff_output_activation(ff_h)
            
        if return_all:
            sim_logits_list = torch.split(sim_logits, q.size(0), dim=0)
            return ff_h, sim_logits_list
        return ff_h
        
      