import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LoRALayer


def transpose(w, fan_in_fan_out):
    return w.T if fan_in_fan_out else w


class MMOELoraLayer(LoRALayer):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        expert_num: int,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
        tunable_scaler: bool = False,
    ):

        super().__init__(
            r, lora_alpha, lora_dropout, merge_weights, tunable_scaler
        )
        self.expert_num = expert_num

    def update_layer(self, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout = lora_dropout_layer

        # Actual trainable parameters
        if r > 0:
            self.lora_A = MMOELinearA(self.in_features, r, self.expert_num)

            self.lora_B = MMOELinearB(r, self.out_features, self.expert_num)

            self.scaling = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters()

        self.to(self.weight.device)

    def reset_lora_parameters(self):
        if self.r > 0:
            # initialize A the same way as the default for nn.Linear and B to zero
            for i in range(self.expert_num):
                nn.init.normal_(
                    self.lora_A.loraA[i].mlp.weight,
                    mean=0.0,
                    std=0.01,
                )
                nn.init.zeros_(self.lora_B.loraB[i].mlp.weight)


class MMOELoraLinear(nn.Linear, MMOELoraLayer):
    # Lora implemented in a dense layer
    # nn.Linear is the pretrained weights in LLM, MMOELoraLayer is the designed trainable Lora
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.expert_num = kwargs.pop("expert_num", True)
        self.task_num = kwargs.pop("task_num", True)
        self.te_dim = kwargs.pop("task_embedding_dim", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        MMOELoraLayer.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            expert_num=self.expert_num,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=False,
            tunable_scaler=False,
        )

        # init the Gate network
        self.lora_task_embedding = nn.Embedding(self.task_num + 1, self.te_dim)
        self.lora_gate = Gate(self.te_dim, self.expert_num)

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(r, lora_alpha, lora_dropout, init_lora_weights)

    def merge(self, task_id):
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r > 0:
            expert_weight = self.lora_gate(self.lora_task_embedding(task_id))
            for i in range(self.expert_num):
                lora_A_weights = self.lora_A.loraA[i].mlp.weight
                lora_B_weights = self.lora_B.loraB[i].mlp.weight
                self.weight.data += (
                    transpose(
                        lora_B_weights @ lora_A_weights,
                        self.fan_in_fan_out,
                    )
                    * self.scaling
                    * expert_weight[..., i]
                )
            self.merged = True

    def unmerge(self, task_id):
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r > 0:
            expert_weight = self.lora_gate(self.lora_task_embedding(task_id))
            for i in range(self.expert_num):
                lora_A_weights = self.lora_A.loraA[i].mlp.weight
                lora_B_weights = self.lora_B.loraB[i].mlp.weight
                self.weight.data -= (
                    transpose(
                        lora_B_weights @ lora_A_weights,
                        self.fan_in_fan_out,
                    )
                    * self.scaling
                    * expert_weight[..., i]
                )
            self.merged = False

    def forward(self, x: torch.Tensor, lambda_index):
        task_id = lambda_index
        previous_dtype = x.dtype

        if self.r > 0 and not self.merged:  # general lora process
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )

            x = x.to(self.lora_A.loraA[0].weight.dtype)

            expert_weight = self.lora_gate(self.lora_task_embedding(task_id))
            for i in range(self.expert_num):
                result += (  # lora process
                    self.lora_B.loraB[i](
                        self.lora_A.loraA[i](self.lora_dropout(x)),
                    )
                    * self.scaling
                    * expert_weight[..., i].unsqueeze(-1).unsqueeze(-1)
                )
        else:
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )

        result = result.to(previous_dtype)

        return result


class MMOELinearA(nn.Module):
    """MMOE based LoRA block"""

    def __init__(self, in_features, out_features, expert_num) -> None:

        super().__init__()

        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features
        self.loraA = nn.ModuleList([])

        assert (
            self.out_features % self.expert_num == 0
        )  # lora rank should be divided by expert number
        self.r = self.out_features // self.expert_num

        for _ in range(self.expert_num):
            self.loraA.append(Expert(self.in_features, self.r))

    def forward(self, x):
        """input x is a vector, return output is a list"""
        outputs = []
        for i in range(self.expert_num):
            outputs.append(self.loraA[i](x))

        return outputs


class MMOELinearB(nn.Module):
    """MMOE based LoRA block"""

    def __init__(self, in_features, out_features, expert_num) -> None:

        super().__init__()

        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features
        self.loraB = nn.ModuleList([])

        assert self.in_features % self.expert_num == 0
        self.r = self.in_features // self.expert_num

        for _ in range(self.expert_num):
            self.loraB.append(Expert(self.r, self.out_features))

    def forward(self, x):
        """input x is a list, return output is also a list"""
        outputs = []
        for i in range(self.expert_num):
            outputs.append(self.loraB[i](x[i]))

        return outputs


class Expert(nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()

        self.in_features, self.out_features = in_features, out_features
        self.mlp = nn.Linear(self.in_features, self.out_features, bias=False)
        self.weight = self.mlp.weight

    def forward(self, x):
        # LoRA A or B block
        y = self.mlp(x)

        return y


class Gate(nn.Module):

    def __init__(self, input_size, expert_num):

        super().__init__()
        self.GateL = nn.Linear(input_size, expert_num, bias=False)
        self.act = nn.Softmax(dim=1)

    def forward(self, x):

        y = self.GateL(x)
        y = self.act(y)

        return y
