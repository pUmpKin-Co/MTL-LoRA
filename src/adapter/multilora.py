import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LoRALayer, should_gather


class MultiLoRALinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_num: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        tunable_scaler: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
            tunable_scaler=tunable_scaler,
        )

        self.fan_in_fan_out = fan_in_fan_out
        self.lora_num = lora_num
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((self.lora_num, r, in_features))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((self.lora_num, out_features, r))
            )
            self.lora_scaling = nn.Parameter(
                self.weight.new_zeros(
                    (
                        self.lora_num,
                        out_features,
                    )
                )
            )
            self.scaling_factor = lora_alpha / r
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            for i in range(self.lora_num):
                # initialize A the same way as the default for nn.Linear and B to kaiming uniform and scaling to zero.
                nn.init.kaiming_uniform_(self.lora_A[i, ...], a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.lora_B[i, ...], a=math.sqrt(5))
                nn.init.zeros_(self.lora_scaling[i, ...])

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)

        should = self.merged if mode else not self.merged

        if self.merge_weights and should:
            if self.r > 0:
                gatherA = should_gather(self.lora_A)
                gatherB = should_gather(self.lora_B)
                gatherW = should_gather(self.weight)
                gatherL = should_gather(
                    self.lora_scaling
                )  # replace the constant.
                gatherS = should_gather(self.lora_scaler)
                with gatherA, gatherB, gatherW, gatherS, gatherL:
                    for i in range(self.lora_num):
                        delta_w = (
                            T(self.lora_B[i, ...] @ self.lora_A[i, ...])
                            * self.lora_scaling[i, ...].unsqueeze(-1)
                            * self.scaling_factor
                        )
                        sign = -1 if mode else 1
                        self.weight.data += sign * delta_w

                print("only forward one time?")
                self.merged = not mode

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                dropout_x = self.lora_dropout(x)
                immediate = torch.einsum(
                    "bsd,ndr->bsnr", dropout_x, self.lora_A.permute(0, 2, 1)
                ).permute(2, 0, 1, 3)
                for i in range(self.lora_num):
                    result += (
                        immediate[i, ...]
                        @ self.lora_B[i, ...].permute(1, 0)
                        * self.lora_scaling[i, ...].unsqueeze(0).unsqueeze(0)
                    ) * self.scaling_factor
                return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
