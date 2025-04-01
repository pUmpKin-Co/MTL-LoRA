import math
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import dropout, linear

from .base import LoRALayer


class LoRALinear(nn.Linear, LoRALayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        **kwargs,
    ) -> None:
        nn.Linear.__init__(
            self, in_features=in_features, out_features=out_features, **kwargs
        )
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
            **kwargs,
        )
        self.fan_in_fan_out = fan_in_fan_out

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (
                        T(self.lora_B.weight @ self.lora_A.weight) * self.scaling
                    )
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (
                        T(self.lora_B.weight @ self.lora_A.weight) * self.scaling
                    )
                self.merged = True

    def forward(self, x: torch.Tensor, statistics=None):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            after_A = self.lora_dropout(x)
            after_A = self.lora_A(after_A)
            after_B = self.lora_B(after_A)
            result = result + after_B * self.scaling

            if statistics is not None:
                statistics["after_A"] = after_A
                statistics["after_B"] = after_B
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
