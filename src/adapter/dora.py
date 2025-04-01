# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional, Union

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import linear
from .base import LoRALayer, should_gather

def transpose(w, fan_in_fan_out):
    return w.T if fan_in_fan_out else w


class DoRALinear(nn.Linear, LoRALayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        Wdecompose: bool = False,
        dora_simple: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.lora_weight_m_wdecomp = nn.Linear(1,out_features,bias=False) # self.weight_m_wdecomp.weight # shape: out_features, 1

        self.fan_in_fan_out = fan_in_fan_out
        self.Wdecompose = Wdecompose # whether to tune only the magnitude component of Wdecompose or not
        self.dora_simple = dora_simple # whether to use dora simple to save up GPU memory
        if self.Wdecompose == False:
            if r > 0:
                self.lora_A = nn.Linear(in_features, r, bias=False)
                self.lora_B = nn.Linear(r, out_features, bias=False)
                self.scaling = self.lora_alpha / self.r
                # Freezing the pre-trained weight matrix

        self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.Wdecompose == False:
            self.lora_A.train(mode)
            self.lora_B.train(mode)
        self.lora_weight_m_wdecomp.train(mode)

        if not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.Wdecompose:
                norm_scale = ( self.lora_weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1) )
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    new_weight_v = self.weight + transpose(self.lora_B.weight @ self.lora_A.weight, fan_in_fan_out=self.fan_in_fan_out) * self.scaling
                    weight = ( self.lora_weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = True
        elif self.merge_weights and self.merged:
            raise NotImplementedError

    def eval(self):
        nn.Linear.eval(self)
        if self.Wdecompose == False:
            self.lora_A.eval()
            self.lora_B.eval()
        self.lora_weight_m_wdecomp.eval()


    def forward(self, x: torch.Tensor):
        previous_dtype = self.weight.dtype

        if self.Wdecompose and not self.merged:


            norm_scale = self.lora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(self.weight,dim=1))

            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))

            result = org_result + (norm_scale-1) * (F.linear(self.lora_dropout(x), transpose(self.weight, self.fan_in_fan_out)))

            if not self.bias is None:
                    result += self.bias.view(1, -1).expand_as(result)

        elif self.r > 0 and not self.merged:
            
            new_weight_v = self.weight + (self.lora_B.weight @ self.lora_A.weight) * self.scaling

            if self.dora_simple:
                norm_scale = self.lora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1)).detach()
            else:
                norm_scale = self.lora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1))

            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))

            dropout_x = self.lora_dropout(x)

            result = org_result + (norm_scale-1) * (F.linear(dropout_x, transpose(self.weight, self.fan_in_fan_out)))

            if not self.bias is None:
                    result += self.bias.view(1, -1).expand_as(result)

            result += ( norm_scale * (self.lora_B(self.lora_A(dropout_x.to(self.lora_A.weight.dtype))))) * self.scaling
            
        else:
             result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result
