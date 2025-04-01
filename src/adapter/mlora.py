#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from .base import LoRALayer, should_gather


class mLoRALinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        B_num: int,
        lambda_num: int,
        diagonal_format: bool,
        B_scale: float = 0.0,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        tunable_scaler: bool = False,
        **kwargs
    ):
        """_summary_

        Parameters
        ----------
        in_features : int. The number of input features
        out_features : int. The number of output features
        B_num : int. The number of B matrices
        lambda_num : int. The number of lambda matrices (e.g., task number)
        diagonal_format : bool. Whether the lambda matrices are diagonal
        B_scale : float, optional. The scale of the B matrices. (e.g., tenpearature)
        r : int, optional. The rank of the LoRA decomposition
        lora_alpha : int, optional. The scaling factor for the LoRA decomposition
        lora_dropout : float, optional. The dropout rate for the LoRA decomposition
        fan_in_fan_out : bool, optional. Whether the layer stores the weight in fan_in, fan_out format
        tunable_scaler : bool, optional. Whether to use a tunable scaler
        """
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
        self.B_num = B_num
        self.lambda_num = lambda_num
        self.diagonal_format = diagonal_format
        self.B_scale = B_scale

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((1, r, in_features)))
            if not self.diagonal_format:
                self.lora_lambdas = nn.Parameter(
                    self.weight.new_zeros((lambda_num, r, r))
                )
            else:
                self.lora_lambdas = nn.Parameter(self.weight.new_zeros((lambda_num, r)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((B_num, out_features, r)))
            self.lora_B_w = nn.Parameter(self.weight.new_zeros((lambda_num, B_num)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            for i in range(self.lora_A.size(0)):
                nn.init.kaiming_uniform_(self.lora_A[i, ...], a=math.sqrt(5))
            if not self.diagonal_format:
                for i in range(self.lora_lambdas.size(0)):
                    nn.init.eye_(self.lora_lambdas[i, ...])
            else:
                nn.init.ones_(self.lora_lambdas)
            nn.init.zeros_(self.lora_B)
            nn.init.kaiming_uniform_(self.lora_B_w, a=math.sqrt(5))

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)

    def forward(self, x: torch.Tensor, lambda_index: torch.Tensor, statistics=None):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        zero_index = torch.zeros_like(lambda_index)
        lora_A = torch.index_select(self.lora_A, 0, zero_index)
        lora_lambdas = torch.index_select(self.lora_lambdas, 0, lambda_index)
        if self.diagonal_format:
            lora_lambdas = torch.diag_embed(lora_lambdas)
        # how to decide the B index? learnable select
        lora_B_w = F.softmax(self.lora_B_w / self.B_scale, dim=-1, dtype=torch.float32)
        lora_B_w = lora_B_w.to(self.lora_B.dtype)
        B_num, out_features, r = self.lora_B.shape
        task_B = lora_B_w @ self.lora_B.view((B_num, -1))
        task_B = task_B.reshape((-1, out_features, r))
        lora_B = torch.index_select(task_B, 0, lambda_index)

        result = F.linear(x, T(self.weight), bias=self.bias)
        if self.r > 0:
            dropout_x = self.lora_dropout(x)
            after_A = torch.bmm(dropout_x, lora_A.transpose(-2, -1))
            after_A = torch.bmm(after_A, lora_lambdas.transpose(-2, -1))
            after_B = torch.bmm(after_A, lora_B.transpose(-2, -1))
            result += (
                after_B * self.scaling * self.compute_tunable_scale(requires_grad=False)
            )
        if statistics is not None:
            statistics["after_A"] = after_A
            statistics["after_B"] = after_B
        return result


class mLoRAMergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        B_num: int,
        lambda_num: int,
        diagonal_format: bool = True,
        B_scale: float = 0.0,
        dec_param: str = "Q.K.V.O",
        lora_param: str = "Q.K.V.O",
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,
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
        assert (
            out_features % len(enable_lora) == 0
        ), "The length of enable_lora must divide out_features"
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        self.B_num = B_num
        self.lambda_num = lambda_num
        self.diagonal_format = diagonal_format
        self.B_scale = B_scale
        self.hidden_size = out_features // len(enable_lora) * sum(enable_lora)
        self.dec_param_set = set(dec_param.split("."))
        self.lora_param_set = set(lora_param.split("."))
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((1, r * sum(enable_lora), in_features))
            )

            if not self.diagonal_format:
                self.lora_lambdas = nn.Parameter(
                    self.weight.new_zeros(
                        (
                            self.lambda_num,
                            r * sum(enable_lora),
                            r * sum(enable_lora),
                        )
                    )
                )
            else:
                self.lora_lambdas = nn.Parameter(
                    self.weight.new_zeros((self.lambda_num, r * sum(enable_lora)))
                )

            if "Q" in self.lora_param_set:
                if "Q" in self.dec_param_set:
                    self.lora_B_q = nn.Parameter(
                        self.weight.new_zeros(
                            (B_num, out_features // len(enable_lora), r)
                        )
                    )
                    self.lora_B_q_w = nn.Parameter(
                        self.weight.new_zeros((lambda_num, B_num))
                    )
                else:
                    self.lora_B_q = nn.Parameter(
                        self.weight.new_zeros((1, out_features // len(enable_lora), r))
                    )
            if "K" in self.lora_param_set:
                if "K" in self.dec_param_set:
                    self.lora_B_k = nn.Parameter(
                        self.weight.new_zeros(
                            (B_num, out_features // len(enable_lora), r)
                        )
                    )
                    self.lora_B_k_w = nn.Parameter(
                        self.weight.new_zeros((lambda_num, B_num))
                    )
                else:
                    self.lora_B_k = nn.Parameter(
                        self.weight.new_zeros((1, out_features // len(enable_lora), r))
                    )
            if "V" in self.lora_param_set:
                if "V" in self.dec_param_set:
                    self.lora_B_v = nn.Parameter(
                        self.weight.new_zeros(
                            (B_num, out_features // len(enable_lora), r)
                        )
                    )
                    self.lora_B_v_w = nn.Parameter(
                        self.weight.new_zeros((lambda_num, B_num))
                    )
                else:
                    self.lora_B_v = nn.Parameter(
                        self.weight.new_zeros((1, out_features // len(enable_lora), r))
                    )

            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features,), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        if hasattr(self, "lora_lambdas"):
            if not self.diagonal_format:
                for i in range(self.lora_lambdas.size(0)):
                    nn.init.eye_(self.lora_lambdas[i, ...])
            else:
                nn.init.ones_(self.lora_lambdas)

        for b, b_w in zip(
            ["lora_B_q", "lora_B_k", "lora_B_v"],
            ["lora_B_q_w", "lora_B_k_w", "lora_B_v_w"],
        ):
            if hasattr(self, b):
                nn.init.zeros_(getattr(self, b))
            if hasattr(self, b_w):
                nn.init.kaiming_uniform_(getattr(self, b_w))

    def zero_pad(self, x):
        x = x.transpose(0, 1)
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1,
            self.out_features // len(self.enable_lora) * sum(self.enable_lora),
        )
        return result.view((*x.shape[:-1], self.out_features)).transpose(0, 1)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)

    def forward(self, x: torch.Tensor, lambda_index: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        batch_size, seq_length, hidden_size = x.shape
        zero_index = torch.zeros_like(lambda_index)

        lora_A = torch.index_select(self.lora_A, 0, zero_index)

        lora_lambdas = torch.index_select(self.lora_lambdas, 0, lambda_index)

        if self.diagonal_format:
            lora_lambdas = torch.diag_embed(lora_lambdas)

        params = []
        for b, b_w in zip(
            ["lora_B_q", "lora_B_k", "lora_B_v"],
            ["lora_B_q_w", "lora_B_k_w", "lora_B_v_w"],
        ):
            if hasattr(self, b):
                lora_B = getattr(self, b)
                if hasattr(self, b_w):
                    lora_B_w = getattr(self, b_w)
                    lora_B_w = F.softmax(
                        lora_B_w / self.B_scale, dim=-1, dtype=torch.float32
                    )
                    lora_B_w = lora_B_w.to(lora_B.dtype)
                    B_num, out_features, r = lora_B.shape
                    task_B = lora_B_w @ lora_B.view((B_num, -1))
                    task_B = task_B.reshape((-1, out_features, r))
                    params.append(torch.index_select(task_B, 0, lambda_index))
                else:
                    params.append(torch.index_select(lora_B, 0, zero_index))

        lora_B = torch.cat(params, dim=1)

        result = F.linear(x, T(self.weight), bias=self.bias)
        if self.r > 0:
            dropout_x = self.lora_dropout(x)
            after_A = torch.bmm(dropout_x, lora_A.transpose(-2, -1))
            after_A = torch.bmm(after_A, lora_lambdas.transpose(-2, -1))
            lora_B = lora_B.view((-1, self.r))
            after_B = (
                F.conv1d(
                    after_A.transpose(-2, -1).reshape(1, -1, seq_length),
                    lora_B.unsqueeze(-1),
                    groups=sum(self.enable_lora) * batch_size,
                )
                .view((batch_size, -1, seq_length))
                .transpose(-2, -1)
            )
            result += (
                self.zero_pad(after_B)
                * self.scaling
                * self.compute_tunable_scale(requires_grad=True)
            )

        return result
