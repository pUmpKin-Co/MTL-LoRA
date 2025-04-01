import os
from functools import partial
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from ..adapter import (
    DoRALinear,
    LoRALinear,
    MMOELoraLinear,
    MultiLoRALinear,
    mLoRALinear,
    mLoRAMergedLinear,
)
from .dist import get_global_rank

ADAPTER_MAPPING = {
    "mlora": mLoRALinear,
    "mlora_merged": mLoRAMergedLinear,
    "multilora": MultiLoRALinear,
    "moelora": MMOELoraLinear,
    "dora": DoRALinear,
    "lora": LoRALinear,
}


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_lora_param_maybe_zero_3(named_params, valid_keys: List[str] = ["lora"]):
    to_return = {k: v for k, v in named_params if any([x in k for x in valid_keys])}
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def save_pretrain(
    model,
    output_dir: str,
    prefix: List[str] = [
        "lora",
    ],
) -> None:
    state_dict = model.state_dict()
    if get_global_rank() == 0:
        return_dict = get_lora_param_maybe_zero_3(state_dict.items(), valid_keys=prefix)
        output_dit = os.path.join(output_dir, "checkpoint")
        os.makedirs(output_dit, exist_ok=True)
        path = os.path.join(output_dit, f"final_checkpoint.pt")
        torch.save(return_dict, path)
        print(f"Model saved to {path}")


def set_no_grad(model, logger=None) -> None:
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_params += num_params
        if param.requires_grad:
            print(name)
            trainable_params += num_params

    if logger is not None:
        logger.info(
            f"Trainable params: {trainable_params:,d}, All params: {all_params:,d}, trainable: {100 * trainable_params/all_params:.2f}"
        )
    else:
        print(
            f"Trainable params: {trainable_params:,d}, All params: {all_params:,d}, trainable: {100 * trainable_params/all_params:.2f}"
        )


def wrap_model(
    model: nn.Module,
    target_layer: Tuple[str],
    adapter_config: Dict[str, Any],
):
    def _get_submodule(key):
        parent = model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = model.get_submodule(key)
        return parent, target, target_name

    adapter_type = adapter_config.pop("type", "").lower()
    assert (
        adapter_type in ADAPTER_MAPPING.keys()
    ), f"Adapter {adapter_type} not supported"

    adapter_class = ADAPTER_MAPPING[adapter_type]
    adapter_impl = partial(adapter_class, **adapter_config)

    key_list = [key for key, _ in model.named_modules()]

    for key in key_list:
        if any([layer in key for layer in target_layer]):
            parent, target, target_name = _get_submodule(key)
            if isinstance(parent, adapter_class):
                continue

            adapter_module = adapter_impl(
                in_features=target.in_features,
                out_features=target.out_features,
            )
            adapter_module.to(device=model.device, dtype=model.dtype)
            adapter_module.load_state_dict(target.state_dict(), strict=False)
            setattr(parent, target_name, adapter_module)

    return model
