from __future__ import annotations

from typing import Any, Union

import math
import warnings
import importlib

from dataclasses import dataclass, field

import torch
from torch import nn

from transformers.pytorch_utils import Conv1D

from peft import LoraConfig

from peft.tuners import lora
from peft.tuners.lora.gptq import QuantLinear
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.tuners.lora.layer import Conv2d, Embedding, LoraLayer

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.utils import (
    get_auto_gptq_quant_linear,
)
from vmamba import VSSLayer

@dataclass
class GALoraConfig(LoraConfig):
    group_size: int = field(default=32, metadata={"help": "Lora attention dimension"})

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = "GALORA"


class GALoraLayer(LoraLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = tuple(list(lora.LoraLayer.adapter_layer_names) + ['ga_group', 'mamba_group'])
    
    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        super().__init__(base_layer, **kwargs)
         # == GA Start ==
        self.ga_group = nn.ModuleDict({})
        # == GA End ==
        # == Mamba Start ==
        self.mamba_group = nn.ModuleDict({})
        # == Mamba End ==

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, group_size=32, mamba_depth=0):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters

        # == GA Start ==
        L = self.in_features
        self.ga_group[adapter_name] = None
        if group_size > 0:
            assert self.in_features % group_size == 0, f"{self.in_features} is not divisible by {group_size}"
            L = int(self.in_features / group_size)
            self.ga_group[adapter_name] = nn.AvgPool1d(group_size)
        # == GA End ==
        # == Mamba Start ==
        if mamba_depth >= 1:
            self.mamba_group[adapter_name] = VSSLayer(r, mamba_depth)
        # == Mamba End ==
        self.lora_A[adapter_name] = nn.Linear(L, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # check weight and qweight (for GPTQ)
        for weight_name in ("weight", "qweight"):
            weight = getattr(self.get_base_layer(), weight_name, None)
            if weight is not None:
                # the layer is already completely initialized, this is an update
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    self.to(weight.device, dtype=weight.dtype)
                else:
                    self.to(weight.device)
                break
        self.set_adapter(self.active_adapters)


class Linear(nn.Module, GALoraLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        group_size: int = 32,
        mamba_depth: int = 1,
        **kwargs) -> None:

        super().__init__()
        GALoraLayer.__init__(self, base_layer, **kwargs)

        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, group_size, mamba_depth) # GA group size
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

        

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                # == GA Start ==
                ga_group = self.ga_group[active_adapter]
                # == GA End ==
                # == Mamba Start ==
                mamba = self.mamba_group[active_adapter]
                # == Mamba End ==
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)
                # == GA Start ==
                if ga_group is not None:
                    x = ga_group(x)
                # == GA End ==
                # == Mamba Start ==
                if mamba is not None:
                    x = mamba(x)
                # == Mamba End ==
                result += lora_B(lora_A(dropout(x))) * scaling

            result = result.to(torch_result_dtype)
        return result


class GALoraModel(lora.LoraModel):
    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # avoid eager bnb import
        if is_bnb_available():
            import bitsandbytes as bnb

            from peft.tuners.lora.bnb import Linear8bitLt

        if is_bnb_4bit_available():
            from peft.tuners.lora.bnb import Linear4bit

        gptq_quantization_config = kwargs.get("gptq_quantization_config", None)
        AutoGPTQQuantLinear = get_auto_gptq_quant_linear(gptq_quantization_config)

        loaded_in_8bit = kwargs.pop("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.pop("loaded_in_4bit", False)

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        megatron_core = None
        if lora_config.megatron_config:
            megatron_core = importlib.import_module(lora_config.megatron_core)

        if loaded_in_8bit and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(target, adapter_name, **eightbit_kwargs)
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target_base_layer, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target_base_layer.compute_dtype,
                    "compress_statistics": target_base_layer.weight.compress_statistics,
                    "quant_type": target_base_layer.weight.quant_type,
                }
            )
            new_module = Linear4bit(target, adapter_name, **fourbit_kwargs)
        elif AutoGPTQQuantLinear is not None and isinstance(target_base_layer, AutoGPTQQuantLinear):
            new_module = QuantLinear(target, adapter_name, **kwargs)
            target.qweight = target_base_layer.qweight
        elif isinstance(target_base_layer, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            embedding_kwargs.update(lora_config.loftq_config)
            new_module = Embedding(target, adapter_name, **embedding_kwargs)
        elif isinstance(target_base_layer, torch.nn.Conv2d):
            kwargs.update(lora_config.loftq_config)
            new_module = Conv2d(target, adapter_name, **kwargs)
        elif isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            kwargs.update(lora_config.loftq_config)
            # == GA Start ==
            new_module = Linear(target, adapter_name, group_size=lora_config.group_size, mamba_depth=lora_config.mamba_depth, **kwargs)
            # == GA End ==
        elif megatron_core and isinstance(
            target_base_layer,
            (megatron_core.tensor_parallel.ColumnParallelLinear, megatron_core.tensor_parallel.RowParallelLinear),
        ):
            from peft.tuners.lora.tp_layer import LoraParallelLinear

            megatron_kwargs = kwargs.copy()
            megatron_config = lora_config.megatron_config
            if isinstance(megatron_config, dict):
                transformer_config_class = megatron_core.transformer.transformer_config.TransformerConfig
                megatron_config = transformer_config_class(**lora_config.megatron_config)
            megatron_kwargs["megatron_config"] = megatron_config
            if megatron_kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `ColumnParallelLinear` "
                    "or `RowParallelLinear`. "
                    "Setting fan_in_fan_out to False."
                )
                megatron_kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            new_module = LoraParallelLinear(
                base_layer=target, adapter_name=adapter_name, backend=megatron_core.tensor_parallel, **megatron_kwargs
            )
        elif isinstance(target_base_layer, Conv1D):
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            kwargs.update(lora_config.loftq_config)
            new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
            )

        return new_module