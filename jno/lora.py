"""Public namespace for LoRA adapter classes.

Import via::

    import jno
    net.lora(rank=4, wrapper=jno.lora.rsLoRALinear)

or::

    from jno.lora import rsLoRALinear, DoRALinear, LoRAConv
"""

from .architectures.lora import (
    ConvLike,
    DoRAConv,
    DoRALinear,
    IA3Conv,
    IA3Linear,
    LinearLike,
    LoKrConv,
    LoKrLinear,
    LoRAConv,
    LoRAFAConv,
    LoRAFALinear,
    LoRALinear,
    LoRAWrapper,
    LoRAXSConv,
    LoRAXSLinear,
    MiLoRAConv,
    MiLoRALinear,
    OFTConv,
    OFTLinear,
    PiSSAConv,
    PiSSALinear,
    VeRAConv,
    VeRALinear,
    apply_lora,
    lora_trainable_filter,
    merge_lora,
    partial_lora_trainable_filter,
    rsLoRAConv,
    rsLoRALinear,
)

__all__ = [
    # Base
    "LoRAWrapper",
    "LinearLike",
    "ConvLike",
    # Linear zoo
    "LoRALinear",
    "rsLoRALinear",
    "LoRAFALinear",
    "DoRALinear",
    "PiSSALinear",
    "LoRAXSLinear",
    "VeRALinear",
    "MiLoRALinear",
    "IA3Linear",
    "LoKrLinear",
    "OFTLinear",
    # Conv zoo
    "LoRAConv",
    "rsLoRAConv",
    "LoRAFAConv",
    "DoRAConv",
    "PiSSAConv",
    "LoRAXSConv",
    "VeRAConv",
    "MiLoRAConv",
    "IA3Conv",
    "LoKrConv",
    "OFTConv",
    # Utilities
    "apply_lora",
    "merge_lora",
    "lora_trainable_filter",
    "partial_lora_trainable_filter",
]
