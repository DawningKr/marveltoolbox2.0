from ..lora import LoRA, LoRALayer
from .exceptions import ModelModificationError


def lora_state_dict(model):
    state_dict = model.state_dict()
    return {key: state_dict[key] for key in state_dict.keys() if "lora" in key}


def set_lora_configs_all(model, rank: int, alpha: int, enable_lora=True):
    # Before calling this function, check the type of the input model
    if isinstance(model, LoRA):
        lora_layers = model.get_lora_layers()
        for layer in lora_layers:
            layer.set_lora_configs(rank, alpha)
            layer.set_lora_status(enable_lora)
    elif isinstance(model, LoRALayer):
        model.set_lora_configs(rank, alpha)
        model.set_lora_status(enable_lora)
    else:
        raise ModelModificationError()
    return model
