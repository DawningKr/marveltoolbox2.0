def lora_state_dict(model):
    state_dict = model.state_dict()
    return {key: state_dict[key] for key in state_dict.keys() if "lora" in key}


def set_lora_configs_all(model, rank: int, alpha: int, enable_lora=True):
    lora_layers = model.get_lora_layers()
    for layer in lora_layers:
        layer.set_lora_configs(rank, alpha)
        if enable_lora:
            layer.set_lora_status(True)
    return model
