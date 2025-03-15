class ModelModificationError(Exception):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Target model does not have LoRA functionality"