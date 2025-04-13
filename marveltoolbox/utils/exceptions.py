class ConfigurationError(Exception):
    def __init__(self, message):
        super().__init__()
        self.message = message

    def __str__(self):
        return f'Parameter "{self.message}" is not set in the configuration file.'
