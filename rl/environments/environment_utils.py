import logging
from typing import Optional


def get_param_from_config(config: dict, key: str, logger: Optional[logging.Logger]):
    """
    Extract parameter from the configuration dict

    @param config: config dictionary
    @param key: key the parameter is stored under in the config yaml
    @param logger: current logger. if None, nothing will be logged
    @raise Value Error if the key cannot be found
    @return: the parameter's value
    """
    if key not in config:
        if logger is not None:
            logger.error(f"key {key} is could not be found in the configuration!")
        raise ValueError
    return config[key]
