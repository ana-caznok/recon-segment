from typing import Dict, Any


def config_flatten(config: Dict[str, Any], flat_config: Dict[str, Any], current_key: str = ''):
    '''
    Recursively flattens configs inplace from config to flat_config without losing track of nested keys
    '''
    for key, value in config.items():
        if isinstance(value, dict):
            config_flatten(value, flat_config, (current_key + '_' + key).strip('_'))
        else:
            flat_config[(current_key + '_' + key).strip('_')] = value
