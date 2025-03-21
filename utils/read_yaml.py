import os
import yaml
from typing import Dict, Any


def read_yaml(file: str) -> Dict[str, Any]:
    with open(file, "r") as yaml_file:
        configurations = yaml.load(yaml_file, Loader=yaml.FullLoader)

    fcn = os.path.basename(file).replace(".yaml", '')
    print(f"WARNING: Setting fixed_checkpoint_name as {fcn}")
    configurations["fixed_checkpoint_name"] = fcn

    return configurations
