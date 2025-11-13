import yaml

import subprocess
import pandas as pd

def read_config(path: str, file_name: str):
    """Read network config from YAML file"""

    with open(f"{path}/{file_name}", "r", encoding="utf-8") as file:
        data = yaml.load(file, Loader=yaml.SafeLoader)
        return data