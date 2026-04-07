import json
from pathlib import Path

class ModelConfig:
    def __init__(self, model_name: str, config_path: Path):
        self.model_name = model_name
        with open(config_path, "r") as f:
            config = json.load(f)
            for key, value in config[model_name].items():
                setattr(self, key, value)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value: any):
        setattr(self, key, value)

    def __delitem__(self, key: str):
        delattr(self, key)
        