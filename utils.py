from safetensors import safe_open
from pathlib import Path
from glob import glob
import torch
import torch.nn as nn

def load_from_safetensors(model: nn.Module, model_path: Path) -> None:
    for file in glob(str(model_path / "*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                if weight_name in model.state_dict():
                    model.get_parameter(weight_name).data.copy_(f.get_tensor(weight_name))