# Mixture of Experts (MoE) Transformer with Llama4 type model
# Author: Tirth Shah
# Inspired by: https://github.com/FareedKhan-dev/train-llama4

# ------------------------------------------------------------------------------

import os
import torch
from torch import nn

def save_model(model: nn.Module, save_dir: str, save_name: str):
    """
    Saves the trained model to the specified path.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to be saved.
    save_dir : str
        The directory where the model will be saved.
    save_name : str
        The file name for the saved model.
    """

    # if the directory does not exist, create it
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, save_name)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")