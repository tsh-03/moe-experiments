# Mixture of Experts (MoE) Transformer with Llama4 type model
# Author: Tirth Shah
# Inspired by: https://github.com/FareedKhan-dev/train-llama4

# ------------------------------------------------------------------------------

import os
import torch
from torch import nn
import pickle
from model import MoETransformer
from train import TrainModel

def save_model(train_obj: TrainModel, path: str):
    """
    Saves the trained model to the specified path.

    Parameters
    ----------
    train_obj : TrainModel
        The training object containing model, model configuration, training configuration, and training history.
    path : str
        The directory as well as filename where the model will be saved. If the directory does not exist, it will be created.
    """

    # if the directory does not exist, create it
    os.makedirs('./saved_models/', exist_ok=True)
    
    # save the model
    torch.save({
        'model_state_dict': train_obj.model.state_dict(),
        'config': train_obj.model.config,
        'train_config': train_obj.train_config,
        'train_losses': train_obj.train_losses,
        'routing_entropies': train_obj.routing_entropies,
        'expert_utilizations': train_obj.expert_utilizations,
    }, path)

    print(f"Model saved to {path}")

def load_model(path: str) -> nn.Module:
    """
    Loads a model from the specified path.

    Parameters
    ----------
    path : str
        The path to the saved model file.
    
    Returns
    -------
    nn.Module
        The loaded PyTorch model.
    """

    # load the model state dict and config
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    # load training configuration and history if available
    train_config = checkpoint.get('train_config', None)
    train_losses = checkpoint.get('train_losses', None)
    routing_entropies = checkpoint.get('routing_entropies', None)
    expert_utilizations = checkpoint.get('expert_utilizations', None)
    
    #-----Create a new model instance with the loaded config-----
    # create a new instance of the model with the loaded config
    model = MoETransformer(checkpoint['config'])
    
    # load the state dict into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {path}")

    return model, train_config, train_losses, routing_entropies, expert_utilizations