# ------------------------------------------------------------------------------
# Mixture of Experts (MoE) Transformer - Training Script
# ------------------------------------------------------------------------------
# Author: Tirth Shah
# Inspired by: https://github.com/FareedKhan-dev/train-llama4
#
# This script implements the training loop for the MoE Transformer project.
# It includes:
#   - TrainConfig: Configuration class for training hyperparameters.
#   - TrainModel: Handles model training, loss computation, logging, and evaluation.
#   - Functions for tracking routing entropy and expert utilization during training.
#   - Utilities for plotting training curves and evaluating test loss/accuracy.
#
# Use this script to train and evaluate MoE Transformer models on character-level or TinyStories datasets.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model import MoETransformer
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np

class TrainConfig:
    def __init__(self, 
                 batch_size: int=16, 
                 learning_rate: float=5e-4, 
                 steps: int=3000, 
                 print_interval: int=300, 
                 test_split: float=0.1):
        """
        Initializes the training configuration.

        Parameters
        ----------
        batch_size : int
            The number of samples per batch during training (default is 16).

        learning_rate : float
            The learning rate for the optimizer (default is 5e-4).

        steps : int
            The total number of steps to train the model (default is 3000).
        
        print_interval : int
            The number of steps after which to print the training loss (default is 300).

        test_split : float
            The fraction of the dataset to be used for test loss computation (default is 0.1).
        """

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.steps = steps
        self.print_interval = print_interval
        self.test_split = test_split

class TrainModel:
    def __init__(self, 
                 model: MoETransformer,
                 train_config: TrainConfig,
                 dataset,
                 ):
        """
        Initializes the training configuration for the MoE Transformer model. 

        Parameters
        ----------
        model : MoETransformer
            The MoE Transformer model to be trained.

        train_config : TrainConfig
            The training configuration containing batch size, learning rate, steps, print interval, 
            and test split.

        dataset : Dataset
            The dataset to be used for training and validation.
        """

        self.train_config = train_config
        self.model = model
        self.dataset = dataset

        # default optimizer chosen is AdamW
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.train_config.learning_rate)

        # Compute split sizes
        test_size = int(len(self.dataset) * self.train_config.test_split)
        train_size = len(self.dataset) - test_size

        # Split the dataset
        train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size])

        # Create data loaders
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.train_config.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.train_config.batch_size, shuffle=False)

        # Device configuration
        self.device = self.model.config.device

        # During training, we will log losses and routing entropies
        self.train_losses = []
        self.routing_entropies = []
        self.expert_utilizations = []

    def loss_fn(self, logits, targets):
        """
        Computes the cross-entropy loss between the model's logits and the target labels.

        Parameters
        ----------
        logits : torch.Tensor
            The output logits from the model (shape: [batch_size, sequence_length, vocab_size]).

        targets : torch.Tensor
            The target labels (shape: [batch_size, sequence_length]).

        Returns
        -------
        torch.Tensor
            The computed loss value.
        """

        B, T, V = logits.shape

        return nn.CrossEntropyLoss()(logits.view(B * T, V), targets.view(B * T))

    def plot_training_curve(self, train_losses, window_size=30):
        """
        Plots the training loss curve over steps.

        Parameters
        ----------
        train_losses : list
            A list of training losses recorded at each step.

        window_size : int
            The size of the moving window for smoothing the training loss curve (default is 30).
        """
        
        train_losses_ma = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
        
        plt.figure(figsize=(10, 5))
        actual_line = plt.plot(train_losses, label='Training Loss', alpha=0.6)[0]
        if len(train_losses) > window_size:
            color = actual_line.get_color()
            plt.plot(np.arange(window_size - 1, len(train_losses)), train_losses_ma, color=color, linewidth=2)
        plt.title('Training Loss Curve')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid()
        plt.show()

    def test_loss(self):
        """
        Computes the average loss on the test dataset.

        Returns
        -------
        float
            The average loss on the test dataset.
        """

        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for xb, yb in self.test_dataloader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)  # Forward pass
                loss = self.loss_fn(logits, yb)  # Compute loss
                total_loss += loss.item() * xb.size(0)  # Accumulate loss
                total_samples += xb.size(0)  # Count samples

        return total_loss / total_samples if total_samples > 0 else 0.0
    
    def test_top_k_accuracy(self, k=5):
        """
        Computes the top-k accuracy on the test dataset.

        Parameters
        ----------
        k : int
            The number of top predictions to consider for accuracy (default is 5).

        Returns
        -------
        float
            The top-k accuracy as a percentage.
        """

        self.model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for xb, yb in self.test_dataloader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)  # Forward pass
                total_correct += self._top_k_accuracy(logits, yb, k) * xb.size(0)  # Accumulate correct predictions
                total_samples += xb.size(0)  # Count samples

        return total_correct / total_samples if total_samples > 0 else 0.0
    
    def _top_k_accuracy(self, logits, targets, k=5):
        """
        Computes the top-k accuracy for the model's predictions.

        Parameters
        ----------
        logits : torch.Tensor
            The output logits from the model (shape: [batch_size, sequence_length, vocab_size]).

        targets : torch.Tensor
            The target labels (shape: [batch_size, sequence_length]).

        k : int
            The number of top predictions to consider for accuracy (default is 5).

        Returns
        -------
        float
            The top-k accuracy as a percentage.
        """
        
        _, top_k_indices = logits.topk(k, dim=-1)  # Get top-k indices
        correct = (top_k_indices == targets.unsqueeze(-1)).any(dim=-1)  # Check if any of the top-k indices match the targets
        
        return correct.float().mean().item() * 100

    def train(self):
        """
        Trains the MoE Transformer model using the specified configuration. It returns the training 
        and validation losses over steps.
        """

        # Move model to the specified device
        self.model.to(self.device)
        
        # Training loop
        print("Starting training...")
        for step in trange(self.train_config.steps):
            xb, yb = next(iter(self.train_dataloader))
            xb, yb = xb.to(self.device), yb.to(self.device) # Move data to device

            logits = self.model(xb) # Forward pass; (B, T, vocab_size)
            loss = self.loss_fn(logits, yb) # Compute loss
            self.optimizer.zero_grad() # Zero Gradients
            loss.backward() # Backward pass
            self.optimizer.step() # Update parameters
                
            current_loss = loss.item()
            self.train_losses.append(current_loss)

            # compute routing entropy
            routing_entropy = [self.model.moe_layers[i].compute_routing_entropy() for i in range(self.model.config.n_layers)]
            self.routing_entropies.append(routing_entropy)

            # compute expert utilization
            expert_utilization = [self.model.moe_layers[i].compute_expert_utilization() for i in range(self.model.config.n_layers)]
            self.expert_utilizations.append(expert_utilization)

            if (step+1) % self.train_config.print_interval == 0 or step == self.train_config.steps - 1:
                print(f"Step {step+1}/{self.train_config.steps}, Loss: {current_loss:.4f}")

        self.routing_entropies = np.array(self.routing_entropies)
        self.expert_utilizations = np.array(self.expert_utilizations)
        
        # Plot the training curve
        self.plot_training_curve(self.train_losses)

        print("Training completed.")