# Mixture of Experts (MoE) Transformer with Llama4 type model
# Author: Tirth Shah
# Inspired by: https://github.com/FareedKhan-dev/train-llama4
# Training script

# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model import MoETransformer
from tqdm import trange
import matplotlib.pyplot as plt
import os

class TrainModel:
    def __init__(self, 
                 model: MoETransformer,
                 batch_size: int=16, 
                 learning_rate: float=5e-4, 
                 epochs: int=3000, 
                 print_interval: int=300, 
                 datadir: str="./dataset/alice_sample_dataset.pt", 
                 test_split: float=0.1
                 ):
        """
        Initializes the training configuration for the MoE Transformer model. 

        Parameters
        ----------
        model : MoETransformer
            The MoE Transformer model to be trained.

        batch_size : int
            The number of samples per batch during training (default is 16).

        learning_rate : float
            The learning rate for the optimizer (default is 5e-4).

        epochs : int
            The total number of epochs to train the model (default is 3000).
        
        print_interval : int
            The number of epochs after which to print the training loss (default is 300).

        datadir : str
            The directory where the training dataset is stored (default is "./dataset/alice_sample_dataset.pt

        test_split : float
            The fraction of the dataset to be used for validation (default is 0.1).
        """

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.print_interval = print_interval
        self.model = model

        self.dataset = torch.load(datadir, weights_only=False)
        
        # default optimizer chosen is AdamW
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # Compute split sizes
        test_size = int(len(self.dataset) * test_split)
        train_size = len(self.dataset) - test_size

        # Split the dataset
        train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size])

        # Create data loaders
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Device configuration
        self.device = self.model.config.device

        # Training losses
        self.train_losses = []

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

    def plot_training_curve(self, train_losses):
        """
        Plots the training loss curve over epochs.

        Parameters
        ----------
        train_losses : list
            A list of training losses recorded at each epoch.
        """

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses)
        plt.title('Training Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
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

    def train(self):
        """
        Trains the MoE Transformer model using the specified configuration. It returns the training and validation losses over epochs.
        """

        # Move model to the specified device
        self.model.to(self.device)
        
        # Training loop
        print("Starting training...")
        for epoch in trange(self.epochs):
            xb, yb = next(iter(self.train_dataloader))
            xb, yb = xb.to(self.device), yb.to(self.device) # Move data to device

            logits = self.model(xb) # Forward pass; (B, T, vocab_size)
            loss = self.loss_fn(logits, yb) # Compute loss
            self.optimizer.zero_grad() # Zero Gradients
            loss.backward() # Backward pass
            self.optimizer.step() # Update parameters
                
            current_loss = loss.item()
            self.train_losses.append(current_loss)

            if (epoch+1) % self.print_interval == 0 or epoch == self.epochs - 1:
                print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {current_loss:.4f}")

        # Plot the training curve
        self.plot_training_curve(self.train_losses)

        print("Training completed.")