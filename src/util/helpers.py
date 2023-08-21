import torch
import torch.nn as nn
import torch.optim as optim
from util.networks import SimpleNetwork
import matplotlib.pyplot as plt

def plot_loss(loss_log, val_loss):
    """
    Plot training loss over epochs and optionally add a point for validation loss.

    Args:
        loss_log (dict): Dictionary containing 'epoch' and 'loss' values.
        val_loss (float): Validation loss value to plot as a point.

    Returns:
        None
    """
    plt.figure(figsize=(10, 2.5))
    plt.plot(loss_log['epoch'], loss_log['loss'], label='Training Loss')
    if val_loss is not None:
        plt.plot(val_loss['epoch'], val_loss['loss'], label='Validation Loss', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid()
    plt.show()

class PhysicsInformedNN:
    def __init__(self, model, supervised=None):
        """
        Initialize the PhysicsInformedNN class.

        Args:
            model (nn.Module): The neural network model.
            supervised (bool or None): Whether the problem is supervised or not.
        """
        self.model = model
        self.supervised = supervised

    def calculate_loss(self, features, labels, criterion):
        """
        Calculate the total loss for the network outputs.

        Args:
            features (torch.Tensor): Input features tensor.
            labels (torch.Tensor): Target labels tensor.
            criterion (nn.Module): Loss criterion.

        Returns:
            final_loss (torch.Tensor): Total calculated loss.
        """
        preds = self.model(features)

        if self.supervised:
            loss1 = criterion(preds, labels)
        else:
            loss1 = 0

        t_zero_indices = (features[:, 0] == 0)
        if t_zero_indices.any():
            boundary_preds = preds[t_zero_indices]
            loss2 = criterion(boundary_preds, torch.ones_like(boundary_preds))
            loss3 = 0
        else:
            loss2 = 0
            loss3 = 0

        loss4 = 0

        final_loss = loss1 + loss2 + loss3 + loss4

        return final_loss

    def train_and_evaluate(self, features, labels, test_features, test_labels, num_epochs, learning_rate, evaluation_interval=100, patience=1000):
        """
        Train and evaluate the neural network with early stopping.

        Args:
            features (torch.Tensor): Training features tensor.
            labels (torch.Tensor): Training labels tensor.
            test_features (torch.Tensor): Test features tensor.
            test_labels (torch.Tensor): Test labels tensor.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for optimizer.
            evaluation_interval (int): Number of epochs between evaluations.
            patience (int): Number of epochs to wait without improvement before early stopping.

        Returns:
            loss_log (dict): Dictionary containing 'epoch' and 'loss' values.
            test_loss (float): Test loss value.
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        best_loss = float('inf')
        patience_counter = 0
        loss_log = {'epoch': [], 'loss': []}
        val_loss = {'epoch': [], 'loss': []}
        for epoch in range(num_epochs):
            self.model.train()
            optimizer.zero_grad()
            loss = self.calculate_loss(features, labels, criterion)
            loss.backward()
            optimizer.step()

            loss_log['epoch'].append(epoch)
            loss_log['loss'].append(loss.item())

            if (epoch + 1) % evaluation_interval == 0 or epoch == num_epochs - 1:
                self.model.eval()
                with torch.no_grad():
                    test_loss = self.calculate_loss(test_features, test_labels, criterion).item()
                    val_loss['epoch'].append(epoch)
                    val_loss['loss'].append(test_loss)
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}")

                if test_loss < best_loss:
                    best_loss = test_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

        return loss_log, val_loss