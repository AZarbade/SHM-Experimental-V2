import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm
import time

from util import DampedHarmonicOscillatorDataset, simple_nn, create_tensors

plt.style.use(['science', 'ieee'])
torch.manual_seed(1024)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
torch.device(device)

# Model settings
supervised = True
if supervised == True:
    model_name = f"supervised_pinn"
else:
    model_name = f"unsupervised_pinn"
num_epochs = 200_000
learning_rate = 1e-3
patience = 100  # Number of epochs to wait for loss improvement

# Parameters
k_values = [100, 200, 300, 400, 500] # spring constant (N/m)
c_values = [2, 4, 6, 8, 10] # damping constant (N.s/m)
m_values = [0.5, 1.0, 1.5, 2.0, 2.5] # load mass (kg)
t_range = (0.0, 1.0) # time domin (sec)
num_samples = 1000
num_train_sets = 5  # Number of training sets to generate

# Create the dataset class instance
dataset = DampedHarmonicOscillatorDataset(k_values, c_values, m_values, t_range, num_samples)
# Generate training sets
training_sets = dataset.generate_training_sets(num_train_sets)
test_set = (500, 10, 2.5)
test_dataset = dataset.generate_testing_set(*test_set)

# Loss calculations
def loss_function(model, criterion, features, labels):
    # Data loss
    if supervised == True:
        pred = model(features)
        loss1 = criterion(pred, labels)

    # Boundary loss
    # pred_boundary = model(features[0])
    # loss2 = criterion(pred_boundary, torch.zeros_like(pred_boundary))
    # dxdt = torch.autograd.grad(pred_boundary, features[0], torch.ones_like(pred_boundary), create_graph=True, retain_graph=True)[0]
    # loss3 = criterion(dxdt, torch.zeros_like(dxdt))

    # if supervised == True:
    #     final_loss = loss1 + loss2 + loss3
    # else:
    #     final_loss = loss2 + loss3
    
    final_loss = loss1

    return final_loss


# Model training 
def train_network(train_loader, model, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define early stopping parameters
    best_loss = float("inf")
    counter = 0

    pbar = tqdm(total=num_epochs, desc="Training Progress")
    for epoch in range(num_epochs):
        for features, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_function(model, criterion, features, labels)
            loss.backward()
            optimizer.step()

            loss_curve['set'][set_number].append(set_number)
            loss_curve['epoch'][set_number].append(epoch)
            loss_curve['loss'][set_number].append(loss.item())

        # Check for loss improvement
        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0
        else:
            counter += 1

        # Check if early stopping criteria are met
        if counter >= patience:
            print(f'Early stopping at epoch {epoch+1} due to lack of loss improvement.')
            break

        # Update the progress bar description
        pbar.set_postfix({"Loss": loss.item()})
        pbar.update(1)

loss_curve = {"set": {}, "epoch": {}, "loss": {}}
for set_number, set_data in training_sets.items():
    features = set_data[['t', 'k', 'c', 'm']].values
    labels = set_data[['x']].values

    features_tensor, labels_tensor = create_tensors(features, labels, device)
    dataset = TensorDataset(features_tensor, labels_tensor)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    print(f"Now training on: {set_number}")

    loss_curve['set'][set_number] = []
    loss_curve['epoch'][set_number] = []
    loss_curve['loss'][set_number] = []

    model = simple_nn(4, 1, 32, 8)  # Initialize your neural network
    start_time = time.time()
    train_network(train_loader, model, num_epochs, learning_rate)
    end_time = time.time()  # Stop the timer
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds\n")

print(f"Training finished!")

