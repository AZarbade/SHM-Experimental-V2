import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class DHO_Dataset:
    def __init__(self, k_values, c_values, m_values, t_range, num_samples):
        # Initialize the DHO_Dataset class with provided parameters
        self.k_values = k_values
        self.c_values = c_values
        self.m_values = m_values
        self.t_range = t_range
        self.num_samples = num_samples

    def exact_function(self, t, k, c, m):
        """
        Calculate the exact displacement of the DHO system at a given time t.

        Args:
            t (float): Time value.
            k (float): Spring constant.
            c (float): Damping coefficient.
            m (float): Mass.

        Returns:
            x (float): Displacement at time t.
        """
        d = c / (2 * m)
        w0 = np.sqrt(k / m)
        w = np.sqrt((w0**2) - (d**2))
        phi = np.arctan(-d / w)
        A = 1 / (2 * np.cos(phi))
        x = (np.exp(-d * t)) * (2 * A * np.cos(phi + w * t))
        return x

    def generate_set(self, k_value, c_value, m_value):
        """
        Generate a dataset of DHO system samples for a specific combination of k, c, and m.

        Args:
            k_value (float): Spring constant value.
            c_value (float): Damping coefficient value.
            m_value (float): Mass value.

        Returns:
            df (pandas.DataFrame): Generated dataset.
        """
        data = []
        for t in np.linspace(self.t_range[0], self.t_range[1], self.num_samples):
            x = self.exact_function(t, k_value, c_value, m_value)
            data.append({'t': t, 'k': k_value, 'c': c_value, 'm': m_value, 'x': x})

        df = pd.DataFrame(data)
        return df

    def generate_testing_set(self, k_value, c_value, m_value):
        """
        Generate a testing dataset using the same logic as generate_set.

        Args:
            k_value (float): Spring constant value.
            c_value (float): Damping coefficient value.
            m_value (float): Mass value.

        Returns:
            df (pandas.DataFrame): Generated testing dataset.
        """
        return self.generate_set(k_value, c_value, m_value)

    def generate_training_sets(self, num_sets):
        """
        Generate multiple training datasets by randomly selecting k, c, and m values.

        Args:
            num_sets (int): Number of training datasets to generate.

        Returns:
            merged_dataset (pandas.DataFrame): Merged training dataset.
        """
        all_data = []
        for _ in range(num_sets):
            k_value = np.random.choice(self.k_values)
            c_value = np.random.choice(self.c_values)
            m_value = np.random.choice(self.m_values)
            set_data = self.generate_set(k_value, c_value, m_value)
            all_data.append(set_data)

        merged_dataset = pd.concat(all_data, ignore_index=True)
        return merged_dataset
    
    def PreprocessData(self, training_sets):
        """
        Preprocesses the input training data using standard scaling.

        Args:
            training_sets (pandas.DataFrame): The input training data containing columns 't', 'k', 'c', 'm', and 'x'.

        Returns:
            features (numpy.ndarray): Scaled feature matrix containing columns 't', 'k', 'c', and 'm'.
            labels (pandas.DataFrame): Labels from the 'x' column.
        """
        # Initialize a StandardScaler and fit it to the training data
        scaler = StandardScaler().fit(training_sets[['t', 'k', 'c', 'm']])
        # Scale the features and extract the labels
        features = scaler.transform(training_sets[['t', 'k', 'c', 'm']])
        labels = np.array(training_sets[['x']])

        return features, labels
    
    def CreateTensor(self, features, labels, device):
        """
        Converts the given features and labels into tensors.

        Args:
            features (numpy.ndarray): Scaled feature matrix.
            labels (pandas.DataFrame): Labels.
            device (str): Device to which tensors will be moved (e.g., 'cpu' or 'cuda').

        Returns:
            features_tensor (torch.Tensor): Tensor containing scaled features.
            labels_tensor (torch.Tensor): Tensor containing labels.
        """
        # Convert features and labels to tensors with the specified device
        features_tensor = torch.tensor(features, dtype=torch.float32, requires_grad=True, device=device)
        labels_tensor = torch.tensor(labels, dtype=torch.float32, requires_grad=True, device=device)
        
        return features_tensor, labels_tensor
    
    def plot_training_dataset(self, training_sets):
        """
        Plot the training dataset with time (t) on the x-axis and displacement (x) on the y-axis.

        Args:
            training_sets (pandas.DataFrame): The input training data containing columns 't' and 'x'.
        """
        plt.figure(figsize=(10, 4))
        
        # Iterate over each unique combination of k, c, and m values
        for (k_value, c_value, m_value), subset in training_sets.groupby(['k', 'c', 'm']):
            plt.plot(subset['t'], subset['x'], label=f'k={k_value}, c={c_value}, m={m_value}')
        
        plt.xlabel('Time (t)')
        plt.ylabel('Displacement (x)')
        plt.title('Training Dataset - Damped Harmonic Oscillator')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_testing_dataset(self, testing_sets):
        """
        Plot the testing dataset with time (t) on the x-axis and displacement (x) on the y-axis.

        Args:
            testing_sets (pandas.DataFrame): The input testing data containing columns 't' and 'x'.
        """
        plt.figure(figsize=(10, 4))
        
        # Iterate over each unique combination of k, c, and m values
        for (k_value, c_value, m_value), subset in testing_sets.groupby(['k', 'c', 'm']):
            plt.plot(subset['t'], subset['x'], label=f'k={k_value}, c={c_value}, m={m_value}')
        
        plt.xlabel('Time (t)')
        plt.ylabel('Displacement (x)')
        plt.title('Testing Dataset - Damped Harmonic Oscillator')
        plt.legend()
        plt.grid()
        plt.show()