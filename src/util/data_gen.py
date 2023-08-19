import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DHO_setDataset:
    def __init__(self, k_values, c_values, m_values, t_range, num_samples):
        self.k_values = k_values
        self.c_values = c_values
        self.m_values = m_values
        self.t_range = t_range
        self.num_samples = num_samples

    def exact_function(self, t, k, c, m):
        d = c / (2 * m)
        w0 = np.sqrt(k / m)
        w = np.sqrt((w0**2) - (d**2))
        phi = np.arctan(-d / w)
        A = 1 / (2 * np.cos(phi))
        x = (np.exp(-d * t)) * (2 * A * np.cos(phi + w * t))
        return x

    def generate_set(self, k_value, c_value, m_value):
        data = []
        for t in np.linspace(self.t_range[0], self.t_range[1], self.num_samples):
            x = self.exact_function(t, k_value, c_value, m_value)
            data.append({'t': t, 'k': k_value, 'c': c_value, 'm': m_value, 'x': x})

        df = pd.DataFrame(data)
        return df

    def generate_training_sets(self, num_sets):
        training_sets = {}
        for set_number in range(1, num_sets + 1):
            k_value = np.random.choice(self.k_values)
            c_value = np.random.choice(self.c_values)
            m_value = np.random.choice(self.m_values)
            set_data = self.generate_set(k_value, c_value, m_value)
            training_sets[f"set_{set_number}"] = set_data

        return training_sets

    def generate_testing_set(self, k_value, c_value, m_value):
        return self.generate_set(k_value, c_value, m_value)


class DHO_combDataset:
    def __init__(self, k_values, c_values, m_values, t_range, num_samples):
        self.k_values = k_values
        self.c_values = c_values
        self.m_values = m_values
        self.t_range = t_range
        self.num_samples = num_samples

    def exact_function(self, t, k, c, m):
        d = c / (2 * m)
        w0 = np.sqrt(k / m)
        w = np.sqrt((w0**2) - (d**2))
        phi = np.arctan(-d / w)
        A = 1 / (2 * np.cos(phi))
        x = (np.exp(-d * t)) * (2 * A * np.cos(phi + w * t))
        return x

    def generate_set(self, k_value, c_value, m_value):
        data = []
        for t in np.linspace(self.t_range[0], self.t_range[1], self.num_samples):
            x = self.exact_function(t, k_value, c_value, m_value)
            data.append({'t': t, 'k': k_value, 'c': c_value, 'm': m_value, 'x': x})

        df = pd.DataFrame(data)
        return df

    def generate_dataset(self, num_sets):
        all_data = []
        for _ in range(num_sets):
            k_value = np.random.choice(self.k_values)
            c_value = np.random.choice(self.c_values)
            m_value = np.random.choice(self.m_values)
            set_data = self.generate_set(k_value, c_value, m_value)
            all_data.append(set_data)

        merged_dataset = pd.concat(all_data, ignore_index=True)
        return merged_dataset

