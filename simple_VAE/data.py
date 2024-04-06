import torch
import numpy as np
import sklearn.datasets

def prepare_balanced_mixture_gaussians_data(train_dataset_size=20000):
    np.random.seed(123)
    data1 = np.random.randn(int(train_dataset_size/2), 2) * 0.5 + np.array([10, 10])  # Mean at [10, 10]
    data2 = np.random.randn(train_dataset_size - int(train_dataset_size/2), 2) * 0.1 + np.array([-10, -10])  # Mean at [-10, 10]
    data = np.concatenate([data1, data2], axis=0)
    np.random.shuffle(data)
    return torch.tensor(data, dtype=torch.float32)

def prepare_imbalanced_mixture_gaussians_data(train_dataset_size=20000):
    np.random.seed(123)
    data1 = np.random.randn(train_dataset_size // 10, 2) * 0.5 + np.array([10, 10])  # Mean at [10, 10]
    data2 = np.random.randn(train_dataset_size - train_dataset_size // 10, 2) * 0.1 + np.array([-10, -10])  # Mean at [-10, 10]
    data = np.concatenate([data1, data2], axis=0)
    np.random.shuffle(data)
    return torch.tensor(data, dtype=torch.float32)
    
def prepare_swissroll_data(train_dataset_size=20000):
    data = sklearn.datasets.make_swiss_roll(
                    n_samples=train_dataset_size,
                    noise=0.25,
                    random_state=123
                )[0]
    data = data.astype('float32')[:, [0, 2]]
    data /= 7.5 # stdev plus a little
    return torch.tensor(data, dtype=torch.float32)
