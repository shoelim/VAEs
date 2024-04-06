import torch
from sklearn.manifold import TSNE
import numpy as np
import sklearn.datasets
import torch.nn.functional as F
import torch.nn as nn


def elbo_loss(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def calculate_mmd(real_samples, generated_samples, device='cuda'):
    real_samples = real_samples.to(device)
    generated_samples = generated_samples.to(device)

    def kernel(x, y):
        return torch.exp(-torch.mean((x.unsqueeze(1) - y.unsqueeze(0)).pow(2), dim=2) / 2.0)

    n = real_samples.size(0)
    m = generated_samples.size(0)
    real_kernel = kernel(real_samples, real_samples)
    gen_kernel = kernel(generated_samples, generated_samples)
    cross_kernel = kernel(real_samples, generated_samples)
    mmd = torch.mean(real_kernel) + torch.mean(gen_kernel) - 2 * torch.mean(cross_kernel)

    return mmd.item()

def calculate_recall(real_samples, generated_samples, k=5):
    real_distances = torch.cdist(real_samples, real_samples)
    gen_distances = torch.cdist(generated_samples, generated_samples)

    real_indices = torch.topk(real_distances, k=k+1, dim=1, largest=False, sorted=True)[1][:, 1:]
    gen_indices = torch.topk(gen_distances, k=k, dim=1, largest=False, sorted=True)[1]

    recall = (torch.isin(real_indices, gen_indices)).sum().item() / (real_samples.size(0) * k)

    return recall

def calculate_improved_recall(real_samples, generated_samples, device='cuda'):
    real_samples = real_samples.to(device)
    generated_samples = generated_samples.to(device)

    recall = calculate_recall(real_samples, generated_samples)

    mmd = calculate_mmd(real_samples, generated_samples, device)

    improved_recall = recall - mmd

    return improved_recall
