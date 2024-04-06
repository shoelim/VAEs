import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from utils import calculate_improved_recall, calculate_mmd
import numpy as np

def plot_training_error(losses, save_path=None):
    fig = plt.figure()  # Create a new figure instance
    plt.plot(losses)
    plt.title('Training Error')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    if save_path:
        plt.savefig(save_path + "_training_error.png")
    else:
        plt.show()

def generate_samples(decoder, latent_dim, device, num_samples=10000):
    with torch.no_grad():
        latent_samples = torch.randn(num_samples, latent_dim).to(device)
        generated_samples = decoder(latent_samples).cpu().detach()
    return generated_samples

def reconstruct_samples(encoder, decoder, latent_dim, device, test_data):
    with torch.no_grad():
        test_data = test_data.to(device)
        mu, logvar = encoder(test_data)
        z = mu + torch.randn_like(mu).to(device) * torch.exp(0.5 * logvar) 
        recon_samples = decoder(z).cpu().detach()
    return recon_samples


def visualize_training_data(X_train, save_path=None):
    fig = plt.figure()  # Create a new figure instance
    plt.scatter(X_train[:, 0], X_train[:, 1], c='blue', marker='o', label='Training Data')
    plt.title('Training Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    if save_path:
        plt.savefig(save_path + "_training_data.png")
    else:
        plt.show()

def visualize_generated_samples(generated_samples, save_path=None):
    fig = plt.figure()  # Create a new figure instance
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], c='orange', marker='+')
    plt.title('Generated Samples')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    if save_path:
        plt.savefig(save_path + "_generated_samples.png")
    else:
        plt.show()

def visualize_reconstructed_samples(reconstructed_samples, save_path=None):
    fig = plt.figure()  # Create a new figure instance
    plt.scatter(reconstructed_samples[:, 0], reconstructed_samples[:, 1], c='orange', marker='+')
    plt.title('Reconstructed Samples')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    if save_path:
        plt.savefig(save_path + "_reconstructed_samples.png")
    else:
        plt.show()

def evaluate_model(X_train, generated_samples, encoder, device, save_path=None):
    # Calculate Improved Recall
    improved_recall = calculate_improved_recall(X_train, generated_samples)
    print(f'Improved Recall Score: {improved_recall:.4f}')

    # Calculate MMD
    mmd = calculate_mmd(X_train, generated_samples)
    print(f'MMD: {mmd:.4f}')

    '''
    # Visualize latent representation using t-SNE
    with torch.no_grad():
        mu_list = []
        for batch in DataLoader(X_train, batch_size=128):
            batch = batch.to(device)
            mu, _ = encoder(batch)
            mu_list.append(mu.cpu().detach().numpy())

        mu_all = np.concatenate(mu_list, axis=0)
        mu_tsne = TSNE(n_components=2).fit_transform(mu_all)

    plt.scatter(mu_tsne[:, 0], mu_tsne[:, 1], c='green', marker='x')
    plt.title('Latent Representation (t-SNE)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    if save_path:
        plt.savefig(save_path + "_latent_representation.png")
    else:
        plt.show()
    '''
