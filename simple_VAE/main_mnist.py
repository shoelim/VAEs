#MNIST
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import time

def set_seed(seed=12345):
    np.random.seed(seed)
    torch.manual_seed(seed)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        mu = self.fc2(x)
        logvar = self.fc3(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

def elbo_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def visualize_samples(samples, title='Generated Samples', rows=8, cols=8, figsize=(12, 12)):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(samples[i*cols+j], cmap='gray')
            axes[i, j].axis('off')
    plt.suptitle(title)
    plt.show()
    plt.savefig("/mimer/NOBACKUP/groups/snic2021-7-147/VAE-train_mnist_samples.png")

#=============== main ==============
set_seed(12345)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten the image
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = 784
hidden_dim = 512
latent_dim = 64

encoder = Encoder(input_dim, hidden_dim, latent_dim).to(device)
decoder = Decoder(latent_dim, hidden_dim, input_dim).to(device)

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))

num_epochs = 50

start_time = time.time()
for epoch in range(num_epochs):
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        mu, logvar = encoder(data)
        z = torch.randn_like(mu) * torch.exp(0.5 * logvar) + mu
        recon_batch = decoder(z)
        loss = elbo_loss(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

end_time = time.time()
training_time = end_time - start_time
print(f"Training Time: {training_time:.2f} seconds")

# Generate samples
start_time = time.time()
with torch.no_grad():
    latent_samples = torch.randn(64, latent_dim).to(device)
    generated_samples = decoder(latent_samples).cpu().detach().view(-1, 28, 28)
end_time = time.time()
sampling_time = end_time - start_time
print(f"Sampling Time: {sampling_time:.2f} seconds")

visualize_samples(generated_samples, title='Generated Samples')

'''
# Visualize latent representation using t-SNE
tsne_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
tsne_loader = DataLoader(tsne_dataset, batch_size=1000, shuffle=True)

with torch.no_grad():
    mu_list = []
    labels_list = []
    for data, labels in tsne_loader:
        data = data.to(device)
        mu, _ = encoder(data)
        mu_list.append(mu.cpu().detach().numpy())
        labels_list.append(labels.numpy())

    mu_all = np.concatenate(mu_list, axis=0)
    labels_all = np.concatenate(labels_list, axis=0)
    mu_tsne = TSNE(n_components=2).fit_transform(mu_all)

plt.figure(figsize=(8, 6))
for i in range(10):
    indices = labels_all == i
    plt.scatter(mu_tsne[indices, 0], mu_tsne[indices, 1], label=str(i))
plt.title('Latent Representation (t-SNE)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.show()
'''
