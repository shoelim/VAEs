import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import elbo_loss

def train_vae(train_loader, encoder, decoder, optimizer, num_epochs, device, dataset, batch_size):
    encoder.train()
    decoder.train()

    training_losses = []

    for epoch in range(num_epochs):
        epoch_losses = []

        for batch in train_loader:
            #if dataset == "mnist":
            #    data, _ = next(iter(train_loader))
            #    batch = data.view((batch_size, -1)).to(device)
            #else:
            #    batch = batch.to(device)

            batch = batch.to(device)
            optimizer.zero_grad()
            mu, logvar = encoder(batch)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) 
            recon_batch = decoder(z)
            loss = elbo_loss(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        epoch_loss = torch.tensor(epoch_losses).mean().item()
        training_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    return training_losses
