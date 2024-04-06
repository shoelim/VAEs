import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models import Encoder, Decoder
from train import train_vae
from evaluate import visualize_training_data, visualize_generated_samples, evaluate_model, plot_training_error, generate_samples
from data import prepare_balanced_mixture_gaussians_data, prepare_imbalanced_mixture_gaussians_data, prepare_swissroll_data
from torchvision import datasets, transforms
import numpy as np

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == "balanced_mixture_gaussians":
        X_train = prepare_balanced_mixture_gaussians_data(train_dataset_size=args.train_dataset_size)
        train_loader = DataLoader(X_train, batch_size=args.batch_size, shuffle=True)
    elif args.dataset == "imbalanced_mixture_gaussians":
        X_train = prepare_imbalanced_mixture_gaussians_data(train_dataset_size=args.train_dataset_size)
        train_loader = DataLoader(X_train, batch_size=args.batch_size, shuffle=True)
    elif args.dataset == "swissroll":
        X_train = prepare_swissroll_data(train_dataset_size=args.train_dataset_size)
        train_loader = DataLoader(X_train, batch_size=args.batch_size, shuffle=True)
    #elif args.dataset == "mnist":
    #    transform = transforms.Compose([transforms.ToTensor(),
    #                                    transforms.Normalize((0.5,), (0.5,))
    #                ])
    #    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    #    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        raise ValueError("Invalid dataset name. Please provide a valid dataset name.")


    encoder = Encoder(args.input_dim, args.hidden_dim, args.latent_dim).to(device)
    decoder = Decoder(args.latent_dim, args.hidden_dim, args.input_dim).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learning_rate)

    print("Training VAE...")
    training_losses = train_vae(train_loader, encoder, decoder, optimizer, args.num_epochs, device, args.dataset, args.batch_size)
    print("Training completed.")

    if args.save_path:
        plot_training_error(training_losses, save_path=args.save_path)
    else:
        plot_training_error(training_losses)

    visualize_training_data(X_train, save_path=args.save_path)
    
    print("Generating samples...")
    generated_samples = generate_samples(decoder, args.latent_dim, device, num_samples=20000)
    visualize_generated_samples(generated_samples, save_path=args.save_path)

    print("Evaluating model...")
    evaluate_model(X_train, generated_samples, encoder, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variational Autoencoder (VAE)")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save plots")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--input_dim", type=int, default=2, help="Dimensionality of input data")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Dimensionality of hidden layers")
    parser.add_argument("--latent_dim", type=int, default=2, help="Dimensionality of latent space")
    parser.add_argument("--dataset", type=str, default="balanced_mixture_gaussians", help="Name of dataset (e.g., mixture_gaussians)")
    parser.add_argument("--train_dataset_size", type=int, default=20000, help="Size of training dataset")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    args = parser.parse_args()
    main(args)
