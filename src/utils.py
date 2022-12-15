#!/usr/bin/env python3
'''Script contains common functions for other scripts'''

import anndata
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import scanpy as sc
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import sys
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def parse_model_arguments() -> tuple:
    """
    Parse command line arguments.
    Returns tuple of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_data', type=str, required=True, 
                        help='Path to training dataset')
    parser.add_argument('-v', '--test_data', type=str, required=True, 
                        help='Path to test dataset')
    parser.add_argument('-o', '--output', type=str, required=True, 
                        help='Name of output files')
    parser.add_argument('-e', '--epochs', type=int, default=20, 
                        help='Number of epochs')
    parser.add_argument('-bs', '--batch_size', type=int, default=36, 
                        help='Batch size')
    parser.add_argument('-s', '--sample', type=float, default=1.0, 
                        help='Sample fraction of data')
    parser.add_argument('-ld', '--latent_dim', type=int, default=100, 
                        help='Latent dimension')
    parser.add_argument('-hd', '--hidden_dim', type=int, default=250, 
                        help='Hidden dimension')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, 
                        help='Learning rate')
    parser.add_argument('-b', '--beta', type=float, default=1.0, 
                        help='KL divergence weight')

    args = parser.parse_args()

    train_data = args.train_data
    test_data = args.test_data
    output = args.output
    epochs = args.epochs
    batch_size = args.batch_size
    sample = args.sample
    latent_dim = args.latent_dim
    hidden_dim = args.hidden_dim
    learning_rate = args.learning_rate
    beta = args.beta

    if not os.path.exists(train_data):
        sys.stderr = print('ERROR: Training data not found.')
        sys.exit(0)
    if not os.path.exists(test_data):
        sys.stderr = print('ERROR: Test data not found.')
        sys.exit(0)
    if hidden_dim < latent_dim:
        sys.stderr = print('ERROR: Latent dimension must be smaller\
                             than hidden dimension.')
        sys.exit(0)
    if sample < 0 or sample > 1:
        sys.stderr = print('ERROR: Sample fraction must be between \
                            0 and 1.')
        sys.exit(0)

    return (train_data, test_data, output, epochs, batch_size, sample,
            latent_dim, hidden_dim, learning_rate, beta)


def train(model, dataloader, optimizer, device='cpu') -> tuple:
    """
    Trains the passed model.
    Returns means of ELBO loss, KL divergence 
    and reconstruction loss for one epoch.
    """
    device = torch.device(device)
    model = model.to(device)
    train_elbo = 0
    train_Dkl = 0
    train_recon_loss = 0
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    for batch, (X, _, _, _, _) in enumerate(dataloader):
        X = X.to(device)
        elbo, Dkl, recon_loss, _ = model(X)
        # Backpropagation
        optimizer.zero_grad()
        elbo.backward()
        optimizer.step()
        train_elbo += elbo
        train_Dkl += Dkl
        train_recon_loss += recon_loss
        
        if batch % 100 == 0:
            elbo, current = elbo.item(), batch * len(X)
            print(f"elbo: {elbo:>7f}  [{current:>5d}/{size:>5d}]")
    
    return (train_elbo/num_batches, 
            train_Dkl/num_batches, 
            train_recon_loss/num_batches)


def test(model, dataloader, last_epoch: bool, device='cpu') -> tuple: 
    """
    Tests the passed model.
    Returns means of ELBO loss, KL divergence, reconstruction loss 
    and latent space vector for one epoch.
    """
    device = torch.device(device)
    model = model.to(device)
    num_batches = len(dataloader)
    test_elbo = 0
    test_Dkl = 0
    test_recon_loss = 0
    z_full = None

    with torch.no_grad():
        for batch, (X, cell_type, s_batch, donorID, site) in enumerate(dataloader):
            X = X.to(device)
            elbo, Dkl, recon_loss, z = model(X)
            if last_epoch:
                z = z.cpu().detach().numpy()
                z = np.c_[z, cell_type, s_batch, donorID, site]
                if batch == 0:
                    z_full = z
                else:
                    z_full = np.concatenate((z_full, z), axis=0)
            test_elbo += elbo
            test_Dkl += Dkl
            test_recon_loss += recon_loss
    return (test_elbo/num_batches, 
            test_Dkl/num_batches, 
            test_recon_loss/num_batches, 
            z_full)


class scRNADataset(Dataset):
    """
    Class for the scRNA-seq dataset.
    Accepts path to the dataset, fraction of the dataset to be used 
    and transform function.
    Returns single observation from the dataset 
    as a vector of values and a cell type.
    """
    def __init__(self, h5ad_file: str, sample: float, transform=None):
        self.data = sc.read_h5ad(h5ad_file)
        self.cell_type = np.array(self.data.obs.cell_type)
        self.batch = np.array(self.data.obs.batch)
        self.DonorID = np.array(self.data.obs.DonorID)
        self.Site = np.array(self.data.obs.Site)
        self.data = self.data.layers['counts'].toarray()
        l_cells = round(len(self.data) * sample)
        self.data = self.data[:l_cells]
        self.data = self.data/np.max(self.data)
        self.transform = transform   


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        single_cell = self.data[idx]
        single_cell_type = self.cell_type[idx]
        single_batch = self.batch[idx]
        single_DonorID = self.DonorID[idx]
        single_Site = self.Site[idx]
    
        if self.transform:
            single_cell = self.transform(single_cell)
    
        return (single_cell, single_cell_type, 
                single_batch, single_DonorID, single_Site)


def create_dataloader(file: str, batch_size: int, sample: float, 
                        transform=None) -> dict:
    """
    Creates dataloader for passed datasets and returns it.
    """
    data = scRNADataset(file, sample, transform)
    dataloader = DataLoader(data, batch_size, num_workers=4)

    return dataloader


def plot_losses(train_elbo_list: list, train_Dkl_list: list, 
            train_recon_loss_list: list, test_elbo_list: list,
            test_Dkl_list: list, test_recon_loss_list: list,
            epochs: int, output: str):
    """
    Plots ELBO loss, KL divergence and reconstruction loss.
    Saves the plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(30,10))

    epochs_list = [t+1 for t in range(epochs)]
    test_elbo_list_np = [float(obs.cpu().detach().numpy()) 
                        for obs in test_elbo_list]
    train_elbo_list_np = [float(obs.cpu().detach().numpy()) 
                        for obs in train_elbo_list]
    train_Dkl_list_np = [float(obs.cpu().detach().numpy()) 
                        for obs in train_Dkl_list]
    test_Dkl_list_np = [float(obs.cpu().detach().numpy()) 
                        for obs in test_Dkl_list]
    train_recon_loss_list_np = [float(obs.cpu().detach().numpy()) 
                                for obs in train_recon_loss_list]
    test_recon_loss_list_np = [float(obs.cpu().detach().numpy()) 
                                 for obs in test_recon_loss_list]

    axes[0].plot(epochs_list, train_elbo_list_np, label='train')
    axes[0].plot(epochs_list, test_elbo_list_np, label='test')
    axes[0].set_title('Train and test losses for each epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (beta-ELBO)')
    if len(epochs_list) <= 20:
        axes[0].set_xticks(epochs_list)
    axes[0].legend()

    axes[1].plot(epochs_list, train_Dkl_list_np, label='train')
    axes[1].plot(epochs_list, test_Dkl_list_np, label='test')
    axes[1].set_title('Train and test Dkl for each epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dkl')
    if len(epochs_list) <= 20:
        axes[1].set_xticks(epochs_list)
    axes[1].legend()

    axes[2].plot(epochs_list, train_recon_loss_list_np, label='train')
    axes[2].plot(epochs_list, test_recon_loss_list_np, label='test')
    axes[2].set_title('Train and test reconstruction losses for each epoch')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Recon loss')
    if len(epochs_list) <= 20:
        axes[2].set_xticks(epochs_list)
    axes[2].legend()
    
    plt.savefig(output+'.png', bbox_inches='tight')