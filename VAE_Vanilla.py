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


def parse_arguments() -> tuple:
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
    parser.add_argument('--shuffle', action='store_true', 
                        help='Shuffle data')

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
    shuffle = args.shuffle

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
            latent_dim, hidden_dim, learning_rate, beta, shuffle)
    

class EncoderNN(nn.Module):
    """
    Class for the encoder neural network.
    To construction accepts input, latent and hidden dimensions.
    Returns vectors of parameters of normal distribution - 
    - mean and standard deviation.
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int):
        super(EncoderNN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dim)
        self.linear3 = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x) -> tuple():
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))

        return mu, sigma


class EncoderGaussian(nn.Module):
    """
    Class introduces stochasticity into the encoder.
    To construction accepts encoder neural network.
    Returns latent space vector and vectors of parameters
    of normal distribution.
    """
    def __init__(self, encoder: EncoderNN):
        super(EncoderGaussian, self).__init__()
        self.encoder = encoder
    
    def sample(mu, std):
        """
        Samples from normal distribution with parameters from encoder.
        Returns latent space vector.
        """
        q = distributions.Normal(mu, std)
        z = q.rsample()
        return z
    
    def log_prob(mu, 
                std, 
                z
                ):
        return distributions.Normal(mu, std).log_prob(z).sum(dim=(1))
        
    def forward(self, x) -> tuple():
        mu, log_var = self.encoder(x)
        std = torch.exp(log_var / 2)
        z = EncoderGaussian.sample(mu, std)
        return z, mu, std


class DecoderNN(nn.Module):
    """
    Class for the decoder neural network.
    To construction accepts input, latent and hidden dimensions.
    Returns vector of mean values of normal distribution.
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int):
        super(DecoderNN, self).__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, z):
        out = F.relu(self.linear1(z))
        out = torch.sigmoid(self.linear2(out))
        return out


class DecoderGaussian(nn.Module):
    """
    Class introduces stochasticity into the decoder.
    To construction accepts decoder neural network.
    Returns vector of mean values of normal distribution 
    and vector of reconstruction loss.
    """
    def __init__(self, decoder: DecoderNN):
        super(DecoderGaussian, self).__init__()
        self.decoder = decoder
        self.log_variance = nn.Parameter(torch.Tensor([0.0]))
    
    def log_prob_xz(self, mean, 
                    log_variance, 
                    x
                    ):
        """
        Measures the logarithm of the probability of seeing data under p(x|z), 
        i.e. the reconstruction loss.
        """
        x = torch.flatten(x, start_dim=1)
        variance = torch.exp(log_variance)
        dist = distributions.Normal(mean, variance)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1))
    
    def forward(self, 
                z, 
                x
                ) -> tuple():
        out = self.decoder(z)
        recon_loss = self.log_prob_xz(out, self.log_variance, x)
        return out, recon_loss


class VariationalAutoencoder(nn.Module):
    """
    Class for the variational autoencoder.
    To construction accepts encoder, decoder 
    and weight of KL divergance (beta).
    Returns vectors of ELBO loss, KL divergence, 
    reconstruction loss and latent space vector.
    """

    def __init__(self, encoder: EncoderNN, 
                decoder: DecoderNN, beta: float):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = EncoderGaussian(encoder)
        self.decoder = DecoderGaussian(decoder)
        self.beta = beta

    def kl_divergence(self, 
                    mu, 
                    sigma
                    ):
        """
        Measures the KL divergence between the prior 
        and the approximate posterior.
        """
        Dkl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return Dkl

    def sample(self, 
            mean, 
            log_variance
            ):
        """
        Samples from normal distribution with parameters from decoder.
        Returns vector of x predictions.
        """
        variance = torch.exp(log_variance)
        dist = distributions.Normal(mean, variance)
        x_hat = dist.sample()
        return x_hat
    
    def forward(self, x) -> tuple():
        z, mu, sigma = self.encoder(x)
        decoder_out, recon_loss = self.decoder(z, x)
        Dkl = self.kl_divergence(mu, sigma)
        elbo = (Dkl * self.beta - recon_loss).mean()
        return elbo, Dkl.mean(), recon_loss.mean(), z


def train(VAE: VariationalAutoencoder, 
        dataloader, 
        optimizer, 
        device='cpu') -> tuple:
    """
    Trains the VAE model.
    Returns means of ELBO loss, KL divergence 
    and reconstruction loss for one epoch.
    """
    device = torch.device(device)
    VAE = VAE.to(device)
    train_elbo = 0
    train_Dkl = 0
    train_recon_loss = 0
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    for batch, (X, _) in enumerate(dataloader):
        X = X.to(device)
        elbo, Dkl, recon_loss, _ = VAE(X)
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


def test(VAE: VariationalAutoencoder, 
        dataloader, 
        device='cpu') -> tuple:
    """
    Tests the VAE model.
    Returns means of ELBO loss, KL divergence, reconstruction loss 
    and latent space vector for one epoch.
    """
    device = torch.device(device)
    VAE = VAE.to(device)
    num_batches = len(dataloader)
    test_elbo = 0
    test_Dkl = 0
    test_recon_loss = 0
    z_full = None

    with torch.no_grad():
        for batch, (X, cell_type) in enumerate(dataloader):
            X = X.to(device)
            elbo, Dkl, recon_loss, z = VAE(X)
            z = z.detach().numpy()
            z = np.c_[z, cell_type]
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

        if self.transform:
            single_cell = self.transform(single_cell)

        return single_cell, single_cell_type


def create_dataloader_dict(train_file: str, 
                        test_file: str, 
                        batch_size: int, 
                        shuffle: bool,
                        sample: float, 
                        transform=None
                        ) -> dict:
    """
    Creates dataloaders for train and test datasets.
    Returns dictionary with dataloaders.
    """
    training_data = scRNADataset(train_file, sample, transform)
    test_data = scRNADataset(test_file, sample, transform)
    train_dataloader = DataLoader(training_data, batch_size, shuffle)
    test_dataloader = DataLoader(test_data, batch_size, shuffle)

    return {'train': train_dataloader,
            'val': test_dataloader}


def run_VAE_training(train_data: str, test_data: str,
                    beta: float, learning_rate: float,
                    ldim: int, hdim: int,
                    epochs: int, batch_size: int,
                    shuffle: bool, sample: float,
                    output: str
                    ) -> tuple:
    """
    Runs training of the VAE model.
    Returns lists with values of ELBO loss, KL divergence, 
    reconstruction loss for each epoch and full latent space.
    Saves the model.
    """
    dataloader_dict = create_dataloader_dict(train_data, test_data, 
                                            batch_size, shuffle, sample,
                                            transform=torch.from_numpy)
    train_dataloader = dataloader_dict['train']
    test_dataloader = dataloader_dict['val']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    for i,_ in train_dataloader:
        n_cols = i.shape[1]
        break
    if ldim > n_cols:
        sys.stderr = print('ERROR: Latent dimension cannot be greater\
                            than the number of columns in the dataset.')
        sys.exit(0)

    encoder_nn = EncoderNN(n_cols, ldim, hdim)
    decoder_nn = DecoderNN(n_cols, ldim, hdim)
    vae = VariationalAutoencoder(encoder_nn, decoder_nn, beta)
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    train_elbo_list = []
    train_Dkl_list = []
    train_recon_loss_list = []
    test_elbo_list = []
    test_Dkl_list = []
    test_recon_loss_list = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        elbo_train, Dkl_train, recon_loss_train, = train(vae, 
                                                        train_dataloader, 
                                                        optimizer)
        train_elbo_list.append(elbo_train)
        train_Dkl_list.append(Dkl_train)
        train_recon_loss_list.append(recon_loss_train)
        
        elbo_test, Dkl_test, recon_loss_test, z = test(vae, 
                                                    test_dataloader)

        test_elbo_list.append(elbo_test)
        test_Dkl_list.append(Dkl_test)
        test_recon_loss_list.append(recon_loss_test)
    print("Done!")

    torch.save(vae, output)

    return (train_elbo_list, train_Dkl_list, train_recon_loss_list, 
            test_elbo_list, test_Dkl_list, test_recon_loss_list, 
            z)


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
    test_elbo_list_np = [float(obs.detach().numpy()) 
                        for obs in test_elbo_list]
    train_elbo_list_np = [float(obs.detach().numpy()) 
                        for obs in train_elbo_list]
    train_Dkl_list_np = [float(obs.detach().numpy()) 
                        for obs in train_Dkl_list]
    test_Dkl_list_np = [float(obs.detach().numpy()) 
                        for obs in test_Dkl_list]
    train_recon_loss_list_np = [float(obs.detach().numpy()) 
                                for obs in train_recon_loss_list]
    test_recon_loss_list_np = [float(obs.detach().numpy()) 
                                for obs in test_recon_loss_list]

    axes[0].plot(epochs_list, train_elbo_list_np, label='train')
    axes[0].plot(epochs_list, test_elbo_list_np, label='test')
    axes[0].set_title('Train and test losses WITH BETA for each epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (beta-ELBO)')
    axes[0].legend()

    axes[1].plot(epochs_list, train_Dkl_list_np, label='train')
    axes[1].plot(epochs_list, test_Dkl_list_np, label='test')
    axes[1].set_title('Train and test Dkl for each epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dkl')
    axes[1].legend()

    axes[2].plot(epochs_list, train_recon_loss_list_np, label='train')
    axes[2].plot(epochs_list, test_recon_loss_list_np, label='test')
    axes[2].set_title('Train and test recon losses for each epoch')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Recon loss')
    axes[2].legend()

    plt.savefig(output+'.png')


def plot_PCA_latent_space(z, output: str, ldim: int,
                        colors = ['black', 'darkgray', 'rosybrown', 
                                'lightcoral', 'darkred', 'red', 
                                'lightsalmon', 'sienna', 'sandybrown', 
                                'peru', 'darkorange', 'gold', 'olive',
                                'darkkhaki', 'yellow', 'forestgreen', 
                                'greenyellow', 'lightgreen', 'lime', 
                                'turquoise', 'teal', 'aqua', 'blue', 
                                'cadetblue', 'deepskyblue', 'steelblue', 
                                'dodgerblue', 'lightsteelblue', 'navy', 
                                'slateblue', 'mediumpurple', 'fuchsia',
                                'rebeccapurple', 'indigo', 'darkviolet',
                                'mediumorchid', 'plum' , 'purple', 
                                'mediumvioletred', 'deeppink', 'hotpink',
                                'palevioletred', 'lightpink']):
    """
    Plots the latent space in 2D using PCA.
    Saves the plot.
    Saves a txt file with information on the number of 
    principal components explaining 95% of the variance.
    """
    # PCA
    latent_space = pd.DataFrame(z)
    pca = decomposition.PCA()
    latent_space_std_reg = StandardScaler(
                            ).fit_transform(latent_space.iloc[:,0:ldim])
    pca_res = pca.fit_transform(latent_space_std_reg)
    pca_res = pd.DataFrame(pca_res, columns=[f'S{i}' 
                                    for i in range(1,len(pca_res[0])+1)])
    pca_res['cell_type'] = latent_space.iloc[:,ldim]
    # Plot
    cell_types = set(pca_res['cell_type'])
    group = pca_res['cell_type']
    color_dict = {key: color for key, color in zip(cell_types, colors)}

    fig, ax = plt.subplots(figsize=(10,10))
    for g in np.unique(group):
        ix = np.where(group == g)[0].tolist()
        ax.scatter(pca_res['S1'][ix],  pca_res['S2'][ix], c = color_dict[g], label = g, s = 10, alpha=0.7)
    ax.legend(fontsize=5)
    plt.savefig(output+'_PCA.png')

    exp_var_pca = pca.explained_variance_ratio_
    var = 0
    i = 0
    for e in exp_var_pca:
        i += 1
        if var>0.95:
            break
        var += e

    with open(output+'_PCA.txt', 'w') as f:
        f.write(f'Number of components explaining 95% of variance: {i}\n')


if __name__ == '__main__':
    (train_data, test_data, output, epochs, 
    batch_size, sample, latent_dim, 
    hidden_dim, learning_rate, beta, shuffle) = parse_arguments()

    (
        train_elbo_list, train_Dkl_list, train_recon_loss_list, 
        test_elbo_list, test_Dkl_list, test_recon_loss_list, z
    ) = run_VAE_training(
                        train_data, test_data, beta, learning_rate, 
                        latent_dim, hidden_dim, epochs, batch_size, 
                        shuffle, sample, output
                        )

    plot_losses(train_elbo_list, train_Dkl_list, train_recon_loss_list, 
                test_elbo_list, test_Dkl_list, test_recon_loss_list, 
                epochs, output)

    plot_PCA_latent_space(z, output, latent_dim)