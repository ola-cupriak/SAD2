import sys
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
from utils import parse_model_arguments
from utils import train, test, create_dataloader
from utils import plot_losses
import scanpy as sc
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

    

class EncoderNN_custom(nn.Module):
    """
    Class for the encoder neural network.
    To construction accepts input, latent and hidden dimensions.
    Returns vectors of parameters of normal distribution - 
    - mean and standard deviation.
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int):
        super(EncoderNN_custom, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dim)
        self.linear3 = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x) -> tuple():
        out = F.relu(self.linear1(x))
        mu =  self.linear2(out)
        sigma = F.relu(self.linear3(out)) + 1e-6
        return mu, sigma


class EncoderGaussian_custom(nn.Module):
    """
    Class introduces stochasticity into the encoder.
    To construction accepts encoder neural network.
    Returns latent space vector and vectors of parameters
    of normal distribution.
    """
    def __init__(self, encoder: EncoderNN_custom):
        super(EncoderGaussian_custom, self).__init__()
        self.encoder = encoder
    
    def sample(mu, sigma):
        """
        Samples from normal distribution with parameters from encoder.
        Returns latent space vector.
        """
        q = distributions.Normal(mu, sigma)
        z = q.rsample()
        return z
    
    def log_prob(mu, sigma, z):
        return distributions.Normal(mu, sigma).log_prob(z).sum(dim=(1))
        
    def forward(self, x) -> tuple():
        mu, sigma = self.encoder(x)
        z = EncoderGaussian_custom.sample(mu, sigma)
        return z, mu, sigma


class DecoderNN_custom(nn.Module):
    """
    Class for the decoder neural network.
    To construction accepts input, latent and hidden dimensions.
    Returns vector of mean values of normal distribution.
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int):
        super(DecoderNN_custom, self).__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, z):
        out = F.relu(self.linear1(z))
        rate = F.relu(self.linear2(out)) + 1e-6
        return rate


class DecoderGaussian_custom(nn.Module):
    """
    Class introduces stochasticity into the decoder.
    To construction accepts decoder neural network.
    Returns vector of mean values of normal distribution 
    and vector of reconstruction loss.
    """
    def __init__(self, decoder: DecoderNN_custom):
        super(DecoderGaussian_custom, self).__init__()
        self.decoder = decoder
    
    def log_prob_xz(self, rate, x):
        """
        Measures the logarithm of the probability ofseeing data
        under p(x|z), i.e. the reconstruction loss.
        """
        x = torch.flatten(x, start_dim=1)
        dist = distributions.exponential.Exponential(rate)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1))
     
    def forward(self, z,  x) -> tuple():
        rate = self.decoder(z)
        recon_loss = self.log_prob_xz(rate, x)
        return rate, recon_loss


class VariationalAutoencoder_custom(nn.Module):
    """
    Class for the variational autoencoder.
    To construction accepts encoder, decoder 
    and weight of KL divergance (beta).
    Returns vectors of ELBO loss, KL divergence, 
    reconstruction loss and latent space vector.
    """
    def __init__(self, encoder: EncoderNN_custom, 
                decoder: DecoderNN_custom, beta: float):
        super(VariationalAutoencoder_custom, self).__init__()
        self.encoder = EncoderGaussian_custom(encoder)
        self.decoder = DecoderGaussian_custom(decoder)
        self.beta = beta

    def kl_divergence(self, mu, sigma):
        """
        Measures the KL divergence between the prior 
        and the approximate posterior.
        """
        Dkl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return Dkl

    def sample(self, rate):
        """
        Samples from normal distribution with parameters from decoder.
        Returns vector of x predictions.
        """
        dist = distributions.exponential.Exponential(rate)
        x_hat = dist.sample()
        return x_hat
    
    def forward(self, x) -> tuple():
        z, mu, sigma = self.encoder(x)
        rate, recon_loss = self.decoder(z, x)
        Dkl = self.kl_divergence(mu, sigma)
        elbo = (Dkl * self.beta - recon_loss).mean()
        return elbo, Dkl.mean(), recon_loss.mean(), z


def run_VAE_training(train_data: str, test_data: str, beta: float,
                    learning_rate: float, ldim: int, hdim: int,
                    epochs: int, batch_size: int,
                    sample: float, output: str) -> tuple:
    """
    Runs training of the VAE model.
    Returns lists with values of ELBO loss, KL divergence, 
    reconstruction loss for each epoch and full latent space.
    Saves the model.
    """
    train_dataloader = create_dataloader(train_data, batch_size, sample,
                                        transform=torch.from_numpy)
    test_dataloader = create_dataloader(test_data, batch_size, sample,
                                        transform=torch.from_numpy)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    for i,_,_,_,_ in train_dataloader:
        n_cols = i.shape[1]
        break
    if ldim > n_cols:
        sys.stderr = print('ERROR: Latent dimension cannot be greater\
                            than the number of columns in the dataset.')
        sys.exit(0)

    encoder_nn = EncoderNN_custom(n_cols, ldim, hdim)
    decoder_nn = DecoderNN_custom(n_cols, ldim, hdim)
    vae = VariationalAutoencoder_custom(encoder_nn, decoder_nn, beta)
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
                                                        optimizer, device)
        train_elbo_list.append(elbo_train)
        train_Dkl_list.append(Dkl_train)
        train_recon_loss_list.append(recon_loss_train)
        if t == epochs-1:
            elbo_test, Dkl_test, recon_loss_test, z = test(vae, 
                                                        test_dataloader, 
                                                        True, device)
        else:
            elbo_test, Dkl_test, recon_loss_test, _ = test(vae, 
                                                        test_dataloader, 
                                                        False, device)
        test_elbo_list.append(elbo_test)
        test_Dkl_list.append(Dkl_test)
        test_recon_loss_list.append(recon_loss_test)
    print("Done!")

    torch.save(vae, output+'.pt')

    return (train_elbo_list, train_Dkl_list, train_recon_loss_list, 
            test_elbo_list, test_Dkl_list, test_recon_loss_list, z)


def get_color_dict(path: str, feature_name: str,
                    colors = ['black', 'darkgray', 'rosybrown', 
                            'lightcoral', 'darkred', 'red', 'tan', 
                            'lightsalmon', 'sienna', 'sandybrown', 
                            'peru', 'darkorange', 'gold', 'olive',
                            'darkkhaki', 'yellow', 'forestgreen', 
                            'greenyellow', 'lightgreen', 'lime', 
                            'turquoise', 'teal', 'aqua', 'blue', 
                            'cadetblue', 'deepskyblue', 'steelblue', 
                            'dodgerblue', 'lightsteelblue', 'navy', 
                            'slateblue', 'mediumpurple', 'fuchsia',
                            'rebeccapurple', 'indigo', 'darkviolet',
                            'mediumorchid', 'plum' , 'lightgray', 
                            'mediumvioletred', 'deeppink', 'hotpink',
                            'palevioletred', 'lightpink', 'purple']):
    
    if feature_name not in ['cell_type', 'batch', 'DonorID', 'Site']:
        sys.stderr = print('ERROR: Feature must be one of the following:\
                            cell_type, batch, DonorID, Site.')
        sys.exit(0)

    df = sc.read_h5ad(path)
    feature = list(np.unique(df.obs[feature_name]))
    color_dict = {key: color for key, color in zip(feature, colors)}
    return color_dict


def plot_PCA_latent_space(z, path: str, ldim: int, 
                        colors: dict, feature_name: str):
    """
    Plots the latent space in 2D using PCA.
    Saves the plot.
    Saves a txt file with information on the number of 
    principal components explaining 95% of the variance and returns it.
    """
    if feature_name == 'cell_type':
        k = -4
    elif feature_name == 'batch':
        k = -3
    elif feature_name == 'DonorID':
        k = -2
    elif feature_name == 'Site':
        k = -1
    # PCA
    latent_space = pd.DataFrame(z)
    pca = decomposition.PCA()
    latent_space_std_reg = StandardScaler(
                            ).fit_transform(latent_space.iloc[:,0:ldim])
    pca_res = pca.fit_transform(latent_space_std_reg)
    pca_res = pd.DataFrame(pca_res, columns=[f'S{i}' 
                                    for i in range(1,len(pca_res[0])+1)])
    pca_res[feature_name] = latent_space.iloc[:,k]
    # Plot
    feature = set(pca_res[feature_name])
    group = pca_res[feature_name]

    fig, ax = plt.subplots(figsize=(10,10))
    for g in np.unique(group):
        ix = np.where(group == g)[0].tolist()
        ax.scatter(pca_res['S1'][ix],  pca_res['S2'][ix], 
                    c = colors[g],  label = g, s = 10, alpha=0.7)
    ax.legend(fontsize=5)
    ax.set_title(f'PCA results for ldim={ldim} by {feature_name}')
    plt.savefig(path+f'_PCA_{feature_name}.png', bbox_inches='tight')

    exp_var_pca = pca.explained_variance_ratio_
    var = 0
    i = 0
    for e in exp_var_pca:
        i += 1
        if var>0.95:
            break
        var += e
    with open(path+f'_PCA_{feature_name}.txt', 'w') as f:
        f.write(f'Number of components explaining 95% of variance: {i}\n')
    
    return i

if __name__ == '__main__':
    (train_data, test_data, output, epochs, 
    batch_size, sample, latent_dim, 
    hidden_dim, learning_rate, beta) = parse_model_arguments()

    output += f'_s{sample}_b{beta}_lr{learning_rate}_ld{latent_dim}_hd{hidden_dim}_bs{batch_size}_epo{epochs}'

    (
        train_elbo_list, train_Dkl_list, train_recon_loss_list, 
        test_elbo_list, test_Dkl_list, test_recon_loss_list, z
    ) = run_VAE_training(
                        train_data, test_data, beta, learning_rate, 
                        latent_dim, hidden_dim, epochs, batch_size, 
                        sample, output
                        )

    plot_losses(train_elbo_list, train_Dkl_list, train_recon_loss_list, 
                test_elbo_list, test_Dkl_list, test_recon_loss_list, 
                epochs, output)


    color_dict = get_color_dict(test_data, feature_name='cell_type')
    plot_PCA_latent_space(z, output, latent_dim, color_dict, feature_name='cell_type')