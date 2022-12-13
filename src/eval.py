import anndata
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import sys
import os
import torch
from utils import create_dataloader, test
from VAE_custom_exp import VariationalAutoencoder_custom
from VAE_custom_exp import EncoderNN_custom, DecoderNN_custom
from VAE_custom_exp import EncoderGaussian_custom, DecoderGaussian_custom
from VAE_Vanilla import VariationalAutoencoder
from VAE_Vanilla import EncoderGaussian, DecoderGaussian
from VAE_Vanilla import EncoderNN, DecoderNN



def read_cmd() -> tuple:
    """
    Reads the arguments passed to run the program 
    and returns them as a tuple.
    """
    parser = argparse.ArgumentParser()
    # Reads the arguments
    parser.add_argument('-t', '--train_dataset', 
                        help='path to training dataset', 
                        type=str, required=True)
    parser.add_argument('-v', '--test_dataset',
                        help='path to test dataset', 
                        type=str, required=True)
    parser.add_argument('-m', '--models', 
                        help='list of models to be evaluated',
                        type=str, nargs='+', required=True)
    train_set = args.train_dataset
    test_set = args.test_dataset
    models = args.models
    # Checks the correctness of the passed arguments
    if not os.path.exists(train_set):
        sys.stderr(print('ERROR: Incorrect path for \
            the file containing the train dataset.'))
        sys.exit(0)
    if not os.path.exists(test_set):
        sys.stderr(print('ERROR: Incorrect path for \
            the file containing the test dataset.'))
        sys.exit(0)

    datasets = {'train dataset': train_set, 'test dataset': test_set}
    
    return datasets, models


def load_dataset(path: str) -> sc.AnnData:
    """
    Reads the data saved in the h5ad file.
    """
    try:
        data = sc.read_h5ad(path)
    except:
        sys.stderr(print("ERROR: Incorrect input file."))
        sys.exit(0)

    return data


def create_shape_table(data_dict: dict, output: str):
    """
    Creates DataFrame with information about the shape of 
    the passed datasets and saves them to a csv file.
    Input data provided in the form of a dict {dataset name: dataset}.
    """
    n_obs = [data.n_obs for data in data_dict.values()]
    n_vars = [data.n_vars for data in data_dict.values()]
    names = [name for name in data_dict.keys()]
    summary = pd.DataFrame(zip(n_obs, n_vars), index=names, 
            columns=['Number of observations', 'Number of variables'])
    summary.to_csv(output)


def create_stats_table(data_list: list, output: str):
    """
    Creates DataFrame with information about the shape of 
    the passed datasets and saves them to a csv file.
    Input data provided in the form of a dict {dataset name: dataset}.
    """
    mins = [data.min() for data in data_list]
    maxs = [data.max() for data in data_list]
    medians = [np.median(data) for data in data_list]
    means = [data.mean() for data in data_list]
    summary = pd.DataFrame(zip(mins, maxs, means, medians), 
            index=['preprocessed train data', 'raw train data', 
                    'preprocessed test data', 'raw test data'], 
            columns=['min', 'max', 'mean', 'median'])
    
    summary.to_csv(output)


def plot_histogram(data_dict: dict, output_hists: str, output_stats: str, 
                    include_zeros: bool, bins_prepro: list, bins_raw: list, 
                    alpha=0.3, ec="gray", fc="blue"):
    """
    Creates histograms using the matplotlib library for the passed
    data (raw and preprocessed) using the passed settings.
    Saves it in png file.
    """
    # Data formatting
    data_X_train = data_dict['train dataset'].X.toarray().reshape(-1)
    data_raw_train = data_dict['train dataset'].layers['counts'].toarray().reshape(-1)
    data_X_test = data_dict['test dataset'].X.toarray().reshape(-1)
    data_raw_test = data_dict['test dataset'].layers['counts'].toarray().reshape(-1)

    if not include_zeros:
        data_X_train = data_X_train[data_X_train != 0 ]
        data_raw_train = data_raw_train[data_raw_train != 0 ]
        data_X_test = data_X_test[data_X_test != 0 ]
        data_raw_test = data_raw_test[data_raw_test != 0 ]

    # Create stats
    create_stats_table([data_X_train, data_raw_train, 
                        data_X_test, data_raw_test], output_stats)

    # Creates histograms
    _, axes = plt.subplots(2, 2, figsize=(15,15))

    axes[0,0].hist(data_X_train, bins=bins_prepro, ec=ec, fc=fc, alpha=alpha)
    axes[0,0].set_title('preprocessed train data')
    axes[0,0].set_xlabel('data values')
    axes[0,0].set_ylabel('frequency')


    axes[0,1].hist(data_raw_train, bins=bins_raw, ec=ec, fc=fc, alpha=alpha)
    axes[0,1].set_title('raw train data')
    axes[0,1].set_xlabel('data values')
    axes[0,1].set_ylabel('frequency')
    axes[0,1].set_xticks(bins_raw)

    axes[1,0].hist(data_X_test, bins=bins_prepro, ec=ec, fc=fc, alpha=alpha)
    axes[1,0].set_title('preprocessed test data')
    axes[1,0].set_xlabel('data values')
    axes[1,0].set_ylabel('frequency')

    axes[1,1].hist(data_raw_test, bins=bins_raw, ec=ec, fc=fc, alpha=alpha)
    axes[1,1].set_title('raw test data')
    axes[1,1].set_xlabel('data values')
    axes[1,1].set_ylabel('frequency')
    axes[1,1].set_xticks(bins_raw)

    plt.savefig(output_hists, bbox_inches='tight')


def get_obs_info(data_dict: dict, output: str):
    """
    Gets information about observations in the passed data 
    and saves it to a csv file.
    Input data provided in the form of a dict {dataset name: dataset}.
    """
    patients_num = [len(data.obs["DonorID"].unique()) 
                    for data in data_dict.values()]
    labs_num = [len(data.obs["Site"].unique()) 
                for data in data_dict.values()]
    cell_types_num = [len(data.obs["cell_type"].unique()) 
                        for data in data_dict.values()]
    names = [name for name in data_dict.keys()]
    columns = ['Number of patients', 
                'Number of labs', 
                'Number of cell types']
    summary = pd.DataFrame(zip(patients_num, labs_num, cell_types_num), 
                            index=names, columns=columns)
    summary.to_csv(output)


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
        if feature_name == 'DonorID':
            g = int(g)
        ix = np.where(group == g)[0].tolist()
        ax.scatter(pca_res['S1'][ix],  pca_res['S2'][ix], 
                    c = colors[g],  label = g, s = 10, alpha=0.7)
    if feature_name == 'cell_type':
        ax.legend(fontsize=5)
    else:
        ax.legend()
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

    
    return i


if __name__ == '__main__':
    #datasets, models = read_cmd()
    datasets_paths = {'train dataset': 'data/SAD2022Z_Project1_GEX_train.h5ad', 'test dataset': 'data/SAD2022Z_Project1_GEX_test.h5ad'}
    models = ['res/vae_vanilla/vae_vanilla_s1.0_b1.0_lr0.0005_ld50_hd250_bs32_epo20.pt']
    datasets = {}
    datasets['train dataset'] = load_dataset(datasets_paths['train dataset'])
    datasets['test dataset'] = load_dataset(datasets_paths['test dataset'])
    create_shape_table(datasets, 'res/shape_table.csv')
    # # Generate histograms
    # bins_prepro_0 = np.arange(0, 10.5, 0.5)
    # bins_raw_0 = np.arange(0, 11, 1)
    # bins_prepro = np.arange(0, 20.5, 0.5)
    # bins_raw = np.arange(0, 21, 1)
    # plot_histogram(datasets, 'res/hists_with_zeros.png', 
    #                 'res/stats_with_zeros.csv', True,
    #                 bins_prepro_0, bins_raw_0)
    # plot_histogram(datasets, 'res/hists_without_zeros.png', 
    #                 'res/stats_without_zeros.csv', False,
    #                 bins_prepro, bins_raw)
    # get_obs_info(datasets, 'res/info_table.csv')

    # Generate latent spaces on the test dataset for each model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features = ['cell_type', 'batch', 'DonorID', 'Site']
    color_dicts = {f: get_color_dict(datasets_paths['test dataset'], f) 
                    for f in features}

    for model in models:
        vae = torch.load(model)
        vae.eval()
        batch_size = int(model.split('_bs')[1].split('_')[0])
        sample = float(model.split('_s')[1].split('_')[0])
        ldim = vae.encoder.encoder.linear3.out_features
        model = model.split('.')[:-1]
        model = '.'.join(model)
        print(model)
        dataloader = create_dataloader(datasets_paths['test dataset'], 
                                        batch_size, sample, 
                                        transform=torch.from_numpy)
        elbo, Dkl, recon, z = test(vae, dataloader, True, device)
        # Plot latent space for each feature
        stats = [float(elbo.cpu()), float(Dkl.cpu()), float(recon.cpu())]
        for f in features:
            pca = plot_PCA_latent_space(z, model, ldim, color_dicts[f], f)
            stats.append(pca)
        stats = [stats]
        stats = pd.DataFrame(stats, columns=['-ELBO', 'Dkl', 'Recon', 
                                            'PCA_cell_type', 'PCA_batch', 
                                            'PCA_DonorID', 'PCA_Site'])
        stats.to_csv(model+f'_stats.csv', index=False)
