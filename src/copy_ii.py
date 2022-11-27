import anndata
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import sys
import os

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
    args = parser.parse_args()
    train_set = args.train_dataset
    test_set = args.test_dataset

    datasets = {'train dataset': train_set, 'test dataset': test_set}
    
    return datasets


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
    print(f'"{output}" has been created.')


def to_dense(data_dict: dict) -> list:
    """
    Changes sparse matrixes to dense matrixes and returns a list of them.
    """
    new_dict = dict()
    for name, data in data_dict.items():
        new_dict[f'preprocessed {name}'] = data.X.toarray().reshape(-1)
        new_dict[f'raw {name}'] = data.layers['counts'].toarray().reshape(-1)

    return new_dict


def create_stats_table(data_dict: dict, output: str, include_zeros: bool):
    """
    Creates DataFrame with information about the shape of 
    the passed datasets and saves them to a csv file.
    Input data provided in the form of a dict {dataset name: dataset}.
    """
    if not include_zeros:
        data_dict = {name: data[data != 0 ] for name, data in data_dict.items()}

    mins = [data.min() for data in data_dict.values()]
    maxs = [data.max() for data in data_dict.values()]
    medians = [np.median(data) for data in data_dict.values()]
    means = [data.mean() for data in data_dict.values()]
    summary = pd.DataFrame(zip(mins, maxs, means, medians), 
            index=[name for name in data_dict.keys()], 
            columns=['min', 'max', 'mean', 'median'])
    
    summary.to_csv(output)
    print(f'"{output}" has been created.')


def plot_histogram(data_dict: dict, output: str, include_zeros: bool,
                    bins_prepro: list, bins_raw: list, 
                    alpha=0.3, ec="gray", fc="blue"):
    """
    Creates histograms using the matplotlib library for the passed
    data (raw and preprocessed) using the passed settings.
    Saves it in png file.
    """
    if not include_zeros:
        data_dict = {name: data[data != 0 ] for name, data in data_dict.items()}

    _, axes = plt.subplots(2, 2, figsize=(15,15))

    for i, (name, data) in enumerate(data_dict.items()):
        idx = [(0,0), (0,1), (1,0), (1,1)]
        if 'preprocessed' in name:
            axes[idx[i][0],idx[i][1]].hist(data, bins=bins_prepro, ec=ec, fc=fc, alpha=alpha)
            axes[idx[i][0],idx[i][1]].set_xlabel('preprocessed counts')
        else:
            axes[idx[i][0],idx[i][1]].hist(data, bins=bins_raw, ec=ec, fc=fc, alpha=alpha)
            axes[idx[i][0],idx[i][1]].set_xlabel('Raw counts')
        axes[idx[i][0],idx[i][1]].set_title(name)
        axes[idx[i][0],idx[i][1]].set_ylabel('frequency')

    plt.savefig(output, bbox_inches='tight')
    print(f'"{output}" has been created.')



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
    print(f'"{output}" has been created.')



if __name__ == '__main__':
    #datasets = read_cmd()
    datasets = {'train dataset': 'data/SAD2022Z_Project1_GEX_train.h5ad', 'test dataset': 'data/SAD2022Z_Project1_GEX_test.h5ad'}
    datasets['train dataset'] = load_dataset(datasets['train dataset'])
    datasets['test dataset'] = load_dataset(datasets['test dataset'])
    create_shape_table(datasets, 'results/shape_table.csv')
    get_obs_info(datasets, 'results/info_table.csv')
    datasets = to_dense(datasets)
    # Generate statistics
    create_stats_table(datasets, 'results/stats_with_zeros.csv', True)
    create_stats_table(datasets, 'results/stats_without_zeros.csv', False)
    # Generate histograms
    bins_prepro = np.arange(0, 10.5, 0.5)
    bins_raw = np.arange(0, 11, 1)
    plot_histogram(datasets, 'results/hists_with_zeros.png', True,
                    bins_prepro, bins_raw)
    plot_histogram(datasets, 'results/hists_without_zeros.png', False,
                    bins_prepro, bins_raw)
    