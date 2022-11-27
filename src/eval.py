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
    # Checks the correctness of the passed arguments
    if not os.path.exists(train_set):
        sys.stderr(print('ERROR: Incorrect path for the file containing the train dataset.'))
        sys.ecit(0)
    if not os.path.exists(test_set):
        sys.stderr(print('ERROR: Incorrect path for the file containing the test dataset.'))
        sys.ecit(0)

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
    axes[0,0].set_xlabel('preprocessed counts')
    axes[0,0].set_ylabel('frequency')

    axes[0,1].hist(data_raw_train, bins=bins_raw, ec=ec, fc=fc, alpha=alpha)
    axes[0,1].set_title('raw train data')
    axes[0,1].set_xlabel('raw counts')
    axes[0,1].set_ylabel('frequency')

    axes[1,0].hist(data_X_test, bins=bins_prepro, ec=ec, fc=fc, alpha=alpha)
    axes[1,0].set_title('preprocessed test data')
    axes[1,0].set_xlabel('preprocessed counts')
    axes[1,0].set_ylabel('frequency')

    axes[1,1].hist(data_raw_test, bins=bins_raw, ec=ec, fc=fc, alpha=alpha)
    axes[1,1].set_title('raw test data')
    axes[1,1].set_xlabel('raw counts')
    axes[1,1].set_ylabel('frequency')

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


if __name__ == '__main__':
    datasets = read_cmd()
    datasets['train dataset'] = load_dataset(datasets['train dataset'])
    datasets['test dataset'] = load_dataset(datasets['test dataset'])
    create_shape_table(datasets, 'results/shape_table.csv')
    # Generate histograms
    bins_prepro = np.arange(0, 10.5, 0.5)
    bins_raw = np.arange(0, 11, 1)
    plot_histogram(datasets, 'results/hists_with_zeros.png', 
                    'results/stats_with_zeros.csv', True,
                    bins_prepro, bins_raw)
    # plot_histogram(datasets, 'results/hists_without_zeros.png', 
    #                 'results/stats_without_zeros.csv', False,
    #                 bins_prepro, bins_raw)
    get_obs_info(datasets, 'results/info_table.csv')