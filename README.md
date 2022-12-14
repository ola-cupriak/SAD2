# VAE for scRNA-seq data #

## Contents ##
  * [General Description](#general-description)
  * [Requirments](#requirments)
  * [Installation](#installation)
  * [General Information](#general-information)
  * [Usage](#usage)
  * [Configuration File](#configuration-file)
  * [Not Allowed Options](#not-allowed-options)
  * [Project Structure](#project-structure)
  * [Examples](#examples)
  * [MODULE 2](#module-2)
    - [General Description Module 2](#general-description-module-2)
    - [Requirments Module 2](requirments-module-2)
    - [Installation Module 2](#installation-module-2)
    - [Usage Module 2](#usage-module-2)
    - [Input File Module 2](#input-file-module-2)
    - [Examples Module 2](#examples-module-2)

## General Description ##
A project to implement a Variational autoencoder (VAE) for data from the scRNA-seq experiment.

It includes an implementation of a standard VAE with two multivariate normal distributions in Encoder and Decoder (Vanilla VAE) and an implementation of VAE with a modified Decoder - Exponencial Decoder.

Enables latent space analysis using PCA.

## Requirments ##
  * [Python3](https://docs.python-guide.org/starting/install3/linux/)
    
## General Information ##
1. The eval.py file allows to generate all results (plots, tables) based on the passed data and trained models.
2. Python files starting with "train_" are used to train individual models. They enable tuning of hyperparameters.
3. The "train_" scripts save trained models as .pt files and generate plots with learning curves and save them to .png files.
4. The res directory is necessary to save files (models, graphs and tables).
5. Train and test datasets are available here: https://drive.google.com/drive/folders/1yG4o9K38HWmw_7aHbfe-Gkq9XtuH-KMf

## Usage - eval.py##
Usage:

    % python3 eval.py [options]

Options:

    -t [ --train_dataset ] arg      specify path to train dataset (required)
    -v [ --test_dataset ] arg       specify path to test dataset (required)
    -m [ --models ] arg             specify paths to trained models (at least one, may be several separated by a space) (required)

## Usage - train_[model_name].py##
Usage:

    % python3 train_[model_name].py [options]

Options:

    -t [ --train_dataset ] arg      specify path to train dataset (required)
    -v [ --test_dataset ] arg       specify path to test dataset (required)
    -o [ --output ] arg             specify prefix to output files names (required)
    -e [--epochs] arg               specify number of epochs (default=20)
    -bs [--batch_size] arg          specify batch size (default=36)
    -s [--sample] arg               specify sample fraction of data to be used (default=1.0)
    -ld [--latent_dim]              specify latent space dimension (default=100)
    -hd [--hidden_dim]              specify hidden layer dimension (default=250)
    -lr [--learning_rate]           specify learning rate (default=1e-3)
    -b [--beta]                     specify KL divergence weight (default=1.0)
    
## Examples ##
    > python3 eval.py -t data/SAD2022Z_Project1_GEX_train.h5ad -v SAD2022Z_Project1_GEX_test.h5ad -m "vae_exp_s1.0_b1.0_lr0.005_ld50_hd100_bs32_epo20.pt" "vae_vanilla_s1.0_b1.0_lr0.0005_ld50_hd250_bs32_epo20_stats.pt"

    > python3 src/train_VAE_custom.py -t data/SAD2022Z_Project1_GEX_train.h5ad -v data/SAD2022Z_Project1_GEX_test.h5ad -o res/custom_decoder/vae_exp -e 20 -bs 32 -ld 100 -hd 250 -lr 8e-3 -b 1 -s 1

    > python3 src/train_VAE_Vanilla.py -t data/SAD2022Z_Project1_GEX_train.h5ad -v data/SAD2022Z_Project1_GEX_test.h5ad -o res/vae_vanilla/vae_vanilla -e 20 -bs 32 -ld 100 -hd 250 -lr 5e-4 -b 1 -s 1 



