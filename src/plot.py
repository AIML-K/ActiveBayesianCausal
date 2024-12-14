import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pickle
import os
import sys


plt.style.use('seaborn-v0_8-white')
LINE = {0: '-', 1: '--', 2: ':', 3: '-.', 4: '-.', 5: '-', 6: '--', 7: ':', 8: '-.', 9: '--'}


def plot_assumption():
    result_path = Path(__file__).parents[1] / 'results' / 'assumption'
    plot_path = Path(__file__).parents[1] / 'plots' / 'assumption'
    os.makedirs(plot_path, exist_ok=True)
    datasets = os.listdir(result_path)
    for dataset in datasets:
        with open(result_path / dataset, 'rb') as f:
            data = pickle.load(f)
        argmin = (data['delta'] * np.array(2) - data['epsilon']).argmin(axis=0)
        two_delta = [(data['delta'] * np.array(2))[argmin[i], i] for i in range(data['delta'].shape[1])]
        epsilon = [data['epsilon'][argmin[i], i] for i in range(data['epsilon'].shape[1])]
        plt.plot(range(data['npoints']), two_delta, label='$\mathbf{2\delta^*(I_n)}$', linewidth=12.0)
        plt.plot(range(data['npoints']), epsilon , label='$\mathbf{\epsilon^*(I_n)}$', linewidth=12.0)
        plt.xticks(fontsize='large')
        plt.grid()
        plt.title(dataset.replace('.pkl', '').upper(), size=45, weight='bold')
        plt.ylim([-0.1, 0.15])
        if dataset == 'ihdp.pkl':
            plt.ylim([-0.02, 0.02])
        plt.savefig(plot_path / dataset.replace('.pkl', '.png'))
        plt.close()


def plot_sampling(dataset):
    sampling_path = Path(__file__).parents[1] / 'results' / 'sampling'
    plot_path = Path(__file__).parents[1] / 'plots' / 'sampling'
    os.makedirs(plot_path, exist_ok=True)
    sampling_files = sorted([i for i in os.listdir(sampling_path)])
    sampling_files = ['Naive null.pkl', 'Sample 10.pkl', 'Sample 20.pkl', 'Opt 10.pkl', 'Opt 20.pkl']
    up, down = 0, np.inf

    for n, result in enumerate(sampling_files):
        cate = list()
        with open(sampling_path / result, 'rb') as f:
            data = pickle.load(f)
        for it in range(data['meta']['ntrials']):
            cate.append(data[it]['cate_error'])
        cate = np.array(cate)
        cate_mean = cate.mean(axis=0)
        cate_std = cate.std(axis=0)
        if result == 'Naive null.pkl':
            label = result.replace(' null.pkl', '').capitalize()
        elif 'Opt' in result:
            label = f"ABC3 (N={result[-7:-4].strip('t ')})"
        elif 'Sample' in result:
            label = f"Sample (N={result[-6:-4].strip()})"
        plt.plot(data['meta']['sampling'], cate_mean, label=label, linestyle=LINE[n], linewidth=4.0)
        plt.fill_between(data['meta']['sampling'], cate_mean - cate_std, cate_mean + cate_std, alpha=.1)
        up = max(cate_mean.max(), up)
        down = min(cate_mean.min(), down)

    down = -0.5    
    plt.ylim(down * 0.9, up * 1.05)
    plt.xticks(fontsize='large')
    plt.xlabel('Percentage of population', fontsize='xx-large', weight='bold')
    plt.ylabel(r'$\mathbf{\epsilon}_{\mathbf{PEHE}}$', fontsize='xx-large', weight='bold')
    plt.title(dataset.upper().replace('/1', ''), size=30, weight='bold')
    plt.legend(loc=1, prop={'size': 17, 'weight':'bold'})
    plt.grid()
    plt.savefig(plot_path / f'sampling.png')
    plt.close()


def plot_hyperparameter_kernel(dataset):
    result_path = Path(__file__).parents[1] / 'results' / 'hyperparameter_kernel' / dataset
    plot_path = Path(__file__).parents[1] / 'plots' / 'hyperparameter_kernel'
    os.makedirs(plot_path, exist_ok=True)
    result_files = sorted(['Matern_[0.5, 0.5].pkl', 'Matern_[0.5, 1.5].pkl', 'Matern_[1.0, 0.5].pkl', 
                           'Matern_[1.0, 1.5].pkl', 'RBF_0.5.pkl', 'RBF_1.0.pkl', 'RBF_1.5.pkl', 'RBF_2.0.pkl',
                           'Sine_0.1.pkl', 'Sine_1.0.pkl'])
    for n, result in enumerate(result_files):
        cate = list()
        with open(result_path / result, 'rb') as f:
            data = pickle.load(f)
        for it in range(data['meta']['ntrials']):
            cate.append(data[it])
        cate = np.array(cate)
        cate_mean = cate.mean(axis=0)
        label = result.replace('.pkl', ')').replace('_', '(')
        plt.plot(data['meta']['sampling'], cate_mean, label=label, linestyle=LINE[n], linewidth=4.0)
    plt.xticks(fontsize='large')
    plt.xlabel('Percentage of population', fontsize='xx-large', weight='bold')
    plt.ylabel(r'$\mathbf{\epsilon}_{\mathbf{PEHE}}$', fontsize='xx-large', weight='bold')
    plt.legend(loc=1, prop={'size': 12, 'weight':'bold'})
    plt.grid()
    plt.title(dataset.upper().replace('/1', ''), size=30, weight='bold')
    plt.savefig(plot_path / f'{dataset}.png')
    plt.close()


def plot_hyperparameter_sigma(dataset):
    result_path = Path(__file__).parents[1] / 'results' / 'hyperparameter_sigma' / dataset
    plot_path = Path(__file__).parents[1] / 'plots' / 'hyperparameter_sigma'
    os.makedirs(plot_path, exist_ok=True)
    result_files = sorted(os.listdir(result_path))
    for n, result in enumerate(result_files):
        cate = list()
        with open(result_path / result, 'rb') as f:
            data = pickle.load(f)
        for it in range(data['meta']['ntrials']):
            cate.append(data[it])
        cate = np.array(cate)
        cate_mean = cate.mean(axis=0)
        cate_std = cate.std(axis=0)
        label = result.replace('.pkl', '')
        plt.plot(data['meta']['sampling'], cate_mean, label=label, linestyle=LINE[n], linewidth=4.0)
    plt.xticks(fontsize='large')
    plt.xlabel('Percentage of population', fontsize='xx-large', weight='bold')
    plt.ylabel(r'$\mathbf{\epsilon}_{\mathbf{PEHE}}$', fontsize='xx-large', weight='bold')
    plt.legend(loc=1, prop={'size': 17, 'weight':'bold'})
    plt.grid()
    plt.title(dataset.upper().replace('/1', ''), size=30, weight='bold')
    plt.savefig(plot_path / f'{dataset}.png')
    plt.close()


def plot_cate_error(dataset):
    result_path = Path(__file__).parents[1] / 'results' / 'benchmark' / dataset
    plot_path = Path(__file__).parents[1] / 'plots' / 'cate_error'
    os.makedirs(plot_path, exist_ok=True)
    result_files = sorted(os.listdir(result_path))
    up, down = 0, np.inf
    for n, result in enumerate(result_files):
        cate = list()
        with open(result_path / result, 'rb') as f:
            data = pickle.load(f)
        for it in range(data['meta']['ntrials']):
            cate.append(data[it]['cate_error'])
        cate = np.array(cate)
        cate_mean = cate.mean(axis=0)
        label = result.replace('.pkl', '')
        plt.plot(data['meta']['sampling'], cate_mean, label=label, linestyle=LINE[n], linewidth=4.0)
        if 'Mackay' not in result:
            up = max(cate_mean.max(), up)
        down = min(cate_mean.min(), down)
    plt.ylim(down * 0.95, up * 1.05)
    plt.xticks(fontsize='large')
    plt.xlabel('Percentage of population', fontsize='xx-large', weight='bold')
    plt.ylabel(r'$\mathbf{\epsilon}_{\mathbf{PEHE}}$', fontsize='xx-large', weight='bold')
    plt.title(dataset.upper().replace('/1', ''), size=30, weight='bold')
    plt.legend(loc=1, prop={'size': 17, 'weight':'bold'})
    plt.grid()
    plt.savefig(plot_path / f'{dataset}.png')
    plt.close()


def plot_mmd(dataset):
    result_path = Path(__file__).parents[1] / 'results' / 'benchmark' / dataset
    plot_path = Path(__file__).parents[1] / 'plots' / 'mmd'
    os.makedirs(plot_path, exist_ok=True)
    result_files = ['ABC3.pkl', 'Naive.pkl']
    up, down = 0, np.inf
    for n, result in enumerate(result_files):
        mmd = list()
        with open(result_path / result, 'rb') as f:
            data = pickle.load(f)
        for it in range(data['meta']['ntrials']):
            mmd.append(data[it]['mmd'])
        mmd = np.array(mmd)
        mmd_mean = mmd.mean(axis=0)
        mmd_std = mmd.std(axis=0)
        label = result.replace('.pkl', '')
        plt.plot(data['meta']['sampling'], mmd_mean, label=label, linewidth=12.0)
        plt.fill_between(data['meta']['sampling'], mmd_mean - mmd_std, mmd_mean + mmd_std, alpha=.1)
        up = max(mmd_mean.max(), up)
        down = min(mmd_mean.min(), down)
    plt.ylim(down * 0.95, up * 1.05)
    plt.xticks(fontsize='large')
    plt.grid()
    plt.title(dataset.upper().replace('/1', ''), size=45, weight='bold')
    plt.savefig(plot_path / f'{dataset}.png')
    plt.close()


def plot_type1_error(dataset):
    alpha = 1.96
    result_path = Path(__file__).parents[1] / 'results' / 'benchmark' / dataset
    plot_path = Path(__file__).parents[1] / 'plots' / 'type1'
    os.makedirs(plot_path, exist_ok=True)
    result_files = sorted(os.listdir(result_path))
    upper, lower = 0, np.inf
    for n, result in enumerate(result_files):
        type1 = list()
        with open(result_path / result, 'rb') as f:
            data = pickle.load(f)
        for it in range(data['meta']['ntrials']):
            zscore = np.array(data[it]['mu']) / np.array(data[it]['std'])
            error_rate = (abs(zscore) >= alpha).mean(axis=1)
            type1.append(error_rate)
        type1 = np.array(type1)
        type1_mean = type1.mean(axis=0)
        label = result.replace('.pkl', '')
        plt.plot(data['meta']['sampling'], type1_mean, label=label, linestyle=LINE[n], linewidth=4.0)
        upper = max(type1_mean.max(), upper)
        lower = min(type1_mean.min(), lower)
    plt.ylim(lower * 0.95, upper * 1.05)
    plt.xticks(fontsize='large')
    plt.xlabel('Percentage of population', fontsize='xx-large', weight='bold')
    plt.ylabel('Type 1 Error (%)', fontsize='xx-large', weight='bold')
    plt.legend(loc=1, prop={'size': 17, 'weight':'bold'})
    plt.grid()
    plt.title(dataset.upper(), size=30, weight='bold')
    plt.savefig(plot_path / f'{dataset}.png')
    plt.close()


def plot_regressor(dataset):
    result_path = Path(__file__).parents[1] / 'results' / 'regressor' / dataset
    plot_path = Path(__file__).parents[1] / 'plots' / 'regressor'
    os.makedirs(plot_path, exist_ok=True)
    result_files = sorted(os.listdir(result_path))
    up, down = 0, np.inf
    for n, result in enumerate(result_files):
        cate = list()
        with open(result_path / result, 'rb') as f:
            data = pickle.load(f)
        for it in range(data['meta']['ntrials']):
            cate.append(data[it]['cate_error'])
        cate = np.array(cate)
        cate_mean = cate.mean(axis=0)
        cate_std = cate.std(axis=0)
        label = result.replace('.pkl', '')
        plt.plot(data['meta']['sampling'], cate_mean, label=label, linestyle=LINE[n], linewidth=4.0)
        plt.fill_between(data['meta']['sampling'], cate_mean - cate_std, cate_mean + cate_std, alpha=.1)
        up = max(cate_mean.max(), up)
        down = min(cate_mean.min(), down)
    if dataset == 'lalonde':
        down = -1000000
    plt.ylim(down * 0.95, up * 1.05)
    plt.xticks(fontsize='large')
    plt.xlabel('Percentage of population', fontsize='xx-large', weight='bold')
    plt.ylabel(r'$\mathbf{\epsilon}_{\mathbf{PEHE}}$', fontsize='xx-large', weight='bold')
    plt.legend(loc=1, prop={'size': 17, 'weight':'bold'})
    plt.grid()
    plt.title(dataset.upper().replace('/1', ''), size=30, weight='bold')
    plt.savefig(plot_path / f'{dataset}.png')
    plt.close()


if __name__ == "__main__":
    todo = sys.argv[1]
    dataset = sys.argv[2]
    if todo == 'kernel':
        plot_hyperparameter_kernel(dataset)
    elif todo == 'sigma':
        plot_hyperparameter_sigma(dataset)
    elif todo == 'sampling':
        plot_sampling(dataset)
    elif todo == 'reg':
        plot_regressor(dataset)
    elif todo == 'assumption':
        plot_assumption()
    else:
        plot_cate_error(dataset)
        plot_mmd(dataset)
        if dataset in ['boston', 'lalonde']:
            plot_type1_error(dataset)