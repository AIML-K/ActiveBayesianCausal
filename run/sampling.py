from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF
from pathlib import Path
import numpy as np
import random
import torch
import sys
import pickle
import time
import os

sys.path.append(Path(__file__).parents[2].as_posix())
from covariate.src.agent import ABC3Opt, ABC3Sample, ABC3Agent, NaiveAgent
from covariate.src.data import load_real_data, train_test_split
from covariate.src.exp import run_experiment_simple


regression_kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1.0)
uncertainty_kernel = RBF(1.0)

method = sys.argv[1] # Naive, Sample, or Opt (Ours)
n_sampling = sys.argv[2] # null for Naive
print(f'Now running {method} {n_sampling}')


if __name__ == '__main__':
    random.seed(0)
    ntrials = 10
    sampling = np.array(range(1, 11, 1)) / 10
    result = dict()
    result['meta'] = {'ntrials': ntrials, 'sampling': sampling}
    result_path = Path(__file__).parents[1] / 'results' / 'sampling'
    os.makedirs(result_path, exist_ok=True)
    for n in range(ntrials):
        X, Y0, Y1 = load_real_data('weather')
        print(f'Trial {n+1} / {ntrials}')
        train_data, test_data = train_test_split(X, Y0, Y1, test_ratio=0.1, seed=n)
        train_X = train_data['X'].tolist()
        test_X = test_data['X'].tolist()
        npoints = train_data['X'].shape[0]
        
        samples_list = []
        for x in sampling:
            samples_list.append(int(x * npoints / 100))
        true_cate = test_data['Y1'] - test_data['Y0']
        true_ate = true_cate.mean()
        
        torch.cuda.empty_cache()
        if method == 'ABC3':
            agent = ABC3Agent(train_X, test_X, regression_kernel, uncertainty_kernel)
        elif method == 'Naive':
            agent = NaiveAgent(train_X, test_X, regression_kernel)
        elif method == 'Opt':
            agent = ABC3Opt(train_X, test_X, regression_kernel, uncertainty_kernel, n_sampling=int(n_sampling))
        elif method == 'Sample':
            agent = ABC3Sample(train_X, test_X, regression_kernel, uncertainty_kernel, n_sampling=int(n_sampling))
        t1 = time.time()
        if method == 'Naive':
            ate_error, cate_error = run_experiment_simple(agent, train_data, samples_list, true_ate, true_cate, policy='Naive')
        else:
            ate_error, cate_error = run_experiment_simple(agent, train_data, samples_list, true_ate, true_cate, policy='ABC3')
        t2 = time.time()
        result[n] = {'time': t2 - t1, 'cate_error': cate_error}

        with open(result_path / f'{method} {n_sampling}.pkl', 'wb') as f:
            pickle.dump(result, f)