from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from pathlib import Path
import numpy as np
import random
import torch
import sys
import pickle
import os

sys.path.append(Path(__file__).parents[2].as_posix())
from covariate.src.agent import ABC3Agent
from covariate.src.data import load_real_data, train_test_split
from covariate.src.exp import run_regressor


regression_kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1.0)
uncertainty_kernel = RBF(1.0)
dataset = sys.argv[1] # GP, SVM, NN, or RF
model = sys.argv[2]
model_dict = {'GP': GaussianProcessRegressor, 'SVM': SVR, 'NN': MLPRegressor, 'RF': RandomForestRegressor}
if model in ['GP', 'SVM']:
    kernel = True
else:
    kernel = False
print(f'Now running {dataset} {model}')


if __name__ == '__main__':
    random.seed(0)
    ntrials = 50
    sampling = range(10, 101, 10)
    result = dict()
    result['meta'] = {'ntrials': ntrials, 'sampling': sampling}
    result_path = Path(__file__).parents[1] / 'results' / 'regressor' / dataset
    os.makedirs(result_path, exist_ok=True)
    X, Y0, Y1 = load_real_data(dataset)
    for n in range(ntrials):
        print(f'Trial {n+1} / {ntrials}')
        train_data, test_data = train_test_split(X, Y0, Y1, test_ratio=0.5, seed=n)
        train_X = train_data['X'].tolist()
        test_X = test_data['X'].tolist()
        npoints = train_data['X'].shape[0]

        samples_list = []
        for x in sampling:
            samples_list.append(int(x * npoints / 100))
        true_cate = test_data['Y1'] - test_data['Y0']
        true_ate = true_cate.mean()

        torch.cuda.empty_cache()
        agent = ABC3Agent(train_X, test_X, regression_kernel, uncertainty_kernel)
        ate_error, cate_error = run_regressor(agent, model_dict[model], kernel, train_data, samples_list, true_ate, true_cate)

        result[n] = {'ate_error': ate_error,
                     'cate_error': cate_error,
                     'true_ate': np.array(true_ate),
                     'true_cate': np.array(true_cate)}

    with open(result_path / f'{model}.pkl', 'wb') as f:
        pickle.dump(result, f)