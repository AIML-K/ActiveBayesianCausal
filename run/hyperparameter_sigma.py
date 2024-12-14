from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF, Matern, ExpSineSquared
from pathlib import Path
import random
import torch
import sys
import pickle
import os

sys.path.append(Path(__file__).parents[2].as_posix())
from covariate.src.agent import ABC3Agent
from covariate.src.data import load_real_data, train_test_split
from covariate.src.exp import run_experiment


regression_kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1.0)
uncertainty_kernel = RBF(1.0)

dataset = sys.argv[1]
sigma0 = float(sys.argv[2])
sigma1 = float(sys.argv[3])

print(f'Now running {dataset} {sigma0}:{sigma1}')


if __name__ == '__main__':
    random.seed(0)
    ntrials = 50
    sampling = range(10, 101, 10)
    result = dict()
    result['meta'] = {'ntrials': ntrials, 'sampling': sampling}
    result_path = Path(__file__).parents[1] / 'results' / 'hyperparameter_sigma' / dataset
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
        agent = ABC3Agent(train_X, test_X, regression_kernel, uncertainty_kernel, sigma0=sigma0, sigma1=sigma1)
        ate_error, cate_error, _, _, _ = run_experiment(agent, train_data, samples_list, true_ate, true_cate, policy='ABC3')
        result[n] = cate_error

    with open(result_path / f'{sigma0}:{sigma1}.pkl', 'wb') as f:
        pickle.dump(result, f)