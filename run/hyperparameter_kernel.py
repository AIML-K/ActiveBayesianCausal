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
kernel_dict = {'RBF': RBF, 'Matern': Matern, 'Sine': ExpSineSquared}
dataset = sys.argv[1]
kernel_name = sys.argv[2] # RBF, Matern, or Sine
params = eval(sys.argv[3]) # For RBF and Sine, input scalar value. For Matern, input list of scalars. e.g. [1.0,1.0]
if kernel_name != 'Matern':
    uncertainty_kernel = kernel_dict[kernel_name](params)
else:
    uncertainty_kernel = kernel_dict[kernel_name](length_scale=params[0], nu=params[1])
print(f'Now running {kernel_name} {params}')


if __name__ == '__main__':
    random.seed(0)
    ntrials = 50
    sampling = range(10, 101, 10)
    result = dict()
    result['meta'] = {'ntrials': ntrials, 'sampling': sampling}
    result_path = Path(__file__).parents[1] / 'results' / 'hyperparameter_kernel' / dataset
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
        ate_error, cate_error, _, _, _ = run_experiment(agent, train_data, samples_list, true_ate, true_cate, policy='ABC3')
        result[n] = cate_error

    with open(result_path / f'{kernel_name}_{params}.pkl', 'wb') as f:
        pickle.dump(result, f)