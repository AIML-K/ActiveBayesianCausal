from sklearn.gaussian_process.kernels import RBF
from pathlib import Path
import numpy as np
import pickle
import sys
import os

sys.path.append(Path(__file__).parents[2].as_posix())
from covariate.src.data import load_real_data, train_test_split
from covariate.src.exp import check_assumption


kernel = RBF(1.0)


if __name__ == '__main__':
    result_path = Path(__file__).parents[1] / 'results' / 'assumption'
    os.makedirs(result_path, exist_ok=True)
    dataset_list = ['ihdp', 'boston', 'lalonde', 'acic']
    for dataset in dataset_list:
        X, Y0, Y1 = load_real_data(dataset)
        print(f'Checking Assumption')
        train_data, test_data = train_test_split(X, Y0, Y1, test_ratio=0.0, seed=0)
        X = train_data['X'].tolist()
        npoints = train_data['X'].shape[0]

        delta, epsilon = check_assumption(kernel, X)

        result = {'npoints': npoints, 'delta': delta, 'epsilon': epsilon}
        with open(result_path / f'{dataset}.pkl', 'wb') as f:
            pickle.dump(result, f)
            