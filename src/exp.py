from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import numpy as np
from tqdm import tqdm


def compute_mmd_rbf(X, Y, gamma=1.0):
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


@ignore_warnings(category=ConvergenceWarning)
def run_regressor(agent, regressor, need_kernel, train_data, samples_list, true_ate, true_cate):
    allocation = list(range(samples_list[-1]))
    ate_error, cate_error = list(), list()
    for i, id in enumerate(tqdm(allocation)):
        idx, treatment = agent.treatment()
        if idx is None:
            idx = id
        obs = train_data[['Y0', 'Y1'][treatment]].iloc[idx].item()
        cov = train_data['X'].iloc[idx]
        agent.update(treatment, np.array(cov), obs, idx)
        if i + 1 in samples_list:
            if regressor == GaussianProcessRegressor:
                model0 = regressor(agent.regression_kernel, normalize_y=True)
                model1 = regressor(agent.regression_kernel, normalize_y=True)
            elif need_kernel:
                model0 = regressor(kernel=agent.regression_kernel)
                model1 = regressor(kernel=agent.regression_kernel)
            else:
                model0 = regressor()
                model1 = regressor()
            model0.fit(agent.x0, agent.y0)
            model1.fit(agent.x1, agent.y1)
            pred = model1.predict(agent.test_covariates) - model0.predict(agent.test_covariates)
            ate_error.append((pred.mean() - true_ate) ** 2)
            cate_error.append(((pred - true_cate) ** 2).mean())
    return ate_error, cate_error


def run_experiment(agent, train_data, samples_list, true_ate, true_cate, policy=None):
    allocation = list(range(samples_list[-1]))
    if policy == 'Naive':
        np.random.shuffle(allocation)
    allocation = tqdm(allocation, desc=f'Running {policy}')
    ate_error, cate_error = list(), list()
    mus, stds, mmds = list(), list(), list()
    for i, id in enumerate(allocation):
        idx, treatment = agent.treatment()
        if idx is None:
            idx = id
        obs = train_data[['Y0', 'Y1'][treatment]].iloc[idx].item()
        cov = train_data['X'].iloc[idx]
        agent.update(treatment, np.array(cov), obs, idx)
        if i + 1 in samples_list:
            agent.fit()
            ate_hat, cate_hat = agent.ate_cate()
            mu, std = agent.cate_credible_interval()
            ate_error.append((ate_hat - true_ate) ** 2)
            cate_error.append(((cate_hat - true_cate) ** 2).mean())
            mmd = compute_mmd_rbf(agent.x0, agent.x1)
            mus.append(mu)
            stds.append(std)
            mmds.append(mmd)
    return ate_error, cate_error, np.array(mus), np.array(stds), np.array(mmds)


def run_experiment_simple(agent, train_data, samples_list, true_ate, true_cate, policy=None):
    allocation = list(range(samples_list[-1]))
    if policy == 'Naive':
        np.random.shuffle(allocation)
    allocation = tqdm(allocation, desc=f'Running {policy}')
    ate_error, cate_error = list(), list()
    for i, id in enumerate(allocation):
        idx, treatment = agent.treatment()
        if idx is None:
            idx = id
        obs = train_data[['Y0', 'Y1'][treatment]].iloc[idx].item()
        cov = train_data['X'].iloc[idx]
        agent.update(treatment, np.array(cov), obs, idx)
        if i + 1 in samples_list:
            agent.fit()
            ate_hat, cate_hat = agent.ate_cate()
            ate_error.append((ate_hat - true_ate) ** 2)
            cate_error.append(((cate_hat - true_cate) ** 2).mean())
    return ate_error, cate_error


def check_assumption(kernel, covariates):
    normalizer = StandardScaler()
    covariates = normalizer.fit_transform(covariates)
    all_d1s = list()
    all_means = list()
    for i in tqdm(range(100)):
        means = list()
        d1s = list()
        np.random.seed(i)
        np.random.shuffle(covariates)
        Kfull = kernel(covariates)
        for i in range(Kfull.shape[0]):
            means.append(Kfull[:i+1,:i+1].mean())
            d1s.append(Kfull[:i+1,:].mean())
        all_means.append(means)
        all_d1s.append(d1s)
    M = Kfull.mean()
    epsilon = M - np.array(all_means)
    return np.array(all_d1s), epsilon